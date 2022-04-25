/**
Author: Chris Henn (https://github.com/chrishenn)

Implements the Fixed-Radius Nearest-Neighbor (frnn) algorithm [Qianli Liao and David Walter]

 modified to support two
 object-groups, where edges are only found between members of different groups.

 Parallel exclusive scan code adapted from [Matt Dean - 1422434 - mxd434].
**/

#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>

#include "frnn_bin_kern.h"
#include "frnn_bipart_kern.h"
#include "scan.h"


// define for error checking
//#define CUDA_ERROR_CHECK

// function prototypes
double get_nanos();

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    do{
        cudaError err = cudaGetLastError();
        if ( cudaSuccess != err )
        {
            fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                     file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }

        err = cudaDeviceSynchronize();
        if( cudaSuccess != err )
        {
            fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                     file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }
    } while(0);
#endif

    return;
}

// timing helper
double get_nanos() {
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (double)ts.tv_nsec;
}






/** 
cpu entry point for frnn_bipart python extension.
**/
std::vector<torch::Tensor> frnn_bipart_cuda_call(
    torch::Tensor pts,
    torch::Tensor imgid,

    torch::Tensor lin_radius,
    torch::Tensor scale_radius,

    torch::Tensor batch_size
) {

    using namespace torch::indexing;
    auto device_id = pts.get_device();
    cudaSetDevice(device_id);

    // Device data options
    auto options = torch::TensorOptions()
            .dtype(torch::kInt32)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device_id)
            .requires_grad(false);

    auto l_options = torch::TensorOptions()
            .dtype(torch::kInt64)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device_id)
            .requires_grad(false);

    auto f_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device_id)
            .requires_grad(false);


    // sepcific to the size of pts used with frnn
    auto pts_x_col = torch::full({1}, 1, l_options);
    auto pts_scale_col = torch::full({1}, 4, l_options);

    // Allocate host data
    frnn_bipart_kern* FRNN_BIPART = new frnn_bipart_kern();
    scan *SCAN = new scan();
    frnn_bin_kern *NEB = new frnn_bin_kern();

    // Setup device sizes
    auto pts_size0 = pts.size(0);
    auto pts_size1 = pts.size(1);
    const dim3 blocks(256);
    const dim3 threads(256);

    // Prepare supporting data for frnn
    auto x = pts.index({Slice(), pts_x_col});
    auto xmin = x.min();
    auto xmax = x.max();
    auto xdiff = (xmax - xmin).to(torch::kF32);

    auto scales_log = pts.index({Slice(), pts_scale_col}).log();
    auto logscalemin = scales_log.min();
    auto logscalemax = scales_log.max();
    auto logscalediff = logscalemax - logscalemin;
    auto n_scabin = (logscalediff / scale_radius) + 1;

    auto bin_logscales = torch::arange(n_scabin.item<int>(), f_options).add(1).mul(scale_radius);
    auto scabin_linrads = bin_logscales.exp().mul(lin_radius + 0.15);

    auto n_xbin_eachscale = xdiff.div(2 * scabin_linrads).to(torch::kInt32).add(1);
    auto n_xbin = n_xbin_eachscale.sum();
    auto n_binstotal = n_xbin * batch_size;

    auto glob_to_im_i = torch::arange(n_binstotal.item<int>(), f_options).fmod(n_xbin.item<float>()).to(torch::kInt32).unsqueeze(1);

    if (n_scabin.lt(2).item<int>()) n_xbin_eachscale.unsqueeze_(0);
    auto n_xbin_eachscale_incume = n_xbin_eachscale.cumsum(0).to(torch::kI32);
    auto n_xbin_eachscale_excume = n_xbin_eachscale_incume.sub(n_xbin_eachscale);

    auto glob_to_scabin_m0 = glob_to_im_i.ge(n_xbin_eachscale_excume);
    auto glob_to_scabin_m1 = glob_to_im_i.lt(n_xbin_eachscale_incume);
    auto glob_to_scabin_m = glob_to_scabin_m0.__and__(glob_to_scabin_m1);
    auto glob_to_scabin = torch::arange(n_xbin_eachscale_incume.size(0), options).masked_select(glob_to_scabin_m);

    glob_to_im_i.squeeze_();
    n_xbin_eachscale_incume.squeeze_();
    n_xbin_eachscale_excume.squeeze_();
    scabin_linrads.squeeze_();
    bin_logscales.squeeze_();

    // allocate device data structures
    auto bin_ptids = torch::empty({pts_size0}, options);
    auto bin_wcounts = torch::zeros({n_binstotal.item<int>(), 1}, options);

    auto neighb_size1 = torch::full({1}, 20, options);
    auto neighbors = torch::empty({n_binstotal.item<int>(), neighb_size1.item<int>()}, options);
    auto neb_counts = torch::empty({n_binstotal.item<int>()}, options);

    auto bin_counts = torch::zeros({n_binstotal.item<int>()}, options);
    auto bin_offsets = torch::empty({n_binstotal.item<int>()}, options);
    auto bid2ptid = torch::empty({pts_size0, 2}, options);
    auto write_count = torch::zeros({1}, options);

    // calculate output size
    auto intermed = (lin_radius * 4 + 1).to(torch::kInt32);
    auto area = intermed.pow(2).to(torch::kFloat32);
    auto n_average = torch::true_divide(torch::full({1}, pts_size0, options) , batch_size);
    auto hyp = (area * n_average * batch_size * 16 * scale_radius).to(torch::kI32);
    auto sq_max = (n_average.pow(2) * batch_size).to(torch::kI32);

    auto edges_size0 = torch::min(hyp, sq_max);
    auto edges_size1 = torch::full({1}, 2, options);

    auto edges = torch::empty({edges_size0.item<int>(), edges_size1.item<int>()}, l_options);
    auto glob_count = torch::zeros({1}, options);

    // Calculate neighbor bin_ids (neighbors); count writes to neighbor rows in neb_counts. Central bins are written to column 0 of neighbors.
    // Locations in neighbors without a bin_id are set to -1.
    NEB->frnn_neighbors_launch(
        neighbors.data_ptr<int>(),
        neb_counts.data_ptr<int>(),

        scabin_linrads.data_ptr<float>(),
        n_xbin_eachscale.data_ptr<int>(),
        n_xbin_eachscale_excume.data_ptr<int>(),
        glob_to_scabin.data_ptr<int>(),
        glob_to_im_i.data_ptr<int>(),

        xmin.item<float>(),
        xmax.item<float>(),

        n_scabin.item<int>(),
        n_xbin.item<int>(),
        n_binstotal.item<int>(),
        batch_size.item<int>(),
        neighb_size1.item<int>(),

        blocks,
        threads
    );
    CudaCheckError();

//     prune redundant bin pairs by row in neighbors; shuffle valid neighbors to lower columns; adjust neb_counts
	NEB->frnn_neighbors_prune_launch(
		neighbors.data_ptr<int>(),
		neb_counts.data_ptr<int>(),

		n_binstotal.item<int>(),
		neighb_size1.item<int>(),
		blocks,
		threads
	);
    CudaCheckError();

//     count number of pts in each bin, write to bin_counts; build bid2ptid (matching a bin_id to a pt_id by row)
    NEB->frnn_bin_counts_launch(
        pts,
        imgid.data_ptr<long>(),
        scabin_linrads.data_ptr<float>(),
        n_xbin_eachscale_excume.data_ptr<int>(),
        bin_counts.data_ptr<int>(),
        bid2ptid.data_ptr<int>(),
        write_count.data_ptr<int>(),

        scale_radius.item<float>(),
        xmin.item<float>(),
        logscalemin.item<float>(),
        pts_size0,
        pts_size1,
        int(n_binstotal.item<int>() / batch_size.item<int>()),
        n_binstotal.item<int>(),
        pts_x_col.item<int>(),
        pts_scale_col.item<int>(),

        blocks,
        threads
    );
    CudaCheckError();

//     exclusive-scan bin_counts; write to bin_offsets
    SCAN->scan_launch(  bin_counts.data_ptr<int>(),
                        bin_offsets.data_ptr<int>(),
                        n_binstotal.item<int>());

    CudaCheckError();

//     build bin_ptids from bid2ptid, bin_offsets. bin_ptids give contiguous indexes into pts                    
//     for a given bin, where a bin's pt indexes start at bin_offsets[bin_id].
    NEB->frnn_ptids_launch(
        bin_counts.data_ptr<int>(),
        bin_offsets.data_ptr<int>(),
        bid2ptid.data_ptr<int>(),
        bin_ptids.data_ptr<int>(),
        bin_wcounts.data_ptr<int>(),

        pts_size0,
        blocks,
        threads
    );
    CudaCheckError();

    // each block stores 6 float32's per point, up to max_binsize points per bin, and 2 bins in shared memory
    auto bin_stride = torch::full({1}, 6, l_options);
    auto max_binsize = bin_counts.max();
    auto shared = 2 * bin_stride * max_binsize * long(sizeof(float));

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    if ( shared.item<int>() >= deviceProp.sharedMemPerBlock ){
        fprintf (stderr, "ERROR FRNN_BIPART_MAIN_KERNEL: attempted to allocate %li bytes of shared mem per block, but the current device arch\n", shared.item<long>());
        fprintf (stderr, "supports a maximum of %lu per block; to fix, reduce batch size or search radii.\n", deviceProp.sharedMemPerBlock-1);
        exit(EXIT_FAILURE);
    }

    // compare bins of points on frnn criteria, write indexes of pts with an edge between them into 'edges'. 
    // Count number of integer writes in glob_counts (glob_counts= 2xnumber_edges_written).
    // frnn_bipart_kern needs a threadblock for at least each valid bin_id in neighbors
    FRNN_BIPART->frnn_bipart_kern_launch(
        neighbors.data_ptr<int>(),
        bin_counts.data_ptr<int>(),
        bin_offsets.data_ptr<int>(),
        bin_ptids.data_ptr<int>(),
        pts,

        glob_count.data_ptr<int>(),
        edges.data_ptr<long>(),

        lin_radius.item<float>(),
        scale_radius.item<float>(),

        max_binsize.item<int>(),
        bin_stride.item<int>(),

        int(neighbors.size(1)),
        int(pts.size(1)),

        dim3(neb_counts.max().item<int>(), n_binstotal.item<int>()),
        dim3(16,16),

        shared.item<long>()
    );

    // free host objects' memory
    free(FRNN_BIPART);
    free(SCAN);
    free(NEB);

    auto edge_count = glob_count.floor_divide(2);
    if ( edge_count.gt(edges_size0).item<int>() ){
        fprintf (stderr, "ERROR frnn_bipart_driver.cu: FRNN_BIPART_MAIN_KERNEL ATTEMPTED TO WRITE TOO MANY EDGES; INCREASE ALLOCATED EDGES SIZE OR DECREASE BATCH SIZE");
        exit(EXIT_FAILURE);
    }

    edges = edges.narrow(0, 0, edge_count.item<int>());

    return {edges};

//    return {edges, glob_count, neighbors, bid2ptid, neb_counts,bin_counts,bin_offsets, bin_ptids, max_binsize, shared, n_binstotal,
//            neighb_size1, bin_stride, lin_radius, scale_radius };
}
