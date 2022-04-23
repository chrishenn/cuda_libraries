/**
Authors: Christian Henn, Qianli Liao

Implements the Fixed-Radius Nearest-Neighbor (frnn) algorithm [Qianli Liao and David Walter]. Parallel exclusive scan code adapted from [Matt Dean - 1422434 - mxd434].

This file relies on Pytorch calls from the Pytorch C++ API. For non-Pytorch implementation, use frnn_driver_CUDA.cu .
Supporting Pytorch code to compute input structures can be found in frnn_bipart_bind.py .
**/

#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>

#include "frnn_bin_kern.h"
#include "frnn_kern.h"
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
cpu entry point for frnn python extension.

Parameters:
    pts:        (n x 6); torch.float32; [y,x,z,angle,scale,ptid]
    imgid:    (n x 1); torch.int32; maps object to its image

    batch_size: 

Returns:
    edges:      (m x 2); torch.int32; each row gives two indexes into pts
    glob_counts:    (1); torch.int32; twice the number of contiguous rows written to edges
**/
std::vector<torch::Tensor> frnn_cuda_call(
    torch::Tensor pts,
    torch::Tensor imgid,

    int batch_size,

    float lin_radius,
    float scale_radius
){
    // specific to the size of pts used with frnn_opt
    const int pts_x_col = 1;
    const int pts_scale_col = 2;
    
    using namespace torch::indexing;
    auto device_id = pts.get_device();
    cudaSetDevice(device_id);

    // Device data options
    auto options = torch::TensorOptions()
            .dtype(torch::kInt32)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device_id);

    auto f_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device_id);

    // Allocate host data
    frnn_bipart_kern* FRNN = new frnn_bipart_kern();
    scan* SCAN = new scan();
    frnn_bin_kern* NEB = new frnn_bin_kern();

    // Setup device sizes
    int pts_size0 = pts.size(0);
    int pts_size1 = pts.size(1);
    const dim3 blocks(256);
    const dim3 threads(256);

    // Prepare supporting data for frnn
    auto x = pts.index({Slice(), 1});
    auto xmin = x.min();
    auto xmax = x.max();
    auto xdiff = xmax - xmin;

    auto scales_log = pts.index({Slice(), 2}).log();
    auto logscalemin = scales_log.min();
    auto logscalemax = scales_log.max();
    auto logscalediff = logscalemax - logscalemin;
    int n_scabin = (logscalediff / scale_radius).item<int>() + 1;

    torch::Tensor bin_logscales = torch::arange(n_scabin, options).add(1).mul(scale_radius);
    torch::Tensor scabin_linrads = bin_logscales.exp().mul(lin_radius +0.15);

    torch::Tensor n_xbin_eachscale = xdiff.div(2 * scabin_linrads).to(torch::kInt32).add(1);
    int n_xbin = n_xbin_eachscale.sum().item<int>();
    int n_binstotal = n_xbin * batch_size;

    torch::Tensor glob_to_im_i = torch::arange(n_binstotal, f_options).fmod(n_xbin).to(torch::kInt32).unsqueeze(1);

    if (n_scabin < 2) n_xbin_eachscale.unsqueeze_(0);
    torch::Tensor n_xbin_eachscale_incume = n_xbin_eachscale.cumsum(0).to(torch::kInt32);
    torch::Tensor n_xbin_eachscale_excume = n_xbin_eachscale_incume.sub(n_xbin_eachscale);

    torch::Tensor glob_to_scabin_m0 = glob_to_im_i.ge(n_xbin_eachscale_excume);
    torch::Tensor glob_to_scabin_m1 = glob_to_im_i.lt(n_xbin_eachscale_incume);
    torch::Tensor glob_to_scabin_m = glob_to_scabin_m0.__and__(glob_to_scabin_m1);
    torch::Tensor glob_to_scabin = torch::arange(n_xbin_eachscale_incume.size(0), options).masked_select( glob_to_scabin_m );

    glob_to_im_i.squeeze_();
    n_xbin_eachscale_incume.squeeze_();
    n_xbin_eachscale_excume.squeeze_();
    scabin_linrads.squeeze_();
    bin_logscales.squeeze_();

    // allocate device data structures
    torch::Tensor bin_ptids = torch::empty({pts_size0, 1}, options);
    torch::Tensor bin_wcounts = torch::zeros({n_binstotal, 1}, options);

    const int neighb_size1 = 20;
    torch::Tensor neighbors = torch::empty({n_binstotal, neighb_size1}, options);
    torch::Tensor neb_counts = torch::empty({n_binstotal, 1}, options);

    torch::Tensor bin_counts = torch::zeros({n_binstotal, 1}, options);
    torch::Tensor bin_offsets = torch::empty({n_binstotal, 1}, options);
    torch::Tensor bid2ptid = torch::empty({pts_size0, 2}, options);
    torch::Tensor write_count = torch::zeros({1}, options);

    // calculate output size
    int intermed = int(lin_radius * 4 + 1);
    float area = float(intermed * intermed);
    float n_average = pts_size0 / batch_size;
    float hyp = area * n_average * batch_size * 16 * scale_radius;
    float sq_max = (n_average * n_average) * batch_size;

    const int edges_size0 = int(min(hyp, sq_max));
    const int edges_size1 = 2;

    torch::Tensor edges = torch::empty({edges_size0, edges_size1}, options);
    torch::Tensor glob_count = torch::zeros({1}, options);

    // Calculate neighbor bin_ids (neighbors); count writes to neighbor rows in neb_counts. Central bins are written to column 0 of neighbors.
    // Locations in neighbors without a bin_id are set to -1.
    NEB->frnn_neighbors_launch(
        neighbors.data<int>(),
        neb_counts.data<int>(),

        scabin_linrads.data<float>(),
        n_xbin_eachscale.data<int>(),
        n_xbin_eachscale_excume.data<int>(),
        glob_to_scabin.data<int>(),
        glob_to_im_i.data<int>(),

        xmin.item<float>(),
        xmax.item<float>(),

        n_scabin,
        n_xbin,
        n_binstotal,
        batch_size,
        neighb_size1,

        blocks,
        threads
    );
    CudaCheckError();

    // prune redundant bin pairs by row in neighbors; shuffle valid neighbors to lower columns; adjust neb_counts
	NEB->frnn_neighbors_prune_launch(
		neighbors.data<int>(),
		neb_counts.data<int>(),

		n_binstotal,
		neighb_size1,
		blocks,
		threads
	);
    CudaCheckError();

    // count number of pts in each bin, write to bin_counts; build bid2ptid (matching a bin_id to a pt_id by row)
    NEB->frnn_bin_counts_launch(
        pts,
        imgid.data<long>(),
        scabin_linrads.data<float>(),
        n_xbin_eachscale_excume.data<int>(),
        bin_counts.data<int>(),
        bid2ptid.data<int>(),
        write_count.data<int>(),

        scale_radius,
        xmin.item<float>(),
        logscalemin.item<float>(),
        pts_size0,
        pts_size1,
        int(n_binstotal / batch_size),
        n_binstotal,
        pts_x_col,
        pts_scale_col,

        blocks,
        threads
    );
    CudaCheckError();

    // exclusive-scan bin_counts; write to bin_offsets
    SCAN->scan_launch(  bin_counts.data<int>(),
                        bin_offsets.data<int>(),
                        n_binstotal);

    CudaCheckError();

    // build bin_ptids from bid2ptid, bin_offsets. bin_ptids give contiguous indexes into pts
    // for a given bin, where a bin's pt indexes start at bin_offsets[bin_id].
    NEB->frnn_ptids_launch(
        bin_counts.data<int>(),
        bin_offsets.data<int>(),
        bid2ptid.data<int>(),
        bin_ptids.data<int>(),
        bin_wcounts.data<int>(),

        pts_size0,
        blocks,
        threads
    );
    CudaCheckError();



    // frnn_bipart_kern a block for at least each valid bin_id in neighbors
    const int nebs_width = neb_counts.max().item<int>();
    const dim3 blocks_frnnkern(nebs_width, n_binstotal);

    const int n_threadsx = 16;
    const dim3 threads_frnnkern(n_threadsx, n_threadsx);

    // each block stores 3 float32's per point, up to max_binsize points per bin, and 2 bins in shared memory
    const int bin_stride = 3;
    const int max_binsize = bin_counts.max().item<int>();
    const size_t shared = (2 * bin_stride * max_binsize)*sizeof(float) + (2 * max_binsize)*sizeof(int);

    if (shared > 49152){
        fprintf (stderr, "ERROR frnn_driver.cu: FRNN_MAIN_KERNEL ATTEMPTED TO ALLOCATE TOO MUCH SHARED MEMORY FOR YOUR DEVICE ARCH; DECREASE BATCH SIZE");
        exit(EXIT_FAILURE);
    }

    // compare bins of points on frnn criteria, write indexes of pts with an edge between them into 'edges'.
    // Count number of integer writes in glob_counts (glob_counts= 2xnumber_edges_written).
    FRNN->frnn_kern_launch(
        neighbors.data<int>(),
        bin_counts.data<int>(),
        bin_offsets.data<int>(),
        bin_ptids.data<int>(),
        pts,

        glob_count.data<int>(),
        edges.data<int>(),

        lin_radius,
        scale_radius,

        max_binsize,
        bin_stride,

        neighb_size1,
        pts_size1,

        blocks_frnnkern,
        threads_frnnkern,
        shared
    );
    CudaCheckError();

    // free host objects' memory
    free(FRNN);
    free(SCAN);
    free(NEB);

    if (glob_count.item<int32_t>()/2 > edges_size0){
        fprintf (stderr, "ERROR frnn_driver.cu: FRNN_MAIN_KERNEL ATTEMPTED TO WRITE TOO MANY EDGES; INCREASE ALLOCATED EDGES SIZE OR DECREASE BATCH SIZE");
        exit(EXIT_FAILURE);
    }
    edges = edges.narrow(0, 0, glob_count.item<int>()/2);
    return {edges};
    //    return {edges, glob_count, neighbors, bid2ptid, bin_ptids, bin_offsets, n_xbin_eachscale, n_xbin_eachscale_excume};
}
//**/