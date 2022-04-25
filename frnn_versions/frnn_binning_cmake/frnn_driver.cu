/**
Author: Chris Henn (https://github.com/chrishenn)

Implements the Fixed-Radius Nearest-Neighbor (frnn) algorithm [Qianli Liao and David Walter].

Parallel exclusive scan code adapted from [Matt Dean - 1422434 - mxd434].

This file relies on Pytorch calls from the Pytorch C++ API.

 NOTE:
 cuda __global__ kernels can be included from a .cu file by adding a declaration in this file, where the call is. The kernels will be compiled and found no
 problem (as long as the names are unique?). HOWEVER, templated __global__ kernels CANNOT be included in this way - the kernel call with the AT_DISPATCH macro
 and the kernel it's calling MUST be in the same file, or else python will throw an "unknown symbol" error.
**/

#include <torch/types.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>

// define for error checking
// #define CUDA_ERROR_CHECK

#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
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





/////////////////////////////////////////////////////
// frnn_bin_kern.cu
__global__ void frnn_neighbors(
        int*     neighbors,
        int*     neb_counts,

        float*   scabin_linrads,

        int*     n_xbin_eachscale,
        int*     n_xbin_eachscale_excume,

        int*     glob_to_scabin,
        int*     glob_to_im_i,

        const int n_binstotal,
        const int batch_size,

        const int neighb_size1
);

__global__ void frnn_neighbors_prune(
        int*  neighbors,
        int*  neb_counts,

        const int num_binstotal,
        const int neighb_size1
);

__global__ void frnn_neighbors_shuffle(
        int*  neighbors,

        const int num_binstotal,
        const int neighb_size1
);

__global__ void frnn_ptids(
        const   int*      bin_offsets,
        const   int*      bid2ptid,
        int*      bin_ptids,
        int*      bin_wcounts,

        const   int                   pts_size0
);

__host__ void frnn_bin_counts_launch(
        torch::Tensor   pts,
        long*    imgid,
        float*   radius_at_each_scabin,
        int*     scabin_offsets,
        int*     bin_counts,
        int*     bid2ptid,
        int*     write_count,

        float   scale_radius,
        float   xmin,
        float   scalemin,
        int     pts_size0,
        int     pts_size1,
        int     num_bins_perimg,
        int     num_binstotal,
        int     pts_x_col,
        int     pts_scale_col,

        dim3    blocks,
        dim3    threads
);


/////////////////////////////////////////////////////
// scan.cu
__host__ void scan_launch(
        int*    data_in,
        int*    data_out,
        int     tot_size
);


/////////////////////////////////////////////////////
// frnn_kern.cu
__host__ void frnn_kern_launch(

        int*    neighbors,
        int*    bin_counts,
        int*    bin_offsets,
        int*    bin_ptids,
        torch::Tensor pts,

        int*    glob_counts,
        long*   edges,

        float radius,
        float scale_radius,

        int max_binsize,
        int bin_stride,

        int neighb_size1,
        int pt_size1,

        dim3 blocks,
        dim3 threads,
        size_t shared
);





/////////////////////////////////////////////////////
// cpu entry point for frnn python extension.
__host__ std::vector<torch::Tensor> frnn_ts_call(
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
    auto i_options = torch::TensorOptions()
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


    // specific to the size of pts used with frnn
    auto pts_x_col = torch::full({1}, 1, l_options);
    auto pts_scale_col = torch::full({1}, 4, l_options);

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
    auto glob_to_scabin = torch::arange(n_xbin_eachscale_incume.size(0), i_options).masked_select(glob_to_scabin_m);

    glob_to_im_i.squeeze_();
    n_xbin_eachscale_incume.squeeze_();
    n_xbin_eachscale_excume.squeeze_();
    scabin_linrads.squeeze_();
    bin_logscales.squeeze_();

    // allocate device data structures
    auto bin_ptids = torch::empty({pts_size0, 1}, i_options);
    auto bin_wcounts = torch::zeros({n_binstotal.item<int>(), 1}, i_options);

    auto neighb_size1 = torch::full({1}, 20, i_options);
    auto neighbors = torch::empty({n_binstotal.item<int>(), neighb_size1.item<int>()}, i_options);
    auto neb_counts = torch::empty({n_binstotal.item<int>(), 1}, i_options);

    auto bin_counts = torch::zeros({n_binstotal.item<int>(), 1}, i_options);
    auto bin_offsets = torch::empty({n_binstotal.item<int>(), 1}, i_options);
    auto bid2ptid = torch::empty({pts_size0, 2}, i_options);
    auto write_count = torch::zeros({1}, i_options);

    // calculate output size
    auto intermed = (lin_radius * 4 + 1).to(torch::kInt32);
    auto area = intermed.pow(2).to(torch::kFloat32);
    auto n_average = torch::true_divide(torch::full({1}, pts_size0, i_options) , batch_size);
    auto hyp = (area * n_average * batch_size * 16 * scale_radius * 1.2).to(torch::kI32);
    auto sq_max = (n_average.pow(2) * batch_size * 1.2).to(torch::kI32);

    auto edges_size0 = torch::min(hyp, sq_max);
    auto edges_size1 = torch::full({1}, 2, i_options);

    auto edges = torch::empty({edges_size0.item<int>(), edges_size1.item<int>()}, l_options);
    auto glob_count = torch::zeros({1}, i_options);

    // Calculate neighbor bin_ids (neighbors); count writes to neighbor rows in neb_counts. Central bins are written to column 0 of neighbors.
    // Locations in neighbors without a bin_id are set to -1.
    frnn_neighbors<<<blocks,threads>>>(
        neighbors.data_ptr<int>(),
        neb_counts.data_ptr<int>(),

        scabin_linrads.data_ptr<float>(),
        n_xbin_eachscale.data_ptr<int>(),
        n_xbin_eachscale_excume.data_ptr<int>(),
        glob_to_scabin.data_ptr<int>(),
        glob_to_im_i.data_ptr<int>(),

        n_binstotal.item<int>(),
        batch_size.item<int>(),
        neighb_size1.item<int>()
    ); CudaCheckError();

//     prune redundant bin pairs by row in neighbors; shuffle valid neighbors to lower columns; adjust neb_counts
    frnn_neighbors_prune<<<blocks,threads>>>(
		neighbors.data_ptr<int>(),
		neb_counts.data_ptr<int>(),

		n_binstotal.item<int>(),
		neighb_size1.item<int>()
	); CudaCheckError();

    int loc_threads = 64;
    auto shared = loc_threads * neighb_size1.item<int>() * sizeof(int);

    frnn_neighbors_shuffle<<<blocks, loc_threads, shared>>>(
        neighbors.data_ptr<int>(),
        n_binstotal.item<int>(),
        neighb_size1.item<int>()
    ); CudaCheckError();

//     count number of pts in each bin, write to bin_counts; build bid2ptid (matching a bin_id to a pt_id by row)
    frnn_bin_counts_launch(
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
    ); CudaCheckError();

//     exclusive-scan bin_counts; write to bin_offsets
    scan_launch(bin_counts.data_ptr<int>(),
                bin_offsets.data_ptr<int>(),
                n_binstotal.item<int>());
    CudaCheckError();

//     build bin_ptids from bid2ptid, bin_offsets. bin_ptids give contiguous indexes into pts                    
//     for a given bin, where a bin's pt indexes start at bin_offsets[bin_id].
    frnn_ptids<<<blocks,threads>>>(
        bin_offsets.data_ptr<int>(),
        bid2ptid.data_ptr<int>(),
        bin_ptids.data_ptr<int>(),
        bin_wcounts.data_ptr<int>(),

        pts_size0
    ); CudaCheckError();

    // each block stores 5 float32's per point, up to max_binsize points per bin, and 2 bins in shared memory
    auto bin_stride = torch::full({1}, 5, i_options);
    auto max_binsize = bin_counts.max();
    auto shared_1 = 2 * bin_stride * max_binsize * long(sizeof(float));

    if ( shared_1.gt(49152).item<int>() ){
        fprintf (stderr, "ERROR frnn_driver.cu: FRNN_MAIN_KERNEL ATTEMPTED TO ALLOCATE TOO MUCH SHARED MEMORY FOR YOUR DEVICE ARCH; DECREASE OBJECT-DENSITY\n");
        fprintf (stderr, "attemped: %li bytes; max supported: 49152 bytes\n", shared_1.item<long>());
        exit(EXIT_FAILURE);
    }

    // compare bins of points on frnn criteria, write indexes of pts with an edge between them into 'edges'. 
    // Count number of integer writes in glob_counts (glob_counts= 2xnumber_edges_written).
    // frnn_kern needs a threadblock for at least each valid bin_id in neighbors
    frnn_kern_launch(
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
        shared_1.item<long>()
    ); CudaCheckError();

    auto edge_count = glob_count.floor_divide(2);
    if ( edge_count.gt(edges_size0).item<int>() ){
        fprintf (stderr, "ERROR frnn_driver.cu: FRNN_MAIN_KERNEL ATTEMPTED TO WRITE MORE EDGES THAN SPACE WAS ALLOCATED FOR\n");
        fprintf (stderr, "attemped: %i edges; allocated: %i edges\n", edge_count.item<int>(), edges_size0.item<int>());
        exit(EXIT_FAILURE);
    }

    edges = edges.narrow(0, 0, edge_count.item<int>());
    return {edges};

//    return {edges, glob_count, neighbors, bid2ptid, neb_counts,bin_counts,bin_offsets, bin_ptids, max_binsize, shared, n_binstotal,
//            neighb_size1, bin_stride, lin_radius, scale_radius };
}



