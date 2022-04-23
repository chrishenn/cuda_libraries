#include <torch/types.h>

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>

#include "frnn_bin_kern.h"
#include "frnn_kern.h"
#include "scan.h"



// Entry point cpu code for python
std::vector<torch::Tensor> frnn_cuda_call(
    torch::Tensor pts,
    torch::Tensor img_ids,
    torch::Tensor radius_at_each_scabin,
    torch::Tensor scabin_offsets,

    int num_imgs,
    int num_bins_perimg,
    int device_id,
    float radius,
    float scale_radius,
    float xmin,
    float scalemin
    )
{

    ///////// INIT
    auto options = torch::TensorOptions()
      .dtype(torch::kInt32)
      .layout(torch::kStrided)
      .device(torch::kCUDA, device_id);

    at::cuda::CUDAStream stream = at::cuda::getDefaultCUDAStream();

    const int num_scabin =      scabin_offsets.size(0) - 1;
          int num_binstotal =   num_bins_perimg * num_imgs;

    const int pts_size0 =       pts.size(0);
    const int pts_size1 =       pts.size(1);

    const int n_blocks = 256;
    const int n_threads = 256;
    const int n_threads_total = n_threads * n_blocks;
    const dim3 blocks(n_blocks);
    const dim3 threads(n_threads);


    //// FRNN NEIGHBORS
    const int max_neighbors = 20;
    torch::Tensor neighbors = torch::empty({num_binstotal, max_neighbors}, options);

    int* neb_counts;
    cudaMalloc((void **)&neb_counts, n_threads_total*sizeof(int) );

    frnn_bin_kern* NEB = new frnn_bin_kern();
    NEB->frnn_neighbors_launch(
        radius_at_each_scabin.data<float>(),
        scabin_offsets.data<int>(),
        neighbors.data<int>(),
        neb_counts,

        num_binstotal,
        num_bins_perimg,
        num_imgs,
        num_scabin,
        max_neighbors,

        blocks,
        threads
    );

    // narrow neighbors to width of widest bin
    int* max_nebs_width = thrust::max_element(thrust::device, neb_counts, neb_counts + n_threads_total);

    int* h_max_nebs_width = (int*)malloc (sizeof(int));
    cudaMemcpy( h_max_nebs_width, max_nebs_width, sizeof(int), cudaMemcpyDeviceToHost);


    ///////// BIN_COUNTS (rows of counts of points in each bin indexed by each row's index)
    torch::Tensor bin_counts = torch::empty({num_binstotal, 1}, options);
    torch::Tensor bid2ptid = torch::empty({pts_size0, 2}, options);
    torch::Tensor write_count = torch::zeros({1}, options);

    NEB->frnn_bin_counts_launch(
        pts.data<float>(),
        img_ids.data<int>(),
        radius_at_each_scabin.data<float>(),
        scabin_offsets.data<int>(),
        bin_counts.data<int>(),
        bid2ptid.data<int>(),
        write_count.data<int>(),

        scale_radius,
        xmin,
        scalemin,
        pts_size0,
        pts_size1,
        num_bins_perimg,
        num_binstotal,

        blocks,
        threads
    );


    //////// BIN_OFFSETS
    bool shrink_flag = false;
    if (num_binstotal%2 != 0){
        num_binstotal++;
        shrink_flag = true;
    }
    torch::Tensor bin_offsets = torch::empty({num_binstotal, 1}, options);

    scan* SCAN = new scan();
    SCAN->scan_launch(bin_counts.data<int>(),
                    bin_offsets.data<int>(),
                    num_binstotal);
    if (shrink_flag){
        num_binstotal--;
        bin_offsets = torch::narrow(bin_offsets, 0, 0, num_binstotal);
    }

    //////// BIN_PTIDS
    torch::Tensor bin_ptids =   torch::empty({pts_size0, 1}, options);
    torch::Tensor bin_wcounts = torch::zeros({num_binstotal, 1}, options);

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


    ////////////// FRNN
    const auto neighb_size0 = num_binstotal;
    const auto neighb_size1 = *h_max_nebs_width;

    // find max binsize to limit shared memory in frnn kernel
    int* d_max_binsize = thrust::max_element(thrust::device, bin_counts.data<int>(), bin_counts.data<int>() + num_binstotal);
    int h_max_binsize[1];
    cudaMemcpy( h_max_binsize, d_max_binsize, sizeof(int), cudaMemcpyDeviceToHost);
    const int max_binsize = *h_max_binsize;

    // outputs: edges, glob_count. Blocks defined by actual width of neighbors; won't be narrowed till end
    const int edges_size0 = 5000000;
    const int edges_size1 = 2;

    const int n_threadsx = 16;
    const dim3 blocks1(neighb_size1, neighb_size0);
    const dim3 threads1(n_threadsx, n_threadsx);

    int* glob_counts;
    cudaMalloc((void **)&glob_counts, sizeof(int) );
    cudaMemset(glob_counts, 0, sizeof(int));

    torch::Tensor edges = torch::empty({edges_size0, edges_size1}, options);

    cudaStreamSynchronize(stream);

    // shared mem allocation: 5 values per point stored in a bin, and two bins at a time
    const int bin_stride = 5;
    const size_t dyna_shared = (2 * bin_stride * max_binsize) * sizeof(float);
    if (dyna_shared > 49152){ printf("FRNN ERROR: KERNEL ATTEMPTED TO ALLOCATE TOO MUCH SHARED MEMORY\n"); }


    // count writes per thread
    frnn_kern* FRNN = new frnn_kern();
    FRNN->frnn_kern_launch(

        neighbors.data<int>(),
        bin_counts.data<int>(),
        bin_offsets.data<int>(),
        bin_ptids.data<int>(),
        pts.data<float>(),

        glob_counts,
        edges.data<int>(),

        radius,
        scale_radius,

        max_binsize,
        bin_stride,

        neighb_size0,
        max_neighbors,

        pts_size0,
        pts_size1,

        edges_size0,

        blocks1,
        threads1,
        dyna_shared
    );

    int* h_glob_counts = (int*)malloc (sizeof(int));
    cudaMemcpy( h_glob_counts, glob_counts, sizeof(int), cudaMemcpyDeviceToHost);
    edges = torch::narrow(edges, 0, 0, *h_glob_counts/2);

    neighbors = torch::narrow(neighbors, 1, 0, *h_max_nebs_width);

    cudaFree(neb_counts);
    cudaFree(glob_counts);
    free(h_max_nebs_width);
    free(h_glob_counts);
    free(FRNN);
    free(SCAN);
    free(NEB);

    return {edges};
}
