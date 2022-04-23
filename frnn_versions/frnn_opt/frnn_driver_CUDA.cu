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

double get_nanos();
void CHECK_LAUNCH_ERROR();

double get_nanos() {
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (double)ts.tv_sec * 1000000000L + ts.tv_nsec;
}
void CHECK_LAUNCH_ERROR(){
    do {
        /* Check synchronous errors, i.e. pre-launch */
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",
                     __FILE__, __LINE__, cudaGetErrorString(err) );
            exit(EXIT_FAILURE);
        }
        /* Check asynchronous errors, i.e. kernel failed (ULF) */
        err = cudaDeviceSynchronize();
        if (cudaSuccess != err) {
            fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",
                     __FILE__, __LINE__, cudaGetErrorString( err) );
            exit(EXIT_FAILURE);
        }
    } while (0);
}

// Entry point cpu code for python
std::vector<torch::Tensor> frnn_cuda_call(
    torch::Tensor pts,
    torch::Tensor img_ids,
    torch::Tensor radius_at_each_scabin,
    torch::Tensor scabin_offsets,

    int num_imgs,
    int num_bins_perimg,
    int num_scabin,
    int pt_size0,
    int pt_size1,
    int device_id,
    float radius,
    float scale_radius,
    float xmin,
    float scalemin
){

    auto options = torch::TensorOptions()
      .dtype(torch::kInt32)
      .layout(torch::kStrided)
      .device(torch::kCUDA, device_id);

    const int num_binstotal =   num_bins_perimg * num_imgs;

    const int neighb_size1 =    20;
    const int edges_size0 =     5000000;
    const int edges_size1 =     2;

    const int n_blocks =        256;
    const int n_threads =       256;
    const dim3 blocks(n_blocks);
    const dim3 threads(n_threads);

    const int n_threadsx = 16;
    const dim3 threads1(n_threadsx, n_threadsx);


    int* neighbors;
    cudaMalloc((void **)&neighbors, (num_binstotal * neighb_size1)*sizeof(int) );

    int* neb_counts;
    cudaMalloc((void **)&neb_counts, (num_binstotal)*sizeof(int) );

    int* bin_ptids;
    cudaMalloc((void **)&bin_ptids, (pt_size0)*sizeof(int) );

    int* bin_wcounts;
    cudaMalloc((void **)&bin_wcounts, (num_binstotal)*sizeof(int) );
    cudaMemset(bin_wcounts, 0, (num_binstotal)*sizeof(int));

	int* bin_counts;
    cudaMalloc((void **)&bin_counts, (num_binstotal)*sizeof(int) );

    int* write_count;
    cudaMalloc((void **)&write_count, sizeof(int) );
    cudaMemset(write_count, 0, sizeof(int));

	int* bin_offsets;
    cudaMalloc((void **)&bin_offsets, (num_binstotal)*sizeof(int) );

	int* glob_count;
	cudaMalloc((void **)&glob_count, sizeof(int) );
	cudaMemset(glob_count, 0, sizeof(int));

	int* bid2ptid;
	cudaMalloc((void **)&bid2ptid, (pt_size0 * 2)*sizeof(int) );

    int* edges;
    cudaMalloc((void **)&edges, (edges_size0 * edges_size1)*sizeof(int) );

    frnn_bin_kern* NEB = new frnn_bin_kern();
    NEB->frnn_neighbors_launch(
        radius_at_each_scabin.data<float>(),
        scabin_offsets.data<int>(),
        neighbors,
        neb_counts,

        num_binstotal,
        num_bins_perimg,
        num_imgs,
        num_scabin,
        neighb_size1,

        blocks,
        threads
    );

	NEB->frnn_neighbors_prune_launch(
		neighbors,
		neb_counts,

		num_binstotal,
		neighb_size1,

		blocks,
		threads
	);

    int* d_nebs_width = thrust::max_element(thrust::device, neb_counts, neb_counts + num_binstotal);
    int* h_nebs_width =  new int[1];
    cudaMemcpy( h_nebs_width, d_nebs_width, sizeof(int), cudaMemcpyDeviceToHost);
    
    const int nebs_width = *h_nebs_width;
    const dim3 blocks1(nebs_width, num_binstotal);

    NEB->frnn_bin_counts_launch(
        pts.data<float>(),
        img_ids.data<int>(),
        radius_at_each_scabin.data<float>(),
        scabin_offsets.data<int>(),
        bin_counts,
        bid2ptid,
        write_count,

        scale_radius,
        xmin,
        scalemin,
        pt_size0,
        pt_size1,
        num_bins_perimg,
        num_binstotal,

        blocks,
        threads
    );

    scan* SCAN = new scan();
    SCAN->scan_launch(  bin_counts,
                        bin_offsets,
                        num_binstotal);

    NEB->frnn_ptids_launch(
        bin_counts,
        bin_offsets,
        bid2ptid,
        bin_ptids,
        bin_wcounts,

        pt_size0,

        blocks,
        threads
    );

    int* d_max_binsize = thrust::max_element(thrust::device, bin_counts, bin_counts + num_binstotal);
    int* h_max_binsize = new int[1];
    cudaMemcpy( h_max_binsize, d_max_binsize, sizeof(int), cudaMemcpyDeviceToHost);
    
    const int max_binsize = *h_max_binsize;
    const int bin_stride = 5;
    const size_t shared = (2 * bin_stride * max_binsize) * sizeof(float);

    if (shared > 49152){ printf("FRNN ERROR: KERNEL ATTEMPTED TO ALLOCATE TOO MUCH SHARED MEMORY\n"); }

    frnn_bipart_kern* FRNN = new frnn_bipart_kern();
    FRNN->frnn_kern_launch(
        neighbors,
        bin_counts,
        bin_offsets,
        bin_ptids,
        pts.data<float>(),

        glob_count,
        edges,

        radius,
        scale_radius,

        max_binsize,
        bin_stride,

        neighb_size1,
        pt_size1,

        blocks1,
        threads1,
        shared
    );

    int* h_glob_count = new int[1];
    cudaMemcpy( h_glob_count, glob_count, sizeof(int), cudaMemcpyDeviceToHost);
    const int edges_out_size = *h_glob_count;

    free(FRNN);
    free(SCAN);
    free(NEB);
	free(h_glob_count);
	free(h_nebs_width);
	free(h_max_binsize);
    cudaFree(neighbors);
    cudaFree(neb_counts);
    cudaFree(bin_ptids);
    cudaFree(bin_wcounts);
    cudaFree(bin_counts);
    cudaFree(write_count);
    cudaFree(bin_offsets);
    cudaFree(glob_count);
    cudaFree(bid2ptid);

    torch::Tensor edges_out = torch::from_blob(edges, {int(edges_out_size/2), edges_size1}, {edges_size1,1}, options);

    return {edges_out}; 
}
