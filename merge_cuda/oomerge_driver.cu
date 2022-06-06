/**
Author: Chris Henn (https://github.com/chrishenn)
**/

#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>


// define for error checking
// #define CUDA_ERROR_CHECK

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








__global__ void merge_kern(
    const   long*   __restrict__  send_map,
    const   int* __restrict__     n_merge_each,

    const   long* __restrict__    edge_ids,
    int* __restrict__ o_counters,
    int* __restrict__ write_count,

    long* __restrict__ m_edge_ids,

    const int e_ids_size0
)
{
    int glob_thid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_width = blockDim.x * gridDim.x;

    for (int i = glob_thid; i < e_ids_size0; i += grid_width)
    {
        auto dst_oid = send_map[i * 2 + 1];

        auto target_count = n_merge_each[dst_oid];

        auto o_count = atomicAdd(&o_counters[dst_oid], 1);

        if (o_count < target_count)
        {
            int thread_i = atomicAdd(write_count, 1);
            m_edge_ids[thread_i] = edge_ids[i];
        }
    }
}


/**
cpu entry point
**/
std::vector<torch::Tensor> oomerge_cuda_call(
  torch::Tensor send_map,
  torch::Tensor n_merge_each,

  int device
){
    cudaSetDevice(device);

    auto options_int = torch::TensorOptions()
      .dtype(torch::kInt32)
      .layout(torch::kStrided)
      .device(torch::kCUDA, device);
    auto options_long = torch::TensorOptions()
      .dtype(torch::kInt64)
      .layout(torch::kStrided)
      .device(torch::kCUDA, device);

    const auto e_ids_size0 = send_map.size(0);
    const auto size_out = n_merge_each.sum().item<int>();

    torch::Tensor m_edge_ids =  torch::empty(size_out, options_long);
    torch::Tensor edge_ids =  torch::arange(e_ids_size0, options_long);
    torch::Tensor o_counters = torch::zeros(n_merge_each.size(0), options_int);
    torch::Tensor write_count = torch::zeros({1}, options_int);

    const auto n_threads =     128;
    const auto n_blocks =      ( (e_ids_size0-1)/n_threads + 1) / 2;
    const dim3 blocks(n_blocks);
    const dim3 threads(n_threads);

    merge_kern<<<blocks, threads>>>(
      send_map.data<long>(),
      n_merge_each.data<int>(),

      edge_ids.data<long>(),
      o_counters.data<int>(),
      write_count.data<int>(),

      m_edge_ids.data<long>(),

      e_ids_size0
    );
    CudaCheckError();

    return {m_edge_ids};
}
