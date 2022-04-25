/**
Author: Chris Henn (https://github.com/chrishenn)

prune edges in 'drop_edges' from 'edges'
**/

#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>


// define for error checking
 #define CUDA_ERROR_CHECK

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





__global__ void prune_kern(

    const   long*  __restrict__  edges,
    const   long*  __restrict__  drop_edges,

    uint8_t* __restrict__ keep_mask,

    int edges_size0,
    int drop_edges_size0
)
{
    for (int glob_i = blockIdx.x * blockDim.x + threadIdx.x; glob_i < edges_size0; glob_i += blockDim.x * gridDim.x)
    {
        bool found = false;

        long pt_lf = edges[glob_i *2 + 0];
        long pt_rt = edges[glob_i *2 + 1];

        for (int i = 0; i < drop_edges_size0; i++)
        {
            long drop_pt_lf = drop_edges[i *2 + 0];

            if (pt_lf == drop_pt_lf){
                long drop_pt_rt = drop_edges[i *2 + 1];

                if (pt_rt == drop_pt_rt){
                    found = true;
                    break;
                }
            }
        }

        if (found){
            keep_mask[glob_i] = 0;
        }else{
            keep_mask[glob_i] = 1;
        }
    }  
}



std::vector<torch::Tensor> edge_prune_cuda_call(
    torch::Tensor edges,
    torch::Tensor drop_edges
){

    auto device = edges.get_device();
    cudaSetDevice(device);

    auto uint8_options = torch::TensorOptions()
            .dtype(torch::kUInt8)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device);

    torch::Tensor keep_mask =  torch::empty(edges.size(0), uint8_options);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    int n_threads = 256;
    int sms = deviceProp.multiProcessorCount;
    int full_cover = (edges.size(0)-1) / n_threads + 1;
    int n_blocks = min(full_cover, 2*sms);
    const dim3 blocks(n_blocks);
    const dim3 threads(n_threads);

    prune_kern<<<blocks, threads>>>(
        edges.data_ptr<long>(),
        drop_edges.data_ptr<long>(),
        keep_mask.data_ptr<uint8_t>(),

        edges.size(0),
        drop_edges.size(0)
    );
    CudaCheckError();

    return {keep_mask};
}


















