/**
Authors: Christian Henn, Qianli Liao
**/

//#include <torch/types.h>
#include <torch/script.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>


// define this line for error checking, and put CudaCheckError() after a kernel call; I've seen little to no performance impact, though
//  the call does theoretically synchronize the device.
// #define CUDA_ERROR_CHECK

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

// timing helper - works from cpu - remember to synchronize before and after kernel call
double get_nanos() {
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (double)ts.tv_nsec;
}


// You can also add the __restrict__ keyword to ensure that (some types of?) memory accesses will not read off the end
// of one array into an array whose starting address is referenced by another defined variable name. Good for debugging
// but can incurr significant performance penalty.

// copies one array of ints 'input' to 'output'
__global__ void example_main(
    const int* input,
          int* output,
    const int size
){
    for (int glob_i = blockIdx.x * blockDim.x + threadIdx.x; glob_i < size; glob_i += blockDim.x * gridDim.x)
    {
        auto in_num = input[glob_i];
        output[glob_i] = in_num;
    }
}


/** 
cpu entry point for python extension.
**/
std::vector<torch::Tensor> example_call(
    torch::Tensor input
) {

    // indexing gives us advanced slicing support using the torch c++ api
    // note: .item() syntax is templated in c++ torch-api, as in:  .item<int>()
    using namespace torch::indexing;

    // set device
    auto device = input.get_device();
    cudaSetDevice(device);

    // tensor allocation
    auto int_opt = torch::TensorOptions()
            .dtype(torch::kInt32)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device)
            .requires_grad(false);

    auto output = torch::empty(input.size(0), int_opt);

    // calculate grid size
    // This code attempts to fit 2 cuda blocks per sm on the device, unless blocks of 256 threads each can cover the global datasize
    //  with fewer total blocks.
    // I've found this scheme to work best in limited experiments.
    // Blocks of more than 512 threads generally see performance regression on pascal and newer.
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    int n_threads = 256;
    int sms = deviceProp.multiProcessorCount;
    int full_cover = (input.size(0)-1) / n_threads + 1;
    int n_blocks = min(full_cover, 2 * sms);

    const dim3 blocks(n_blocks);
    const dim3 threads(n_threads);

    // there is a way to dispatch to a templated kernel definition, switching on a tensor dtype. Good luck finding documentation;
    //  however, frnn can dispatch to templates supporting torch.float and torch.float16 if you need that syntax.
    example_main<<<blocks, threads>>>(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        output.size(0)
    );
    CudaCheckError();

    // the std::vector of torch::Tensor 's in the c++ api will return a python-list of tensors to python
    // some c++ primitives appear to be supported.
    return {input, output};
}



