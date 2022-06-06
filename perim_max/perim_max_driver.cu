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

#define CUDA_ERROR_CHECK
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



// https://forums.developer.nvidia.com/u/Andy_Lomas
__device__ int floatToOrderedInt( float floatVal )
{
    int intVal = __float_as_int( floatVal );
    return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
}
__device__ float orderedIntToFloat( int intVal )
{
    return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF );
}


// values is actually pts here; read sizes from column 4
__global__ void perim_max_main(
    const long* imgid,
    const int imgid_size0,

    const float* values,
    const int values_size1,
    const int values_readcol,

    int* inter_ints

){
    for (int glob_i = blockIdx.x * blockDim.x + threadIdx.x; glob_i < imgid_size0; glob_i += blockDim.x * gridDim.x)
    {
        auto im_id = imgid[glob_i];
        auto value = floatToOrderedInt( values[glob_i * values_size1 + values_readcol] );
        atomicMax(&inter_ints[im_id], value);
    }
}

__global__ void int_to_float(
        const int* inter_ints,
        float* max_perim,
        const int batch_size
){
    for (int glob_i = blockIdx.x * blockDim.x + threadIdx.x; glob_i < batch_size; glob_i += blockDim.x * gridDim.x)
    {
        max_perim[glob_i] = orderedIntToFloat( inter_ints[glob_i] );
    }
}


/** 
cpu entry point for python extension.
**/
std::vector<torch::Tensor> perim_max_call(
    torch::Tensor imgid,
    torch::Tensor values,
    torch::Tensor batch_size
) {

    using namespace torch::indexing;

    // set device
    auto device = imgid.get_device();
    cudaSetDevice(device);

    // tensor allocation
    auto float_opt = torch::TensorOptions()
            .dtype(torch::kF32)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device)
            .requires_grad(false);
    auto int_opt = torch::TensorOptions()
            .dtype(torch::kI32)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device)
            .requires_grad(false);

    auto max_perim = torch::empty(batch_size.item<int>(), float_opt);
    auto inter_ints = torch::zeros(batch_size.item<int>(), int_opt);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    int n_threads = 256;
    int sms = deviceProp.multiProcessorCount;
    int full_cover = (imgid.size(0)-1) / n_threads + 1;
    int n_blocks = min(full_cover, 2 * sms);

    const dim3 blocks(n_blocks);
    const dim3 threads(n_threads);

    int values_readcol;
    if (values.size(1) == 6) values_readcol = 4;
    else values_readcol = 0;

    perim_max_main<<<blocks, threads>>>(
            imgid.data_ptr<long>(),
            imgid.size(0),

            values.data_ptr<float>(),
            values.size(1),
            values_readcol,

            inter_ints.data_ptr<int>()
    ); CudaCheckError();

    n_threads = min(256, batch_size.item<int>());
    int_to_float<<<1, n_threads>>>(
            inter_ints.data_ptr<int>(),
            max_perim.data_ptr<float>(),
            batch_size.item<int>()
    ); CudaCheckError();

    return {max_perim};
}



