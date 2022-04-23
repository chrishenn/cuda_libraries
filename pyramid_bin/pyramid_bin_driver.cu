/**
Author: Chris Henn (https://github.com/chrishenn)

find percentage of top-scoring edges in each image
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




__global__ void count_max_binsize(

        const long* scabin,
        const long* locs_imgid,
        const float* locs,
        const int locs_size0,
        const int nim_pyramid,

        int* bin_counts
) {
    for (int glob_i = blockIdx.x * blockDim.x + threadIdx.x; glob_i < locs_size0; glob_i += blockDim.x * gridDim.x)
    {
        long img_i = locs_imgid[glob_i];
        long py_bin = scabin[glob_i];

        int write_col = atomicAdd(&bin_counts[img_i * nim_pyramid + py_bin], 1);
    }
}


__global__ void bin_pyramid(

        const long* scabin,
        const long* locs_imgid,
        const float* locs,
        const int locs_size0,

        int* write_counts,
        uint8_t* mark_valid,
        long* patch_ids,
        long* new_patchids,

        float* grid,
        const int grid_size0,
        const int grid_size1,
        const int grid_size2,
        const int grid_size3
) {
    for (int glob_i = blockIdx.x * blockDim.x + threadIdx.x; glob_i < locs_size0; glob_i += blockDim.x * gridDim.x)
    {
        long img_i = locs_imgid[glob_i];
        long py_bin = scabin[glob_i];

        int pyb_size = grid_size2 * grid_size3;
        int imb_size = grid_size1 * pyb_size;

        int write_col = atomicAdd(&write_counts[img_i * grid_size1 + py_bin], 1);

        // NOTE: grid uses [x,y] for coordinates - to mesh with grid_sample
        grid[(img_i * imb_size) + (py_bin * pyb_size) + (write_col * grid_size3) + 1] = locs[glob_i * 2 + 0];
        grid[(img_i * imb_size) + (py_bin * pyb_size) + (write_col * grid_size3) + 0] = locs[glob_i * 2 + 1];

        new_patchids[img_i * (imb_size/2) + py_bin * (pyb_size/2) + write_col] = patch_ids[glob_i];
        mark_valid[img_i * (imb_size/2) + py_bin * (pyb_size/2) + write_col] = 1;
    }
}


std::vector<torch::Tensor> pyramid_bin_cuda_call(
    torch::Tensor scabin,
    torch::Tensor locs,
    torch::Tensor locs_imgid,
    torch::Tensor patch_ids,
    torch::Tensor batch_size

){
    using namespace torch::indexing;

    auto device = scabin.get_device();
    cudaSetDevice(device);

    auto int_options = torch::TensorOptions()
            .dtype(torch::kInt32)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device);

    auto uint8_options = torch::TensorOptions()
            .dtype(torch::kUInt8)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device);

    auto long_options = torch::TensorOptions()
            .dtype(torch::kInt64)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device);

    auto float_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    int n_threads = 256;
    int sms = deviceProp.multiProcessorCount;
    int full_cover = (locs.size(0)-1) / n_threads + 1;
    int n_blocks = min(full_cover, 2*sms);
    const dim3 blocks(n_blocks);
    const dim3 threads(n_threads);

    auto nim_pyramid = scabin.max() +1;
    torch::Tensor bin_counts = torch::zeros(batch_size.item<int>()* nim_pyramid.item<int>(), int_options);

    count_max_binsize<<<blocks, threads>>>(

            scabin.data_ptr<long>(),
            locs_imgid.data_ptr<long>(),
            locs.data_ptr<float>(),
            locs.size(0),
            nim_pyramid.item<int>(),

            bin_counts.data_ptr<int>()
    );
    CudaCheckError();

    auto max_scabin_size = bin_counts.max();
    torch::Tensor grid = torch::zeros({batch_size.item<int>(), nim_pyramid.item<int>(), max_scabin_size.item<int>(), 2}, float_options).sub_(5000);

    torch::Tensor new_patchids = torch::empty({batch_size.item<int>(), nim_pyramid.item<int>(), max_scabin_size.item<int>()}, long_options);
    torch::Tensor mark_valid =   torch::zeros({batch_size.item<int>(), nim_pyramid.item<int>(), max_scabin_size.item<int>()}, uint8_options);

    torch::Tensor write_counts = torch::zeros(batch_size.item<int>() * nim_pyramid.item<int>(), int_options);

    bin_pyramid<<<blocks, threads>>>(

            scabin.data_ptr<long>(),
            locs_imgid.data_ptr<long>(),
            locs.data_ptr<float>(),
            locs.size(0),

            write_counts.data_ptr<int>(),
            mark_valid.data_ptr<uint8_t>(),
            patch_ids.data_ptr<long>(),
            new_patchids.data_ptr<long>(),

            grid.data_ptr<float>(),
            grid.size(0),
            grid.size(1),
            grid.size(2),
            grid.size(3)
    );
    CudaCheckError();

    return {grid, mark_valid, new_patchids};
}


















