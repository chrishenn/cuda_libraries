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




template <typename scalar_t>
__global__ void bin_edges(
        const long* edges,
        const scalar_t* scores,

        const long* e_imgid,
            scalar_t* im_escores,
              long* im_eids,
              int* bin_counts,

        const int max_binsize,
        const int edges_size0
) {
    for (int glob_i = blockIdx.x * blockDim.x + threadIdx.x; glob_i < edges_size0; glob_i += blockDim.x * gridDim.x)
    {
        long imgid = e_imgid[glob_i];

        int write_col = atomicAdd(&bin_counts[imgid], 1);
        im_escores[imgid * max_binsize + write_col] = scores[glob_i];
        im_eids[imgid * max_binsize + write_col] = long(glob_i);
    }  
}

__global__ void collate_top(
                long* im_eids,
                int* bin_counts,

                long* keep_ids,
                int* glob_count,

        const int max_binsize,
        const double keep_frac
) {
    int row = blockIdx.x;
    int bin_size = bin_counts[row];
    int block_keep = int( bin_size * keep_frac ) + 1;
    block_keep = min(block_keep, bin_size);

    for (int block_i = threadIdx.x; block_i < max_binsize; block_i += blockDim.x)
    {
        if (block_i < block_keep){
            int glob_write = atomicAdd(glob_count, 1);
            keep_ids[glob_write] = im_eids[row * max_binsize + block_i];
        }
    }
}



std::vector<torch::Tensor> edge_topim_cuda_call(
    torch::Tensor edges,
    torch::Tensor scores,
    torch::Tensor imgid,
    torch::Tensor batch_size,

    double keep_frac
){
    using namespace torch::indexing;

    auto device = edges.get_device();
    cudaSetDevice(device);

    auto int_options = torch::TensorOptions()
            .dtype(torch::kInt32)
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

    auto dyn_float_options = torch::TensorOptions()
            .dtype(scores.scalar_type())
            .layout(torch::kStrided)
            .device(torch::kCUDA, device);

    auto e_imgid = imgid.index_select(0, edges.index({Slice(), 0}));
    auto max_binsize = e_imgid.bincount().max();

    torch::Tensor keep_ids =   torch::empty(edges.size(0), long_options);
    torch::Tensor glob_count = torch::zeros(1, int_options);

    torch::Tensor bin_counts = torch::zeros(batch_size.item<int>(), int_options);
    torch::Tensor im_escores = torch::full({batch_size.item<int>(), max_binsize.item<int>()}, -10000, dyn_float_options);
    torch::Tensor im_eids = torch::empty({batch_size.item<int>(), max_binsize.item<int>()}, long_options);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    int n_threads = 256;
    int sms = deviceProp.multiProcessorCount;
    int full_cover = (edges.size(0)-1) / n_threads + 1;
    int n_blocks = min(full_cover, 2*sms);
    const dim3 blocks(n_blocks);
    const dim3 threads(n_threads);

    // bin edges along a row for each image, for both scores and edge_ids. match cols.
    AT_DISPATCH_FLOATING_TYPES_AND(torch::ScalarType::Half, scores.scalar_type(), "bin_edges", ([&] {
        bin_edges<<<blocks, threads>>>(
            edges.data_ptr<long>(),
            scores.data_ptr<scalar_t>(),

            e_imgid.data_ptr<long>(),
            im_escores.data_ptr<scalar_t>(),
            im_eids.data_ptr<long>(),
            bin_counts.data_ptr<int>(),

            max_binsize.item<int>(),
            edges.size(0)
        ); })); CudaCheckError();

    // use im_escores.sort(dim=1) to get ids, to sort im_escores and im_eids
    auto sortids  = std::get<1>( im_escores.sort(1, true) );
    im_eids = im_eids.gather(1, sortids);

    // launch block for each row, calc keep num for im, put top edge scores in keep_ids
    collate_top<<<batch_size.item<int>(), 512>>>(
        im_eids.data_ptr<long>(),
        bin_counts.data_ptr<int>(),

        keep_ids.data_ptr<long>(),
        glob_count.data_ptr<int>(),

        max_binsize.item<int>(),
        keep_frac
    ); CudaCheckError();
    keep_ids = keep_ids.narrow(0, 0, glob_count.item<int>());

    return {keep_ids};
//    return {im_escores, im_eids, sortids, e_imgid, edges, scores, glob_count, max_binsize, bin_counts, keep_ids};

}


















