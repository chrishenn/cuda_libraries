/**
Authors: Christian Henn, Qianli Liao
**/

#include <torch/types.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

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


// tex must have width = 1
template <typename scalar_t>
__global__ void row_write_kernel(

        scalar_t* rel_vec,
        const int rel_vec_size0,
        const int rel_vec_size1,

        int* write_cols,

        const scalar_t* tex_rt,

        const scalar_t* vecs,
        const long* rowids,

        const int rowids_size0
){
    const int item_width = 3;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rowids_size0; i += blockDim.x * gridDim.x)
    {
        // rowids gives the object's rowid, for the obj in the left col of edges for this vector
        auto o_id = rowids[i];

        auto write_col = atomicAdd(&write_cols[o_id], item_width);

        if (write_col + 2 >= rel_vec_size1) continue;

        // vecs are relative vectors, starting at the obj with o_id, pointing to a neb
        auto vec0 = vecs[ i * 2 + 0 ];
        auto vec1 = vecs[ i * 2 + 1 ];

        // texture of neb at terminus of this vector
        auto tex2 = tex_rt[i];

        rel_vec[ o_id * rel_vec_size1 + write_col + 0 ] = vec0;
        rel_vec[ o_id * rel_vec_size1 + write_col + 1 ] = vec1;

        rel_vec[ o_id * rel_vec_size1 + write_col + 2 ] = tex2;
    }
}







/////////////////////////////////////////////////////
// cpu entry point for frnn python extension.
__host__ std::vector<torch::Tensor> write_row_call(
    torch::Tensor rel_vec,
    torch::Tensor write_cols,

    torch::Tensor tex,
    torch::Tensor tex_rt,

    torch::Tensor vecs,
    torch::Tensor rowids
) {

    using namespace torch::indexing;
    auto device_id = rel_vec.get_device();
    cudaSetDevice(device_id);

    // rel_vec and write_cols buffers are reused - must re-init here
    rel_vec.fill_(0);
    write_cols.fill_(1);

    // col 0 of rel_vec gives the tex of the obj at this row - one rel-vec for each obj
    rel_vec.index({Slice(), 0}) = tex.squeeze();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    int n_threads = 256;
    int sms = deviceProp.multiProcessorCount;
    int full_cover = (rowids.size(0)-1) / n_threads + 1;
    int n_blocks = min(full_cover, 2 * sms);

    AT_DISPATCH_ALL_TYPES(rel_vec.scalar_type(), "row_write_kernel", ([&] {
    row_write_kernel<<<n_blocks, n_threads>>>
    (
        rel_vec.data_ptr<scalar_t>(),
        rel_vec.size(0),
        rel_vec.size(1),

        write_cols.data_ptr<int>(),

        tex_rt.data_ptr<scalar_t>(),

        vecs.data_ptr<scalar_t>(),
        rowids.data_ptr<long>(),
        rowids.size(0)

    ); }));

    return {};
}



