/**
Author: Chris Henn (https://github.com/chrishenn)

Find minimum-score edges, for each group of edges contributing to a given target
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




__global__ void bin_edge_kern(

    const   long*  __restrict__  edges,
    const   float* __restrict__  scores,

            float* __restrict__  g_binscores,
            long*  __restrict__  g_edgeids,
            int*   __restrict__  g_bincounts,

    const   int                  max_binsize,
    const   int                  n_edges
) {
    // build global target object-bins; stride grid over 'edges' and 'scores' where edges.size(0) == scores.size(0)
    for (int glob_i = blockIdx.x * blockDim.x + threadIdx.x; glob_i < n_edges; glob_i += blockDim.x * gridDim.x) {

        // each target object from edges[:,1] has its own 'bin'. bid's are object-ids (index against pts.size(0))
        int bid = edges[glob_i * 2 + 1];
        int write_col = atomicAdd(&g_bincounts[bid], 1);

        g_binscores[bid * max_binsize + write_col] = scores[glob_i];
        g_edgeids[bid * max_binsize + write_col] = long(glob_i);
    }
}

__global__ void topfrac_edge_kern(

    const   long*  __restrict__  edges,
    const   float* __restrict__  scores,

            float* __restrict__  g_binscores,
            long*  __restrict__  g_edgeids,
            int*   __restrict__  g_bincounts,

            long*  __restrict__  drop_ids,
            int*   __restrict__  glob_count,

    const   int                  max_binsize,
    const   int                  n_bins,
    const   float                drop_frac
) {
    for (int glob_i = blockIdx.x * blockDim.x + threadIdx.x; glob_i < n_bins; glob_i += blockDim.x * gridDim.x)
    {
        int bin_count = g_bincounts[glob_i];
        if (bin_count < 1) continue;

        int drop_num = int( float(bin_count) * drop_frac );
        drop_num = min(drop_num, bin_count);
        
        // for each min to find
        for (int n_mins = 0; n_mins < drop_num; n_mins++)
        {
            int min_binloc;
            float min_score = 1000;
            
            // look through this row (this global bin)
            for (int col = 0; col < bin_count; col++){
    
                int curr_loc = glob_i * max_binsize + col;
                float curr_score = g_binscores[curr_loc];
                
                if (curr_score < min_score){
                    min_score = curr_score;
                    min_binloc = curr_loc;
                }
            }
    
            int glob_write = atomicAdd(glob_count, 1);
            drop_ids[glob_write] = g_edgeids[min_binloc];
            g_binscores[min_binloc] = 100000;
        }
    }  
}



std::vector<torch::Tensor> edge_min_cuda_call(
    torch::Tensor edges,
    torch::Tensor scores,
    torch::Tensor imgid,

    torch::Tensor max_binsize,

    float drop_frac
){

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

    torch::Tensor drop_ids =   torch::empty(edges.size(0), long_options);
    torch::Tensor glob_count = torch::zeros({1}, int_options);

    torch::Tensor g_binscores = torch::empty({imgid.size(0), max_binsize.item<int>()}, float_options);
    torch::Tensor g_edgeids =   torch::empty({imgid.size(0), max_binsize.item<int>()}, long_options);
    torch::Tensor g_bincounts = torch::zeros(imgid.size(0), int_options);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    int n_threads = 256;
    int sms = deviceProp.multiProcessorCount;
    int full_cover = (edges.size(0)-1) / n_threads + 1;
    int n_blocks = min(full_cover, 2*sms);

    const dim3 blocks(n_blocks);
    const dim3 threads(n_threads);

    bin_edge_kern<<<blocks, threads>>>(
        edges.data_ptr<long>(),
        scores.data_ptr<float>(),

        g_binscores.data_ptr<float>(),
        g_edgeids.data_ptr<long>(),
        g_bincounts.data_ptr<int>(),

        max_binsize.item<int>(),
        edges.size(0)
    );
    CudaCheckError();

    topfrac_edge_kern<<<blocks, threads>>>(
            edges.data_ptr<long>(),
            scores.data_ptr<float>(),

            g_binscores.data_ptr<float>(),
            g_edgeids.data_ptr<long>(),
            g_bincounts.data_ptr<int>(),

            drop_ids.data_ptr<long>(),
            glob_count.data_ptr<int>(),

            max_binsize.item<int>(),
            imgid.size(0),
            drop_frac
    );
    CudaCheckError();

    drop_ids = drop_ids.narrow(0, 0, glob_count.item<int>());
    return {drop_ids};

//    return {edges,scores,g_binscores, g_edgeids, g_bincounts,drop_ids, glob_count , max_binsize  };
}


















