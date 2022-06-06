/**
Author: Chris Henn (https://github.com/chrishenn)

find top-scoring edges in each group of edges, where a group of edges provides predictions for a given target object
 global threshold for scores acts across all images, is an absolute value-threshold
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




__global__ void bin_edge_kern(

    const   long*  __restrict__  edges,
    const   float* __restrict__  scores,

            float* __restrict__  g_binscores,
            long*  __restrict__  g_edgeids,
            int*   __restrict__  g_bincounts,

            long*  __restrict__  keep_ids,
            int*   __restrict__  glob_count,

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

            long*  __restrict__  keep_ids,
            int*   __restrict__  glob_count,

    const   int                  max_binsize,
    const   int                  n_bins,
    const   float                keep_frac,
    const   float                threshold
) {
    for (int glob_i = blockIdx.x * blockDim.x + threadIdx.x; glob_i < n_bins; glob_i += blockDim.x * gridDim.x)
    {
        int bin_count = g_bincounts[glob_i];
        if (bin_count < 1) continue;

        int keep_num = int( float(bin_count) * keep_frac ) +1;
        keep_num = min(keep_num, bin_count);
        
        // for each max to find
        for (int n_maxes = 0; n_maxes < keep_num; n_maxes++)
        {
            int max_binloc = -1;
            float max_score = threshold;
            
            // look through this row (this global bin)
            for (int col = 0; col < bin_count; col++){
    
                int curr_loc = glob_i * max_binsize + col;
                float curr_score = g_binscores[curr_loc];
                
                if (curr_score > max_score){
                    max_score = curr_score;                      
                    max_binloc = curr_loc;
                }
            }

            if (max_binloc > -1) {
                int glob_write = atomicAdd(glob_count, 1);
                keep_ids[glob_write] = g_edgeids[max_binloc];
                g_binscores[max_binloc] = -100000;
            } else { break; }
        }
    }  
}



std::vector<torch::Tensor> edge_bin_cuda_call(
    torch::Tensor edges,
    torch::Tensor scores,
    torch::Tensor imgid,

    float keep_frac,
    float threshold
) {

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

    auto max_binsize = edges.index({Slice(), Slice(1)}).squeeze().bincount().max();

    torch::Tensor keep_ids =   torch::empty(edges.size(0), long_options);
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

        keep_ids.data_ptr<long>(),
        glob_count.data_ptr<int>(),

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

            keep_ids.data_ptr<long>(),
            glob_count.data_ptr<int>(),

            max_binsize.item<int>(),
            imgid.size(0),
            keep_frac,
            threshold
    );
    CudaCheckError();

    keep_ids = keep_ids.narrow(0, 0, glob_count.item<int>());
//    return {keep_ids};

    return {keep_ids, edges,scores, g_binscores, g_edgeids, g_bincounts, glob_count, max_binsize  };
}


















