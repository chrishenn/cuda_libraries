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








__global__ void grid_kern(
    
    const   int*   __restrict__  counts,
    const   int*   __restrict__  offsets,
    const   int*   __restrict__  binid,
    const   float* __restrict__  scores,
    const   int*   __restrict__  o_ids,

            int*   __restrict__  active_ids,
            int*   __restrict__  glob_count,
            
    const int       n_bins,
    const int       bin_size,
    const float     keep_frac   
)
{
    extern __shared__ float s[];
    float* sh_scores = s;
    int* sh_ids = (int*)& sh_scores[ n_bins*bin_size ];
    int* sh_counters = (int*)& sh_ids[ n_bins*bin_size ];
    
    int imid = blockIdx.x;
    auto bl_offset = offsets[imid];
    auto bl_objs = counts[imid];

    // zero sh_counters
    for (int i = threadIdx.x; i < n_bins; i += blockDim.x){
        sh_counters[i] = 0;
    }
    __syncthreads();
    
    // build image-bins
    for (int glob_i = bl_offset + threadIdx.x; glob_i < bl_offset+bl_objs; glob_i += blockDim.x){
    
        auto pt_bid = binid[glob_i];
        auto pt_score = scores[glob_i];
        auto pt_id = o_ids[glob_i];
    
        int write_col = atomicAdd(&sh_counters[pt_bid], 1);

        sh_ids[pt_bid * bin_size + write_col] = pt_id;
        sh_scores[pt_bid * bin_size + write_col] = pt_score;
    }    
    __syncthreads();
    
    // for each bin
    for (int loc_i = threadIdx.x; loc_i < n_bins; loc_i += blockDim.x)
    {
        // pts in this image in this bin at row loc_i
        int bin_count = sh_counters[loc_i];
        if (bin_count <= 0){ continue; }

        // there must be at least 1 object in this bin at this point. 
        int max_binloc;
        float max_score = -1000;
        
        // look through this row (this bin in this image for this thread)
        for (int col = 0; col < bin_count; col++){

            int curr_loc = loc_i*bin_size + col;
            float curr_score = sh_scores[curr_loc];
            
            if (curr_score > max_score){
                max_score = curr_score;                      
                max_binloc = curr_loc;
            }
        }

        int glob_write = atomicAdd(glob_count, 1);
        active_ids[glob_write] = sh_ids[max_binloc];        
    }  
}


/** 
cpu entry point 
**/
std::vector<torch::Tensor> one_bin_cuda_call(
    torch::Tensor counts,
    torch::Tensor offsets,
    torch::Tensor binid,
    torch::Tensor scores,
    torch::Tensor o_ids,
    
    float keep_frac,
    int n_bins,
    
    int batch_size,
    int locality_size,
        
    int device_id  
){

    cudaSetDevice(device_id);

    auto options = torch::TensorOptions()
      .dtype(torch::kInt32)
      .layout(torch::kStrided)
      .device(torch::kCUDA, device_id);
    
    const auto out_size =    binid.size(0);
    torch::Tensor active_ids =  torch::empty({out_size}, options);  
    torch::Tensor active_mask = torch::zeros({out_size}, options);   
    torch::Tensor glob_count =  torch::zeros({1}, options);
        
    const auto n_blocks =        batch_size;  
    const auto n_threads =       256;
    const dim3 blocks(n_blocks);
    const dim3 threads(n_threads);
        
    int bin_size = int( (locality_size*locality_size) + 4*(locality_size) + 4 ); 
    
    size_t fl_shared = (n_bins * bin_size)*sizeof(float);
    size_t int_shared = (n_bins * bin_size)*sizeof(int);   
    size_t count_shared = n_bins * sizeof(int); 
    size_t shared = fl_shared + int_shared + count_shared;

    if (shared > 49152){
        fprintf (stderr, "ERROR ONE_BIN: ATTEMPTED TO ALLOCATE TOO MUCH SHARED MEMORY FOR YOUR DEVICE ARCH; DECREASE OBJECT-DENSITY\n");
        fprintf (stderr, "attemped: %li bytes; max supported: 49152 bytes\n", shared);
        exit(EXIT_FAILURE);
    }
    
    grid_kern<<<blocks, threads, shared>>>(
        counts.data<int>(),
        offsets.data<int>(),
        binid.data<int>(),   
        scores.data<float>(),                
        o_ids.data<int>(),    
    
        active_ids.data<int>(),    
        glob_count.data<int>(),
    
        n_bins,
        bin_size,
        keep_frac        
    );
    CudaCheckError();
    
    active_ids = torch::_cast_Long(active_ids, false);
    active_ids = torch::narrow(active_ids, 0, 0, glob_count[0].item<int32_t>());
    active_mask.index_fill_(0, active_ids, 1);
    
    active_ids = std::get<0>(torch::sort(active_ids, -1, false));
    
    return {active_ids, active_mask, glob_count};
}


















