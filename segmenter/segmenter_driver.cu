/**
Author: Chris Henn (https://github.com/chrishenn)
**/

#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>


// define for error checking
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


using namespace cooperative_groups;

const int X_THREADS = 32;
const int TILE_SIZE = 32;


// vectors of size 2 only; serial in each image; parallel work across images; slower than arbitrary_size
__global__ void segmenter_vec_size_2(
        const long* edges,
        const int   edges_size0,

        long* seg_ids,
        const int seg_ids_size0,

        uint8_t* seg_counts,
        float* seg_angles,
        float* seg_ang_sums,

        int* locks,
        long* imgid,
        const float seg_thresh
){
    for (int glob_i = blockIdx.x * blockDim.x + threadIdx.x; glob_i < edges_size0; glob_i += blockDim.x * gridDim.x) {
        long one_id = edges[glob_i * 2 + 0];
        long other_id = edges[glob_i * 2 + 1];

        // lock on this image for all work; serial in image
        long one_imgid = imgid[one_id];

        bool holding_lock;
        do {
            holding_lock = (atomicCAS(&locks[one_imgid], 0, -1) == 0);
            if (holding_lock)
            {
                int one_segid = seg_ids[one_id];
                int other_segid = seg_ids[other_id];

                if (one_segid != other_segid)
                {
                    float ang_one0 = seg_angles[one_segid * 2 + 0];
                    float ang_one1 = seg_angles[one_segid * 2 + 1];

                    float ang_other0 = seg_angles[other_segid * 2 + 0];
                    float ang_other1 = seg_angles[other_segid * 2 + 1];

                    float dist = pow(ang_one0 - ang_other0, 2) + pow(ang_one1 - ang_other1, 2);
                    dist = sqrt(dist);

                    if (dist < seg_thresh)
                    {
                        // one_count_new = one_count + other_count
                        // one_sum_new = one_sum + other_angle
                        // sums[one] = one_sum_new
                        // angles[one] = one_sum_new / one_count_new

                        // always add smaller group to larger group; here, add other -> one
                        auto count_one = seg_counts[one_segid];
                        auto count_other = seg_counts[other_segid];
                        if (count_one < count_other) {
                            auto tmp = one_segid;
                            one_segid = other_segid;
                            other_segid = tmp;
                        }

                        // counts[one] <-- one_count_new
                        auto one_count_new = count_one + count_other;
                        seg_counts[one_segid] = one_count_new;

                        float other_ang0 = seg_angles[other_segid * 2 + 0];
                        float other_ang1 = seg_angles[other_segid * 2 + 1];

                        float one_sum0 = seg_ang_sums[one_segid * 2 + 0];
                        float one_sum1 = seg_ang_sums[one_segid * 2 + 1];

                        // one_sum_new = one_sum + other_angle
                        float one_sum_new0 = one_sum0 + other_ang0;
                        float one_sum_new1 = one_sum1 + other_ang1;

                        // sums[one] = one_sum_new
                        seg_ang_sums[one_segid * 2 + 0] = one_sum_new0;
                        seg_ang_sums[one_segid * 2 + 1] = one_sum_new1;

                        // angles[one] = one_sum_new / one_count_new
                        seg_angles[one_segid * 2 + 0] = one_sum_new0 / float(one_count_new);
                        seg_angles[one_segid * 2 + 1] = one_sum_new1 / float(one_count_new);

                        // seg_ids[seg_ids == other] <-- one
                        // no need for atomicCas because work is serial in this image and thus on these seg_ids
                        for (int i = 0; i < seg_ids_size0; i++) {
                            if (seg_ids[i] == other_segid) seg_ids[i] = one_segid;
                        }
                    }
                }

                // make this thread's changes visible to all threads on device, without synchronizing
                __threadfence();
                atomicExch( &locks[one_imgid], 0 );
            }
        } while (!holding_lock);

    }
}





// serial in each image; parallel work across images
__global__ void seg_arbitrary_vec_size(

              long* edges,
        const int   edges_size0,

              long* seg_ids,
        const int seg_ids_size0,

              uint8_t* seg_counts,
              float* seg_angles,
              float* seg_ang_sums,
        const int vec_size,

              int* locks,
              long* imgid,
        const float seg_thresh
) {

    thread_block block = this_thread_block();
    thread_block_tile<TILE_SIZE> g = tiled_partition<TILE_SIZE>(block);

    int glob_groupid = int( (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE );
    int block_groupid = int( block.thread_rank() / TILE_SIZE );
    auto groups_per_bl = X_THREADS;

    // shared
    extern __shared__ float s[];
    float* vec_buffer = s;
    float* s_dists = (float*) &vec_buffer[ groups_per_bl * vec_size ];

    // confirmed glob_i visits each edge. Confirmed each edge is visited exactly once.
    for (int glob_i = glob_groupid; glob_i < edges_size0; glob_i += groups_per_bl * gridDim.x)
    {
        auto one_id = edges[glob_i * 2 + 0];
        auto other_id = edges[glob_i * 2 + 1];

        // lock on this image for all work, both read and write; each edge's work is serial in each image
        long one_imgid = imgid[one_id];
        bool holding_lock = false;

        do
        {
            if (g.thread_rank() == 0) holding_lock = (atomicCAS(&locks[one_imgid], 0, -1) == 0);

            if (g.shfl(holding_lock, 0))
            {
                auto one_segid = seg_ids[one_id];
                auto other_segid = seg_ids[other_id];

                if (one_segid != other_segid)
                {
                    // zero s_dists for this group
                    if (g.thread_rank() == 0) s_dists[block_groupid] = 0;

                    // read (angle[one] - angle[other])^2 element-wise and store in buffer
                    for (int i = g.thread_rank(); i < vec_size; i += g.size()) {
                        float ang_oneseg = seg_angles[one_segid * vec_size + i];
                        float ang_otherseg = seg_angles[other_segid * vec_size + i];

                        vec_buffer[block_groupid * vec_size + i] = pow(ang_oneseg - ang_otherseg, 2);
                    }
                    g.sync();

                    for (int i = g.thread_rank(); i < vec_size; i += g.size()) {
                        atomicAdd(&s_dists[block_groupid], vec_buffer[block_groupid * vec_size + i]);
                    }
                    g.sync();

                    float dist = sqrt(s_dists[block_groupid]);
                    if (dist < seg_thresh) {

                        // always add smaller group to larger group; here, add other -> one
                        auto one_count = seg_counts[one_segid];
                        auto other_count = seg_counts[other_segid];
                        if (one_count < other_count) {
                            auto tmp = one_segid;
                            one_segid = other_segid;
                            other_segid = tmp;
                        }

                        // counts[one] <-- one_count_new
                        auto one_count_new = one_count + other_count;
                        if (g.thread_rank() == 0) seg_counts[one_segid] = one_count_new;

                        // read into buffer <-- angle[other]
                        for (int i = g.thread_rank(); i < vec_size; i += g.size()) {
                            vec_buffer[block_groupid * vec_size + i] = seg_angles[other_segid * vec_size + i];
                        }
                        g.sync();

                        // addition element-wise to buffer += sums[one]
                        for (int i = g.thread_rank(); i < vec_size; i += g.size()) {
                            vec_buffer[block_groupid * vec_size + i] += seg_ang_sums[one_segid * vec_size + i];
                        }
                        g.sync();
                        // buffer now holds one_sum_new

                        // write one_sum_new from buffer --> seg_ang_sums[one]
                        for (int i = g.thread_rank(); i < vec_size; i += g.size()) {
                            seg_ang_sums[one_segid * vec_size + i] = vec_buffer[block_groupid * vec_size + i];
                        }

                        // write (one_sum_new<in buffer> / one_count_new) --> angles[one]
                        for (int i = g.thread_rank(); i < vec_size; i += g.size()) {
                            seg_angles[one_segid * vec_size + i] = vec_buffer[block_groupid * vec_size + i] / float(one_count_new);
                        }

                        // reassign seg_ids[seg_ids == other] <-- one
                        // no need for atomicCas because work is serial in this image and thus on these seg_ids
                        for (int i = g.thread_rank(); i < seg_ids_size0; i += g.size()) {
                            if (seg_ids[i] == other_segid) seg_ids[i] = one_segid;
                        }
                    }
                }
                g.sync();

                // unlock this image for another thread group
                if (g.thread_rank() == 0) atomicExch(&locks[one_imgid], 0);
            }

        } while (!g.shfl(holding_lock, 0));

    }

}




/** 
cpu entry point for python extension.
**/
std::vector<torch::Tensor> segmenter_call(
    torch::Tensor edges,
    torch::Tensor imgid,
    torch::Tensor batch_size,

    torch::Tensor angles,
    torch::Tensor seg_thresh
) {

    // set device
    auto device = angles.get_device();
    cudaSetDevice(device);

    // tensor allocation
    auto int_opt = torch::TensorOptions()
            .dtype(torch::kI32)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device)
            .requires_grad(false);
    auto long_opt = torch::TensorOptions()
            .dtype(torch::kI64)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device)
            .requires_grad(false);
    auto uint8_opt = torch::TensorOptions()
            .dtype(torch::kUInt8)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device)
            .requires_grad(false);

    auto seg_ids = torch::arange(angles.size(0), long_opt);
    auto seg_counts = torch::ones(angles.size(0), uint8_opt);
    auto seg_angles = angles.clone();
    auto seg_ang_sums = angles.clone();

    auto locks = torch::zeros(batch_size.item<int>(), int_opt);

    auto vec_size = seg_angles.size(1);

    // calculate grid size
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    int sms = deviceProp.multiProcessorCount;

    int full_cover = (edges.size(0) - 1) / X_THREADS + 1;
    int n_blocks = min(full_cover, 2 * sms);

    const dim3 blocks(n_blocks);
    const dim3 threads(X_THREADS * TILE_SIZE);

    auto groups_per_bl = X_THREADS;
    auto shared = (groups_per_bl * vec_size + groups_per_bl) * sizeof(float);

    seg_arbitrary_vec_size<<<blocks, threads, shared>>>(
            edges.data_ptr<long>(),
            edges.size(0),

            seg_ids.data_ptr<long>(),
            seg_ids.size(0),

            seg_counts.data_ptr<uint8_t>(),
            seg_angles.data_ptr<float>(),
            seg_ang_sums.data_ptr<float>(),
            vec_size,

            locks.data_ptr<int>(),
            imgid.data_ptr<long>(),
            seg_thresh.item<float>()
    ); CudaCheckError();

    return {seg_ids};
}



