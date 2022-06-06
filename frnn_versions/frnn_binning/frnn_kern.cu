#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>

#include "frnn_kern.h"

__device__ void LoadBin(const float* pts, const int* bin_ptids, float* bin,
                        int bin_start, int bin_size,
                        int pt_size1,
                        int bin_stride, int thid, int block_threads)
{
    for (int i = thid; i < bin_size; i += block_threads)
    {
        if (i >= bin_size){ break; }

        int pt_id = bin_ptids[ bin_start + i ];

        bin[i * bin_stride + 0] = pts[pt_id * pt_size1 + 0];
        bin[i * bin_stride + 1] = pts[pt_id * pt_size1 + 1];
        bin[i * bin_stride + 2] = pts[pt_id * pt_size1 + 2];
        bin[i * bin_stride + 3] = pts[pt_id * pt_size1 + 4];
        bin[i * bin_stride + 4] = pts[pt_id * pt_size1 + 5];
    }
}


__device__ void CompareBin(
                            const float* pts, int* edges, int* glob_counts,
                            float* bin_a, float* bin_b,
                            const int bin_Awidth, const int bin_Bwidth,
                            float radius, float scale_radius,

                            const int neighb_size1,
                            const int pt_size1,
                            const int edges_size0,
                            int bin_stride, int thid, int thid_grid, int block_threads,
                            bool samebin_flag )
{

    for (int ia = threadIdx.x; ia < bin_Awidth; ia += blockDim.x)
    {
        if (ia >= bin_Awidth) { break; }

        float ay =      bin_a[ia * bin_stride + 0];
        float ax =      bin_a[ia * bin_stride + 1];
        float az =      bin_a[ia * bin_stride + 2];
        float as =      bin_a[ia * bin_stride + 3];
        float a_ptid =  bin_a[ia * bin_stride + 4];


        for (int ib = threadIdx.y; ib < bin_Bwidth; ib += blockDim.y)
        {
            if (ib >= bin_Bwidth) { break; }

            float b_ptid =  bin_b[ib * bin_stride + 4];

            if ( (fabs(a_ptid - b_ptid) > 0.5) && !(samebin_flag && (a_ptid - b_ptid < 0)) )
            {
                float by =  bin_b[ib * bin_stride + 0];
                float bx =  bin_b[ib * bin_stride + 1];
                float bz =  bin_b[ib * bin_stride + 2];
                float bs =  bin_b[ib * bin_stride + 3];

                float diffx = bx - ax;
                float diffy = by - ay;
                float diffz = bz - az;

                float dist = sqrtf( powf(diffx,2) + powf(diffy,2) + powf(diffz,2) );

                if ( (dist < radius * sqrtf(as*bs)) && (fabsf(as - bs) < scale_radius) ){

                    int thread_i = atomicAdd(glob_counts, 2);
                    edges[thread_i + 0] = static_cast<int>( a_ptid );
                    edges[thread_i + 1] = static_cast<int>( b_ptid );
                }
            }

        }
    }
}


__global__ void frnn_cuda_forward_kernel(

    const   int*        __restrict__   neighbors,
    const   int*        __restrict__   bin_counts,
    const   int*        __restrict__   bin_offsets,
    const   int*        __restrict__   bin_ptids,
    const   float*      __restrict__   pts,

            int*                       glob_counts,
            int*                       edges,

    const float         radius,
    const float         scale_radius,

    const int           max_binsize,
    const int           bin_stride,

    const int           neighb_size0,
    const int           neighb_size1,

    const int           pt_size0,
    const int           pt_size1,

    const int           edges_size0
){
    __shared__ int bin_Astart[1];
    __shared__ int bin_Anum[1];
    __shared__ int bin_Bstart[1];
    __shared__ int bin_Bnum[1];
    __shared__ bool run_flag[1];
    __shared__ bool samebin_flag[1];

    extern __shared__ float s[];
    float* bin_a = s;
    float* bin_b = (float*)& bin_a[ max_binsize * bin_stride ];

    int thid_grid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x + threadIdx.x);
    int block_threads = blockDim.x * blockDim.y;
    int thid = threadIdx.y * blockDim.x + threadIdx.x;


    if (thid == 0){
        run_flag[0] = true;

        int binA = neighbors[ blockIdx.y * neighb_size1 + 0 ];
        if (binA >= 0){
            *bin_Astart = bin_offsets[ binA ];
            *bin_Anum =  bin_counts[ binA ];
        } else { run_flag[0] = false; }

        int binB = neighbors[ blockIdx.y * neighb_size1 + blockIdx.x ];
        if (binB >= 0){
            *bin_Bstart = bin_offsets[ binB ];
            *bin_Bnum =   bin_counts[ binB ];
        } else { run_flag[0] = false; }

        *samebin_flag = binA == binB;
    }
    __syncthreads();
    if (!run_flag[0]){ return; }

    LoadBin(pts, bin_ptids, bin_a, *bin_Astart, *bin_Anum, pt_size1, bin_stride, thid, block_threads);

    LoadBin(pts, bin_ptids, bin_b, *bin_Bstart, *bin_Bnum, pt_size1, bin_stride, thid, block_threads);

    CompareBin(
                pts, edges, glob_counts,
                bin_a, bin_b, *bin_Anum, *bin_Bnum,
                radius, scale_radius,

                neighb_size1,
                pt_size1,
                edges_size0,
                bin_stride, thid, thid_grid, block_threads,
                *samebin_flag );
}

void frnn_kern::frnn_kern_launch(

    int* neighbors,
    int* bin_counts,
    int* bin_offsets,
    int* bin_ptids,
    float* pts,

    int* glob_counts,
    int* edges,

    float radius,
    float scale_radius,

    int max_binsize,
    int bin_stride,

    int neighb_size0,
    int neighb_size1,

    int pts_size0,
    int pts_size1,

    int edges_size0,

    dim3 blocks,
    dim3 threads,
    size_t shared
){
    frnn_cuda_forward_kernel<<<blocks, threads, shared>>>(

        neighbors,
        bin_counts,
        bin_offsets,
        bin_ptids,
        pts,

        glob_counts,
        edges,

        radius,
        scale_radius,

        max_binsize,
        bin_stride,

        neighb_size0,
        neighb_size1,

        pts_size0,
        pts_size1,

        edges_size0
    );

}
