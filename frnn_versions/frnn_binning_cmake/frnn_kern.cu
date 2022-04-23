#include <torch/types.h>

#include <stdio.h>
#include <iostream>


// load a bin's pts, where we are interested in [y, x, z, scale, ptid] at columns
// [0,1,2,4,5] in pts. A bin starts at bin_start in bin_ptids and includes contiguous
// rows, of number bin_size. bin_ptids gives indexes into pts.
template <typename scalar_t>
__device__ void LoadBin(const scalar_t* pts, const int* bin_ptids, float* bin,
                        int bin_start, int bin_size,
                        const int pt_size1,
                        const int bin_stride)
{
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < bin_size; i += blockDim.x * blockDim.y)
    {
        int pt_id = bin_ptids[ bin_start + i ];

        bin[i * bin_stride + 0] = float( pts[pt_id * pt_size1 + 0] );
        bin[i * bin_stride + 1] = float( pts[pt_id * pt_size1 + 1] );
        bin[i * bin_stride + 2] = float( pts[pt_id * pt_size1 + 2] );
        bin[i * bin_stride + 3] = float( pts[pt_id * pt_size1 + 4] );
        bin[i * bin_stride + 4] = float( pt_id );
    }
}

// compare two bins worth of pts for frnn criteria; write to edges. The samebin flag indicates that
// bin_a and bin_b are the same; avoid redundant edges by enforcing ptid_diff > 0 in this case.
__device__ void CompareBin(
        long* edges, int* glob_counts,
        float* bin_a, float* bin_b, int bin_Awidth, int bin_Bwidth,
        float radius, float scale_radius,
        int bin_stride, bool samebin )
{
    int thid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = (blockDim.x * blockDim.y);

    for (int ia = thid; ia < bin_Awidth; ia += block_threads)
    {
        float ay =      bin_a[ia * bin_stride + 0];
        float ax =      bin_a[ia * bin_stride + 1];
        float az =      bin_a[ia * bin_stride + 2];
        float as =      bin_a[ia * bin_stride + 3];
        float a_ptid =  bin_a[ia * bin_stride + 4];

        for (int ib = 0; ib < bin_Bwidth; ib++)
        {
            float by =      bin_b[ib * bin_stride + 0];
            float bx =      bin_b[ib * bin_stride + 1];
            float bz =      bin_b[ib * bin_stride + 2];
            float bs =      bin_b[ib * bin_stride + 3];
            float b_ptid =  bin_b[ib * bin_stride + 4];

            float diffx = bx - ax;
            float diffy = by - ay;
            float diffz = bz - az;

            float dist = sqrtf( diffx*diffx + diffy*diffy + diffz*diffz );

            float ptid_diff = a_ptid - b_ptid;

            // a_ptid is not b_ptid
            bool check0 = fabsf(ptid_diff) > 0.2;

            // if same bin a and b, no redundant edges written
            bool check1 = (!samebin) || (ptid_diff > 0);

            // frnn criteria in linear, scale space
            bool check2 = dist < ( radius * sqrtf(as*bs) ) && fabsf(logf(as) - logf(bs)) < scale_radius;

            // write valid edge
            if (check0 && check1 && check2)
            {
                int thread_i = atomicAdd(glob_counts, 2);
                edges[thread_i + 0] = long(a_ptid);
                edges[thread_i + 1] = long(b_ptid);
            }
        }
    }

}

// each block loads two bins into shared mem for comparison, including comparing the central bin at column 0 to itself.
template <typename scalar_t>
__global__ void frnn_main_kernel(

        const   int*    neighbors,
        const   int*     bin_counts,
        const   int*     bin_offsets,
        const   int*     bin_ptids,
        const   scalar_t*   pts,

        int*      glob_counts,
        long*     edges,

        const float    radius,
        const float    scale_radius,

        const int      max_binsize,
        const int      bin_stride,

        const int      neighb_size1,
        const int      pt_size1
){

    extern __shared__ float s[];
    float* bin_a = s;
    float* bin_b = (float*)& bin_a[ max_binsize * bin_stride ];

    int __shared__ bin_Astart[1];
    int __shared__ bin_Anum[1];
    int __shared__ bin_Bstart[1];
    int __shared__ bin_Bnum[1];
    bool __shared__ samebin[1];
    bool __shared__ end[1];

    int thid = threadIdx.y * blockDim.x + threadIdx.x;
    if (thid == 0)
    {
        *end = false;

        int binA = neighbors[ blockIdx.y * neighb_size1 + 0 ];
        if (binA >= 0){
            *bin_Astart = bin_offsets[ binA ];
            *bin_Anum =   bin_counts[ binA ];
        } else { *end = true; }

        int binB = neighbors[ blockIdx.y * neighb_size1 + blockIdx.x ];
        if (binB >= 0){
            *bin_Bstart = bin_offsets[ binB ];
            *bin_Bnum =   bin_counts[ binB ];
        } else { *end = true; }

        *samebin = binA == binB;
    }
    __syncthreads();
    if ( *end ) { return; }

    LoadBin<scalar_t>(pts, bin_ptids, bin_a, *bin_Astart, *bin_Anum, pt_size1, bin_stride);

    LoadBin<scalar_t>(pts, bin_ptids, bin_b, *bin_Bstart, *bin_Bnum, pt_size1, bin_stride);
    __syncthreads();

    CompareBin(
            edges, glob_counts,
            bin_a, bin_b, *bin_Anum, *bin_Bnum,
            radius, scale_radius,

            bin_stride, *samebin
    );
}

__host__ void frnn_kern_launch(

    int*    neighbors,
    int*    bin_counts,
    int*    bin_offsets,
    int*    bin_ptids,
    torch::Tensor pts,

    int*    glob_counts,
    long*   edges,

    float radius,
    float scale_radius,

    int max_binsize,
    int bin_stride,

    int neighb_size1,
    int pt_size1,

    dim3 blocks,
    dim3 threads,
    size_t shared
){
    AT_DISPATCH_FLOATING_TYPES_AND(torch::ScalarType::Half, pts.scalar_type(), "frnn_main_kernel", ([&] {
        frnn_main_kernel<<<blocks, threads, shared>>>(

            neighbors,
            bin_counts,
            bin_offsets,
            bin_ptids,
            pts.data_ptr<scalar_t>(),

            glob_counts,
            edges,

            radius,
            scale_radius,

            max_binsize,
            bin_stride,

            neighb_size1,
            pt_size1
        );
    }));

}
