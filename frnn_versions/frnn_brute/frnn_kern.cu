#include <torch/types.h>

#include <stdio.h>
#include <iostream>




//template <typename scalar_t>
__global__ void frnn_main_kernel(
    const float* pts,
    const int    pts_size0,
    const int    pts_size1,

          long* edges,
          int*  glob_count,
    const long* im_counts,
    const long* im_offsets,
    const int   batch_size,

    const float lin_radius,
    const float scale_radius
){
    int imid = blockIdx.x;
    int row_start = im_offsets[imid];
    int row_end = row_start + im_counts[imid];

    for (int row_a = row_start + threadIdx.x; row_a < row_end; row_a += blockDim.x)
    {
        for (int row_b = row_a + 1; row_b < row_end; row_b++)
        {
            if (row_b >= pts_size0) continue;

            float ay =      pts[row_a * pts_size1 + 0];
            float ax =      pts[row_a * pts_size1 + 1];
            float as =      pts[row_a * pts_size1 + 4];

            float by =      pts[row_b * pts_size1 + 0];
            float bx =      pts[row_b * pts_size1 + 1];
            float bs =      pts[row_b * pts_size1 + 4];

            float diffy = by - ay;
            float diffx = bx - ax;

            float dist = sqrtf( diffx*diffx + diffy*diffy );

            bool check = (dist < ( lin_radius * sqrtf(as*bs) )) && (fabsf(logf(as) - logf(bs)) < scale_radius);

            if (check)
            {
                int thread_i = atomicAdd(glob_count, 2);
                edges[thread_i + 0] = long(row_a);
                edges[thread_i + 1] = long(row_b);
            }
        }
    }

}

__host__ void frnn_kern_launch(

    torch::Tensor pts,

    torch::Tensor edges,
    torch::Tensor glob_count,
    torch::Tensor im_counts,
    torch::Tensor im_offsets,
    torch::Tensor batch_size,

    torch::Tensor lin_radius,
    torch::Tensor scale_radius,

    dim3 blocks,
    dim3 threads
){
        frnn_main_kernel<<<blocks, threads>>>(
            pts.data_ptr<float>(),
            int(pts.size(0)),
            int(pts.size(1)),

            edges.data_ptr<long>(),
            glob_count.data_ptr<int>(),
            im_counts.data_ptr<long>(),
            im_offsets.data_ptr<long>(),
            batch_size.item<int>(),

            lin_radius.item<float>(),
            scale_radius.item<float>()
        );
}
