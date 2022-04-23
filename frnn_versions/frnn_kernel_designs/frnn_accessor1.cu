#include <ATen/ATen.h>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <math.h>
#include <stdio.h>





__global__ void frnn_cuda_forward_kernel(
    const torch::PackedTensorAccessor<int,      2,torch::RestrictPtrTraits,size_t> neighbor_bins,

    const torch::PackedTensorAccessor<float,    2,torch::RestrictPtrTraits,size_t> pts,

    const torch::PackedTensorAccessor<int,      1,torch::RestrictPtrTraits,size_t> pt_idxs,

    const torch::PackedTensorAccessor<int,      1,torch::RestrictPtrTraits,size_t> first_pt_idxs,

    const torch::PackedTensorAccessor<float,    0,torch::RestrictPtrTraits,size_t> radius,

    const torch::PackedTensorAccessor<float,    0,torch::RestrictPtrTraits,size_t> scale_radius,

    const int n_max_neighbors,
    const int n_bins,

    torch::PackedTensorAccessor<int,      2,torch::RestrictPtrTraits,size_t> edges,

    torch::PackedTensorAccessor<int,      0,torch::RestrictPtrTraits,size_t> i_edges
    )
{

    const int n_max_pts_bin = 1000;

    const int pt_size = 4;
    const int edge_size = 2;

    const int offset_x = 0;
    const int offset_y = 1;
    const int offset_z = 2;
    const int offset_s = 3;


    // stores the points for bin_a
    __shared__ float bin_a[n_max_pts_bin * pt_size];
    __shared__ float bin_b[n_max_pts_bin * pt_size];

    // stores the pt ids for bin_a
    __shared__ int bin_a_ids[n_max_pts_bin];
    __shared__ int bin_b_ids[n_max_pts_bin];

    for (int idx_i_bin_a = blockIdx.x;  idx_i_bin_a < n_bins;  idx_i_bin_a += gridDim.x)
    {

        __syncthreads();
        __threadfence();

        int i_bin_a = idx_i_bin_a;

        // if the bin is empty:
        if (first_pt_idxs[i_bin_a] == -1) {
            continue;
        }

        //////////////
        // load bin_a
        //////////////
        if (threadIdx.x == 0 && threadIdx.y == 0) {

            bool set_neg_1 = false;
            int inext = first_pt_idxs[i_bin_a];
            for (int i=0; i < n_max_pts_bin; i++) {

                if (set_neg_1) {
                    bin_a_ids[i] = -1;
                    continue;
                }

                bin_a_ids[i] = inext;

                if (inext == -1) {
                    set_neg_1 = true;
                    continue;
                }

                int i_pts_start = inext * pt_size;
                int i_bin = i * pt_size;
                bin_a[i_bin + offset_x] = pts[i_pts_start][offset_x];
                bin_a[i_bin + offset_y] = pts[i_pts_start][offset_y];
                bin_a[i_bin + offset_z] = pts[i_pts_start][offset_z];
                bin_a[i_bin + offset_s] = pts[i_pts_start][offset_s];

                inext = pt_idxs[inext];
            }
        }

        for (int idx_i_bin_b = 0;
            idx_i_bin_b < n_max_neighbors;
            idx_i_bin_b += 1) {

            // int idx_i_bin_b = blockIdx.y;
            int i_bin_b = neighbor_bins[i_bin_a][idx_i_bin_b];

            // neighboring bins in the matrix that are empty
            // should have been set to -1
            // but there might be more bin_b's that
            // are not -1 after a bin_b that is -1 . . .
            if (i_bin_b == -1) {continue;}

            if (first_pt_idxs[i_bin_b] == -1) {
                continue;}

            // don't double check bin pairs
            if (i_bin_b < i_bin_a) {continue;}

            /*----------
            LOAD BIN B
            -----------*/
            int set_neg_1 = false;
            int inext = first_pt_idxs[i_bin_b];
            for (int i=0; i < n_max_pts_bin; i++) {

                if (set_neg_1) {
                    bin_b_ids[i] = -1;
                    continue;
                }

                bin_b_ids[i] = inext;

                if (inext == -1) {
                    set_neg_1 = true;
                    continue;
                }

                int i_pts_start = inext * pt_size;
                int i_bin = i * pt_size;
                bin_b[i_bin + offset_x] = pts[i_pts_start][offset_x];
                bin_b[i_bin + offset_y] = pts[i_pts_start][offset_y];
                bin_b[i_bin + offset_z] = pts[i_pts_start][offset_z];
                bin_b[i_bin + offset_s] = pts[i_pts_start][offset_s];

                inext = pt_idxs[inext];
            }

            __syncthreads();
            __syncthreads();
            __threadfence();




            /*---------------------
            THE COMPARISIONS
            now do the comparison between
            bin_a's pts and bin_b's pts
            ----------------------*/

            // ia is the bin index for the current pt a
            // so it is NOT the index into the pts matrix
            for (int ia = threadIdx.x; ia < n_max_pts_bin; ia+=blockDim.x) {

                if (ia >= n_max_pts_bin) {break;}

                if (bin_a_ids[ia] <= -1) {break;}

                float ax = bin_a[ia * pt_size + offset_x];
                float ay = bin_a[ia * pt_size + offset_y];
                float az = bin_a[ia * pt_size + offset_z];
                float as = bin_a[ia * pt_size + offset_s];

                for (int ib = threadIdx.y; ib < n_max_pts_bin; ib+=blockDim.y) {

                    if (ib >= n_max_pts_bin) {break;}

                    if (bin_b_ids[ib] <= -1) {break;}

                    // don't compare the same point to itself:
                    if (bin_b_ids[ib] == bin_a_ids[ia]) {continue;}

                    // if it's the same bin,
                    // only compare lower points to higher points
                    if ((i_bin_a == i_bin_b) && (bin_b_ids[ib] <= bin_a_ids[ia])) {
                        continue;
                    }

                    float bx = bin_b[ib * pt_size + offset_x];
                    float by = bin_b[ib * pt_size + offset_y];
                    float bz = bin_b[ib * pt_size + offset_z];
                    float bs = bin_b[ib * pt_size + offset_s];

                    // check that the scales are close enough
                    float scale_max = as;
                    float scale_min = bs;
                    if (as < bs) {
                        scale_max = bs;
                        scale_min = as;
                    }

                    if ((scale_max / scale_min) > scale_radius]) {continue;}

                    float diffx = bx - ax;
                    float diffy = by - ay;
                    float diffz = bz - az;
                    float dist = diffx * diffx + diffy * diffy + diffz * diffz;
                    dist = sqrt(dist);

                    float log_avg_scale = sqrt(as * bs);
                    if (dist >= radius[0] * log_avg_scale) {continue;}

                    int this_i_edges = atomicAdd(&i_edges[0], edge_size);
                    edges[this_i_edges][0] = bin_a_ids[ia];
                    edges[this_i_edges][1] = bin_b_ids[ib];
                }
            }
            __syncthreads();
            __threadfence();
        }
        __syncthreads();
        __threadfence();
    }

    return;
}







std::vector<torch::Tensor> frnn_cuda_forward(
        torch::Tensor neighbor_bins,
        torch::Tensor pts,
        torch::Tensor pt_idxs,
        torch::Tensor first_pt_idxs,
        torch::Tensor radius,
        torch::Tensor scale_radius )
{

    auto rad = radius.item();

    printf("rad: %i", find);


    const int n_threadsx = 16;
    const int n_threadsy = 16;
    //const int n_threads = 1024;

    const int n_edges0 = 10000000;
    const int n_edges1 = 2;


    //const dim3 blocks((n_edgesx*n_edgesy + n_threads - 1) / n_threads);

    const int blocks = 8192;
    const dim3 threads(n_threadsx, n_threadsy);

    const auto n_bins = neighbor_bins.size(0);
    const auto n_max_neighbors = neighbor_bins.size(1);

    auto edges = torch::zeros({n_edges0, n_edges1}, torch::kInt32).sub_(1);
    auto i_edges = torch::zeros(1, torch::kInt32);


    frnn_cuda_forward_kernel<<<blocks, threads>>>(

        neighbor_bins.packed_accessor<int,      2,torch::RestrictPtrTraits,size_t>(),

        pts.packed_accessor<float,              2,torch::RestrictPtrTraits,size_t>(),

        pt_idxs.packed_accessor<int,            1,torch::RestrictPtrTraits,size_t>(),

        first_pt_idxs.packed_accessor<int,      1,torch::RestrictPtrTraits,size_t>(),

        radius.packed_accessor<float,           0,torch::RestrictPtrTraits,size_t>(),

        scale_radius.packed_accessor<float,     0,torch::RestrictPtrTraits,size_t>(),

        n_max_neighbors,
        n_bins,

        edges.packed_accessor<int,              2,torch::RestrictPtrTraits,size_t>(),

        i_edges.packed_accessor<int,            0,torch::RestrictPtrTraits,size_t>() );

    return {edges, i_edges, neighbor_bins, pts, pt_idxs, first_pt_idxs, radius, scale_radius};
}











