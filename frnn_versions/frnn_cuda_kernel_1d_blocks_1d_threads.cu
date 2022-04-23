#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

#include <vector>

// #include <iostream>
#include <string>
#include <math.h>
#include <stdio.h>

using namespace std;

namespace {


const int n_max_pts_bin = 1000;
const int pt_size = 4;
// TODO: SET THIS LARGER?
const int max_n_edges_buff = 200;
const int edge_size = 2;
const int offset_x = 0;
const int offset_y = 1;
const int offset_z = 2;
const int offset_s = 3;
// const int wait_count = 1e7;



__global__ void frnn_cuda_forward_kernel(
    const int* neighbor_bins,
    const float* pts,
    const int* pt_idxs,
    const int* first_pt_idxs,
    const float radius,
    const float scale_radius,
    const int n_max_neighbors,
    int* edges,
    int* i_edges,
    const int max_size_edges
   ) {

    // if the bin is empty:
    if (first_pt_idxs[blockIdx.x] == -1) {
        return;
    }

    assert(blockDim.x == 1);

    // stores the points for bin_a
    __shared__ float bin_a[n_max_pts_bin * pt_size];
    __shared__ float bin_b[n_max_pts_bin * pt_size];
    // stores the pt ids for bin_a
    __shared__ int bin_a_ids[n_max_pts_bin];
    __shared__ int bin_b_ids[n_max_pts_bin];
    __shared__ int edge_buff[max_n_edges_buff * edge_size];

    int i_buff = 0;

    int i_bin_a = blockIdx.x;

    //////////////
    // load bin_a
    //////////////
    int inext = first_pt_idxs[i_bin_a];

    for (int i=0; i < n_max_pts_bin; i++) {

        bin_a_ids[i] = inext;

        if (inext == -1) {break;}

        int i_pts_start = inext * pt_size;
        int i_bin = i * pt_size;
        bin_a[i_bin + offset_x] = pts[i_pts_start + offset_x];
        bin_a[i_bin + offset_y] = pts[i_pts_start + offset_y];
        bin_a[i_bin + offset_z] = pts[i_pts_start + offset_z];
        bin_a[i_bin + offset_s] = pts[i_pts_start + offset_s];

        inext = pt_idxs[inext];
    }
    /*-----------------------
    ITERATE THROUGH ALL BIN Bs
    -------------------------*/
    int i_bin_b;
    for (int idx_bin_b = 0; idx_bin_b < n_max_neighbors; idx_bin_b++) {

        i_bin_b = neighbor_bins[i_bin_a * n_max_neighbors + idx_bin_b];

        // neighboring bins in the matrix that are empty
        // should have been set to -1
        // but there might be more bin_b's that 
        // are not -1 after a bin_b that is -1 . . .
        if (i_bin_b == -1) {continue;}

        // don't double check bin pairs
        if (i_bin_b < i_bin_a) {continue;}

        // if the bin is empty:
        if (first_pt_idxs[i_bin_b] == -1) {continue;}

        /*----------
        LOAD BIN B
        -----------*/
        int inext = first_pt_idxs[i_bin_b];

        for (int i=0; i < n_max_pts_bin; i++) {

            bin_b_ids[i] = inext;

            if (inext == -1) {break;}

            int i_pts_start = inext * pt_size;
            int i_bin = i * pt_size;
            bin_b[i_bin + offset_x] = pts[i_pts_start + offset_x];
            bin_b[i_bin + offset_y] = pts[i_pts_start + offset_y];
            bin_b[i_bin + offset_z] = pts[i_pts_start + offset_z];
            bin_b[i_bin + offset_s] = pts[i_pts_start + offset_s];

            inext = pt_idxs[inext];
        }

        /*---------------------
        THE COMPARISIONS
        now do the comparison between 
        bin_a's pts and bin_b's pts
        ----------------------*/

        // ia is the bin index for the current pt a
        // so it is NOT the index into the pts matrix
        for (int ia = 0; ia < n_max_pts_bin; ia++) {

            if (ia >= n_max_pts_bin) {continue;}

            if (bin_a_ids[ia] <= -1) {break;}

            float ax = bin_a[ia * pt_size + offset_x];
            float ay = bin_a[ia * pt_size + offset_y];
            float az = bin_a[ia * pt_size + offset_z];
            float as = bin_a[ia * pt_size + offset_s];

            for (int ib = 0; ib < n_max_pts_bin; ib++) {

                if (ib >= n_max_pts_bin) {continue;}

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

                if ((scale_max / scale_min) > scale_radius) {continue;}

                float diffx = bx - ax;
                float diffy = by - ay;
                float diffz = bz - az;
                float dist = diffx * diffx + diffy * diffy + diffz * diffz;
                dist = sqrt(dist);

                float log_avg_scale = sqrt(as * bs);
                if (dist > radius / log_avg_scale) {continue;}

                // now we can add the edge to the edge buffer
                edge_buff[i_buff + 0] = bin_a_ids[ia];
                edge_buff[i_buff + 1] = bin_b_ids[ib];

                i_buff += edge_size;

                /*---------------------------
                empty the buffer in the loop:
                -----------------------------*/
                if (i_buff == max_n_edges_buff * edge_size) {
                    
                    // empty the buffer and put it in the edges array
                    int this_i_edges = atomicAdd(&i_edges[0], i_buff);

                    // make sure we don't overfill the edge output tensor
                    int e_stop = i_buff;
                    if (this_i_edges + i_buff > max_size_edges) {
                        e_stop = max_size_edges - this_i_edges;
                    }

                    for (int ie = 0; ie < e_stop; ie ++) {
                        edges[this_i_edges + ie] = edge_buff[ie];
                    }

                    i_buff = 0;
                }
            }
        }
    }

    /*---------------------------------------------
    empty the buffer one more time at the very end:
    -----------------------------------------------*/
    if (i_buff > 0) {

        int this_i_edges = atomicAdd(&i_edges[0], i_buff);

        // make sure we don't overfill the edge output tensor
        int e_stop = i_buff;
        if (this_i_edges + i_buff > max_size_edges) {
            e_stop = max_size_edges - this_i_edges;
        }

        for (int ie = 0; ie < e_stop; ie ++) {
            edges[this_i_edges + ie] = edge_buff[ie];
        }
    }

    return;
}

} // closing tag for <namespace> 




std::vector<at::Tensor> frnn_cuda_forward(
        at::Tensor neighbor_bins,
        at::Tensor pts,
        at::Tensor pt_idxs,
        at::Tensor first_pt_idxs,
        float radius,
        float scale_radius,
        at::Tensor edges,
        at::Tensor i_edges
       ) { 

    const int n_threads = 1;
    const int n_bins = neighbor_bins.size(0);
    const dim3 blocks(n_bins);
    const int n_max_neighbors = neighbor_bins.size(1);

    const int max_size_edges = edges.size(0) * edges.size(1);
  
    AT_DISPATCH_INTEGRAL_TYPES(edges.type(), "frnn_forward_cuda", ([&] {
    frnn_cuda_forward_kernel<<<blocks, n_threads>>>(
        neighbor_bins.data<int>(),
        pts.data<float>(),
        pt_idxs.data<int>(),
        first_pt_idxs.data<int>(),
        radius,
        scale_radius,
        n_max_neighbors,
        edges.data<int>(),
        i_edges.data<int>(),
        max_size_edges);
    }));

    // return {edges.slice(0, i_edges.data<int>()[0]), i_edges};
    return {edges, i_edges};
}
