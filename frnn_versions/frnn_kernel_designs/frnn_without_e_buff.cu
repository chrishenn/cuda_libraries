#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <iostream>
#include <string>
#include <math.h>
//#include <helper_cuda.h>
using namespace std; 

namespace {


const int n_max_pts_bin = 1000;
const int pt_size = 4;
// TODO: SET THIS LARGER?
// const int n_edges_buff = 64;
const int edge_size = 2;
const int offset_x = 0;
const int offset_y = 1;
const int offset_z = 2;
const int offset_s = 3;
const int wait_count = 1e7;



__global__ void frnn_cuda_forward_kernel(
    const int* __restrict__ neighbor_bins,
    const float* __restrict__ pts,
    const int* __restrict__ pt_idxs,
    const int* __restrict__ first_pt_idxs,
    const float radius,
    const float scale_radius,
    const int n_max_neighbors,
    const int n_max_edges_per_bin,
    int* __restrict__ edges, int max_edges
   ) {

    // printf("bl %d first pt id = %d \n", blockIdx.x, first_pt_idxs[blockIdx.x] );
    if (first_pt_idxs[blockIdx.x] == -1) {
        // printf("bl %d block is empty ... i_bin_a = %d\n", blockIdx.x, first_pt_idxs[blockIdx.x] );
        return;
    }
    // printf("bl %d block not empty ... i_bin_a = %d\n", blockIdx.x, first_pt_idxs[blockIdx.x] );

    /* 
    - Thread 0 loads bin_a
    - Thread 1 loads bin_b
    - Thread 0 resets the edge buffer
    */

    int thread_offset = 2;

    // __shared__ int i_edges;
    __shared__ int i_bin_a;
    // the index for the next value in the buffer
    // __shared__ int atomic_i_buff;
    __shared__ int i_edges_start;
    __shared__ int atomic_i_edges;
    __shared__ int max_atomic_i_edges;
    // counts how many edges are in the buffer
    // (will be different than the number of elements
    // if edge_size > 0)
    // __shared__ int atomic_buff_counter;
    // keeps track of how many edges still need to
    // compare points
    __shared__ int threads_left_counter;
    // stores the points for bin_a
    __shared__ float bin_a[n_max_pts_bin * pt_size]; 
    __shared__ float bin_b[n_max_pts_bin * pt_size];
    // stores the pt ids for bin_a
    __shared__ int bin_a_ids[n_max_pts_bin];
    __shared__ int bin_b_ids[n_max_pts_bin];
    // __shared__ int edge_buff[n_edges_buff * edge_size];

    //////////////
    // load bin_a
    //////////////
    if (threadIdx.x == 0) {

        i_bin_a = blockIdx.x;
        i_edges_start = blockIdx.x * n_max_edges_per_bin * edge_size;
        atomic_i_edges = 0;
        max_atomic_i_edges = (n_max_edges_per_bin - 1) * edge_size;
        // atomic_i_buff = 0;
        // atomic_buff_counter = 0;
        threads_left_counter = 0;

        int ia = first_pt_idxs[i_bin_a];

        for (int i=0; i < n_max_pts_bin; i++) {

            bin_a_ids[i] = ia;

            if (ia == -1) {break;}

            bin_a[i * pt_size + offset_x] = pts[ia * pt_size + offset_x];
            bin_a[i * pt_size + offset_y] = pts[ia * pt_size + offset_y];
            bin_a[i * pt_size + offset_z] = pts[ia * pt_size + offset_z];
            bin_a[i * pt_size + offset_s] = pts[ia * pt_size + offset_s];

            ia = pt_idxs[ia];
        }
    }

    int i_bin_b;
    for (int idx_bin_b=0; idx_bin_b<n_max_neighbors; idx_bin_b++) {

        i_bin_b = neighbor_bins[i_bin_a * n_max_neighbors + idx_bin_b];

        // neighboring bins in the matrix that are empty
        // should have been set to -1
        // but there might be more bin_b's that 
        // are not -1 after a bin_b that is -1 . . .
        if (i_bin_b == -1) {continue;}

        // don't double check bin pairs
        if (i_bin_b < i_bin_a) {continue;}
        // TODO: THIS IS COMMENTED OUT WHEN PTS ARE SKIPPED IN THE SAME WAY
        // if (i_bin_b > i_bin_a) {continue;} 

        ////////////////////////
        // thread 1 loads bin_b
        ////////////////////////
        if (threadIdx.x == 1) {

            while (threads_left_counter != 0) {
                // wait:
                // while ((atomic_i_buff / edge_size) < n_edges_buff &&
                //     atomic_buff_counter < n_edges_buff &&
                //     threads_left_counter > 0) {
                clock_t start_clock = clock64();
                clock_t offset_clock = 0;
                while (offset_clock < wait_count) {
                    offset_clock = clock64() - start_clock;
                }
                // }
            }

            threads_left_counter = blockDim.x;

            int ib = first_pt_idxs[i_bin_b];
            // printf("bl %d to load i_bin_b = %d \n", blockIdx.x, i_bin_b);
            for (int i=0; i < n_max_pts_bin; i++) {

                bin_b_ids[i] = ib;

                // TODO: CHECK THAT WE DON'T HAVE 
                // MORE POINTS AFTER A -1 PT...
                if (ib == -1) {break;}

                bin_b[i * pt_size + offset_x] = pts[ib * pt_size + offset_x];
                bin_b[i * pt_size + offset_y] = pts[ib * pt_size + offset_y];
                bin_b[i * pt_size + offset_z] = pts[ib * pt_size + offset_z];
                bin_b[i * pt_size + offset_s] = pts[ib * pt_size + offset_s];

                ib = pt_idxs[ib];
            }
        }

        __syncthreads();

        // check if bin_b is empty
        if (bin_b_ids[0] == -1) {
            threads_left_counter = 0;
            continue;}

        // now do the comparison between bin_a and all bin_b's
        // thread 0 and thread 1 don't do this step
        int ia;
        for (int ja = 0; ja < n_max_pts_bin; ja++) {

            ia = ja * (blockDim.x - thread_offset) + threadIdx.x - thread_offset;
            if (ia >= n_max_pts_bin) {continue;}
            
            // TODO: CHECK THAT WE DON'T HAVE 
            // MORE POINTS AFTER A -1 PT...
            if (bin_a_ids[ia] <= -1) {break;}
        
            // thread 0 and 1 don't do the comparisions
            if (threadIdx.x == 0 or threadIdx.x == 1) {break;}

            float ax = bin_a[ia * pt_size + offset_x];
            float ay = bin_a[ia * pt_size + offset_y];
            float az = bin_a[ia * pt_size + offset_z];
            float as = bin_a[ia * pt_size + offset_s];

            for (int ib = 0; ib < n_max_pts_bin; ib++) {

                if (ib >= n_max_pts_bin) {continue;}

                // TODO: CHECK THAT WE DON'T HAVE 
                // MORE POINTS AFTER A -1 PT...
                if (bin_b_ids[ib] <= -1) {break;}

                // if it's the same bin, 
                // only compare lower points to higher points

                // TODO: this is commented out only when bins are compared and skipped in the same way
                if (bin_b_ids[ib] == bin_a_ids[ia]) {continue;}

                if ((i_bin_a == i_bin_b) && (bin_b_ids[ib] < bin_a_ids[ia])) {
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

                // TODO: check that this should be <= and not > . . .
                // TODO: CHECK THAT THIS STATEMENT IS EVEN CORRECT 
                // OR THAT WE NEED IT...
                if ((scale_max / scale_min) > scale_radius) {continue;}

                float diffx = bx - ax;
                float diffy = by - ay;
                float diffz = bz - az;
                float dist = diffx * diffx + diffy * diffy + diffz * diffz;
                dist = sqrt(dist);

                // check the distance with the radius
                // but scale the radius by the average scale
                // TODO: NOT DOING THE AVERAGE SCALE THING ANYMORE . . . 
                // CHECK THIS
                float log_avg_scale = sqrt(as * bs);
                if (dist > radius / log_avg_scale) {continue;}
                // if (dist > radius) {continue;}

                // int i_buff = atomicAdd(&atomic_i_buff, edge_size);

                // /*
                // TODO: add the waiting if the buffer is full...
                // */
                // int max_atomic_i_buff = (n_edges_buff - 1) * edge_size;
                // while (i_buff >  max_atomic_i_buff) {
                //     // printf("i_buff = %d / %d\n", i_buff, n_edges_buff * edge_size);
                //     clock_t start_clock = clock64();
                //     clock_t offset_clock = 0;
                //     while (offset_clock < wait_count) {
                //         offset_clock = clock64() - start_clock;
                //     }
                //     if (atomic_i_buff <= max_atomic_i_buff) {
                //         int i_buff = atomicAdd(&atomic_i_buff, edge_size);
                //     }
                // }

                int id_a = bin_a_ids[ia];
                int id_b = bin_b_ids[ib];

                // we already check this above...
                // if (id_a == id_b) {continue;}

                // edge_buff[i_buff + 0] = id_a;
                // edge_buff[i_buff + 1] = id_b;
                int idx_edges = atomicAdd(&atomic_i_edges, edge_size);
                // atomic add returns the old value: so commented out
                // idx_edges -= edge_size;
                // if (atomic_i_edges < 1000) {
                //     // printf("atomic_i_edges = %d, idx_edges = %d\n", atomic_i_edges, idx_edges);

                //     if (idx_edges > max_atomic_i_edges) {
                //         printf("th %d bl %d ran out of space in edges: %d / %d\n", 
                //             threadIdx.x, blockIdx.x, idx_edges, max_atomic_i_edges);
                //         continue;
                //     }
                // }
                // int max_edges = edges.size(0) * edges.size(1) * edges.size(2);
                // if (i_edges_start + idx_edges > max_edges - 100) {
                //     printf("adding edges at location %d and %d\n and max val is %d\n", 
                //         i_edges_start + idx_edges + 0,
                //         i_edges_start + idx_edges + 1,
                //         max_edges);
                // }
                
                edges[i_edges_start + idx_edges + 0] = id_a;
                edges[i_edges_start + idx_edges + 1] = id_b;
                

                // atomicAdd(&atomic_buff_counter, 1);

                // if (atomic_buff_counter == n_edges_buff) {
                //     printf("emptying buffer . . .");
                //     // empty the buffer and put it in the edges array
                //     for (int ie = 0; ie < i_buff; ie ++) {
                //         edges[i_edges + ie] = edge_buff[ie];
                //     }
                //     i_edges += i_buff;
                //     atomic_i_buff = 0;
                //     atomic_buff_counter = 0;
                //     printf("emptied buffer . . .");
                // }
            }

        }

        // any thread that gets to this 
        // location (except thread 0) is done
        // for this bin_b
        atomicAdd(&threads_left_counter, -1);
        // continue;

        // if (threadIdx.x == 0) {
        //     while(threads_left_counter != 0) {
        //         // wait
        //         clock_t start_clock = clock64();
        //         clock_t offset_clock = 0;
        //         while (offset_clock < wait_count) {
        //             offset_clock = clock64() - start_clock;
        //         }
        //         // check i_buff condition in the case that some of the
        //         // threads are waiting for thread 0 to empty the edge buffer
        //         // because the edge buffer is full
        //         // if (i_buff - edge_size >=  n_edges_buff * edge_size)
        //         // EDIT: make the buffer that put the last edge in the edge 
        //         // buffer be the thread that empties the buffer
        //     }

        // //     // TODO: remove this once the max stop functionality
        // //     // is implemented
        // //     assert(atomic_i_buff == atomic_buff_counter * edge_size);

        //     for (int ie = 0; ie < atomic_i_buff; ie ++) {
        //         // printf()
        //         edges[i_edges + ie] = edge_buff[ie];
        //     }
        //     i_edges += atomic_i_buff;
        //     atomic_i_buff = 0;
        //     atomic_buff_counter = 0;
        // }
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
        at::Tensor edges
       ) { 
  const int threads = 32;
  //const int threads = 3;
  int n_bins = neighbor_bins.size(0);
  const dim3 blocks(n_bins);
  int n_max_neighbors = neighbor_bins.size(1);
  int n_max_edges_per_bin = edges.size(1);
  int max_edges = edges.size(0) * edges.size(1) * edges.size(2);
  
  //AT_DISPATCH_FLOATING_TYPES(edges.type(), "frnn_forward_cuda", ([&] {
    AT_DISPATCH_INTEGRAL_TYPES(edges.type(), "frnn_forward_cuda", ([&] {
    frnn_cuda_forward_kernel<<<blocks, threads>>>(
        neighbor_bins.data<int>(),
        pts.data<float>(),
        pt_idxs.data<int>(),
        first_pt_idxs.data<int>(),
        radius,
        scale_radius,
        n_max_neighbors,
        n_max_edges_per_bin,
        edges.data<int>(),
        max_edges);
    }));
  
  return {edges};
}
