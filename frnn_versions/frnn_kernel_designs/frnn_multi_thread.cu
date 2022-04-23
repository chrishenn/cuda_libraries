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
const int max_n_edges_buff = 128;
const int edge_size = 2;
const int offset_x = 0;
const int offset_y = 1;
const int offset_z = 2;
const int offset_s = 3;
const int wait_count = 1e7;



__device__ __forceinline__ void wait_forward() {
    clock_t start_clock = clock64();
    clock_t offset_clock = 0;
    while (offset_clock < wait_count) {
        offset_clock = clock64() - start_clock;
    }
}


__global__ void frnn_cuda_forward_kernel(
    const int* neighbor_bins,
    const float* pts,
    const int* pt_idxs,
    const int* first_pt_idxs,
    const float radius,
    const float scale_radius,
    const int n_max_neighbors,
    // const int n_max_edges_per_bin,
    int* edges,
    // int *i_edges_ptr,
    const int max_size_edges
   ) {
    
    // wait_forward();
    // printf("here kernel \n");
    // return;

    // if the bin is empty:
    if (first_pt_idxs[blockIdx.x] == -1) {
        return;
    }
    if (blockIdx.x != 0) {
        return;
    }

    // return;

    // aren't using thread offset anymore...
    // int thread_offset = 2;

    // the index for the next open idx in the buffer
    __shared__ int atomic_i_buff;
    // counts how many elements (NOT) edges are in the buffer
    __shared__ int atomic_buff_size;
    // keeps track of how many edges still need to
    // compare points
    __shared__ int threads_left_counter;
    // stores the points for bin_a
    __shared__ float bin_a[n_max_pts_bin * pt_size];
    __shared__ float bin_b[n_max_pts_bin * pt_size];
    // stores the pt ids for bin_a
    __shared__ int bin_a_ids[n_max_pts_bin];
    __shared__ int bin_b_ids[n_max_pts_bin];
    __shared__ int edge_buff[max_n_edges_buff * edge_size];

    int i_bin_a = blockIdx.x;
    int max_atomic_i_buff = (max_n_edges_buff - 1) * edge_size;
    // return;
    //////////////
    // load bin_a
    //////////////
    if (threadIdx.x == 0) {

        atomic_i_buff = 0;
        atomic_buff_size = 0;
        threads_left_counter = 0;

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
    }
    // printf("here kernel 1\n");
    // return;
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
        // TODO: CHECK IF THIS IS TRUE ^ 
        // IT MIGHT BE THE CASE THAT THERE WON'T BE ANY
        // MORE BINS AFTER A -1 BIN...
        if (i_bin_b == -1) {continue;}

        // don't double check bin pairs
        if (i_bin_b < i_bin_a) {continue;}

        // if the bin is empty:
        if (first_pt_idxs[i_bin_b] == -1) {continue;}

        ////////////////////////
        // thread 0 loads bin_b
        ////////////////////////
        __syncthreads();
        if (threadIdx.x == 0) {

            // while (threads_left_counter != 0) {
            //     printf("waiting....\n");
            //     wait_forward();
            // }
            /*----------
            LOAD BIN B
            -----------*/
            int inext = first_pt_idxs[i_bin_b];

            for (int i=0; i < n_max_pts_bin; i++) {
                if (blockIdx.x == 0) {
                    printf("load b .. bl %d th %d inext %d \n",
                        blockIdx.x, threadIdx.x, inext);
                }
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
            
            // this allows the other threads to continue
            atomicAdd(&threads_left_counter, blockDim.x);
        }

        __syncthreads();


        /*---------------------
        THE COMPARISIONS
        now do the comparison between 
        bin_a's pts and bin_b's pts
        ----------------------*/

        // ia is the bin index for the current pt a
        // so it is NOT the index into the pts matrix
        // int ia;
        // for (int ja = 0; ja < n_max_pts_bin; ja++) {
        int ia_stop = n_max_pts_bin + n_max_pts_bin % blockDim.x;
        for (int ia = threadIdx.x; ia < ia_stop; ia += blockDim.x) {

            // ia = ja * blockDim.x + threadIdx.x;

            if (ia >= n_max_pts_bin) {continue;}

            // TODO: CHECK THAT WE DON'T HAVE 
            // MORE POINTS AFTER A -1 PT...
            if (bin_a_ids[ia] <= -1) {break;}

            float ax = bin_a[ia * pt_size + offset_x];
            float ay = bin_a[ia * pt_size + offset_y];
            float az = bin_a[ia * pt_size + offset_z];
            float as = bin_a[ia * pt_size + offset_s];

            for (int ib = 0; ib < n_max_pts_bin; ib++) {
                if (bin_b_ids[ib] == 4 && bin_a_ids[ia] == 0 ||
                    bin_b_ids[ib] == 0 && bin_a_ids[ia] == 4) {
                    printf("here 3.9 bl %d th %d i_bin_a %d i_bin_b %d bin_a_ids[ia] %d bin_b_ids[ib] %d \n",
                    blockIdx.x, threadIdx.x, i_bin_a, i_bin_b, bin_a_ids[ia],bin_b_ids[ib]);
                }

                if (ib >= n_max_pts_bin) {continue;}

                // TODO: CHECK THAT WE DON'T HAVE 
                // MORE POINTS AFTER A -1 PT...
                if (bin_b_ids[ib] <= -1) {break;}

                if (bin_b_ids[ib] == 4 && bin_a_ids[ia] == 0 ||
                    bin_b_ids[ib] == 0 && bin_a_ids[ia] == 4) {
                    printf("here 4.0 bl %d th %d i_bin_a %d i_bin_b %d bin_a_ids[ia] %d bin_b_ids[ib] %d \n",
                    blockIdx.x, threadIdx.x, i_bin_a, i_bin_b, bin_a_ids[ia],bin_b_ids[ib]);
                }

                // don't compare the same point to itself:
                if (bin_b_ids[ib] == bin_a_ids[ia]) {continue;}

                if (bin_b_ids[ib] == 4 && bin_a_ids[ia] == 0 ||
                    bin_b_ids[ib] == 0 && bin_a_ids[ia] == 4) {
                    printf("here 4.1 bl %d th %d i_bin_a %d i_bin_b %d bin_a_ids[ia] %d bin_b_ids[ib] %d \n",
                    blockIdx.x, threadIdx.x, i_bin_a, i_bin_b, bin_a_ids[ia],bin_b_ids[ib]);
                }
                // if it's the same bin, 
                // only compare lower points to higher points
                if ((i_bin_a == i_bin_b) && (bin_b_ids[ib] <= bin_a_ids[ia])) {
                    if (bin_b_ids[ib] == 4 && bin_a_ids[ia] == 0 ||
                        bin_b_ids[ib] == 0 && bin_a_ids[ia] == 4) {
                        printf("here 4.1 bl %d th %d i_bin_a %d i_bin_b %d bin_a_ids[ia] %d bin_b_ids[ib] %d \n",
                        blockIdx.x, threadIdx.x, i_bin_a, i_bin_b, bin_a_ids[ia],bin_b_ids[ib]);
                    }
                    continue;
                }
                if (bin_b_ids[ib] == 4 && bin_a_ids[ia] == 0 ||
                    bin_b_ids[ib] == 0 && bin_a_ids[ia] == 4) {
                    printf("here 4.2\n");
                }
                // printf("bl %d th %d (id a, id b) (%d, %d) \n", 
                //     blockIdx.x, threadIdx.x, bin_a_ids[ia], bin_b_ids[ib]);

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
                if (bin_b_ids[ib] == 4 && bin_a_ids[ia] == 0 ||
                    bin_b_ids[ib] == 0 && bin_a_ids[ia] == 4) {
                    printf("here 4.3\n");
                }

                if ((scale_max / scale_min) > scale_radius) {continue;}

                if (bin_b_ids[ib] == 4 && bin_a_ids[ia] == 0 ||
                    bin_b_ids[ib] == 0 && bin_a_ids[ia] == 4) {
                    printf("here 4.4\n");
                }
                float diffx = bx - ax;
                float diffy = by - ay;
                float diffz = bz - az;
                float dist = diffx * diffx + diffy * diffy + diffz * diffz;
                dist = sqrt(dist);

                float log_avg_scale = sqrt(as * bs);
                // printf("bl %d th %d id a %d id b %d\n", 
                //     blockIdx.x, threadIdx.x, bin_a_ids[ia], bin_b_ids[ib]);
                if (dist > radius / log_avg_scale) {continue;}
                if (bin_b_ids[ib] == 4) {
                    printf("here 4.5\n");
                }

                printf("- - - - - - - here kernel 000\n");
                // now we can add the edge to the edge buffer

                // grab an i_buff
                int i_buff = atomicAdd(&atomic_i_buff, edge_size);
                
                /* if i_buff is too large, 
                  wait until you get an i_buff that is small 
                  enough to fit in the edges buffer
                */
                while (i_buff > max_atomic_i_buff) {
                    wait_forward();
                    if (atomic_i_buff <= max_atomic_i_buff) {
                        int i_buff = atomicAdd(&atomic_i_buff, edge_size);
                    }
                }

                printf("- - - - - - - here kernel 111\n");
                edge_buff[i_buff + 0] = bin_a_ids[ia];
                edge_buff[i_buff + 1] = bin_b_ids[ib];

                int this_e_buffer_size = atomicAdd(&atomic_buff_size, edge_size) + edge_size;

                /*
                empty the buffer in the loop:
                */
                if (this_e_buffer_size == max_n_edges_buff * edge_size) {
                    
                    // empty the buffer and put it in the edges array
                    int this_i_edges = atomicAdd(&edges[0], this_e_buffer_size);

                    // make sure we don't overfill the edge output tensor
                    int e_stop = this_e_buffer_size;
                    if (this_i_edges + this_e_buffer_size > max_size_edges) {
                        e_stop = max_size_edges - this_i_edges;
                    }

                    for (int ie = 0; ie < e_stop; ie ++) {
                        edges[this_i_edges + ie] = edge_buff[ie];
                    }

                    atomic_i_buff = 0;
                    atomic_buff_size = 0;
                }
            }
        }

        /* 
        any thread that gets to this 
        location is done
        for this bin_b
        */
        atomicAdd(&threads_left_counter, -1);
        __syncthreads();
    }

    // TODO: MAKE SURE THREADS DON'T GET STUCK HERE
    // WHILE OTHER ARE AT THE OTHER SYNCTHREAD LOCATION WHEN
    // LOADING BIN B
    __syncthreads();
    /*
    empty the buffer one more time at the very end:
    */
    if (threadIdx.x == 0) {

        // assert(atomic_i_buff == atomic_buff_size);
        if (atomic_buff_size > 0) {

            int this_e_buffer_size = atomic_buff_size;
            int this_i_edges = atomicAdd(&edges[0], this_e_buffer_size);

            // make sure we don't overfill the edge output tensor
            int e_stop = this_e_buffer_size;
            if (this_i_edges + this_e_buffer_size > max_size_edges) {
                e_stop = max_size_edges - this_i_edges;
            }

            for (int ie = 0; ie < e_stop; ie ++) {
                edges[this_i_edges + ie] = edge_buff[ie];
            }
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
        at::Tensor edges
       ) { 
    // const int n_threads = 32;
    const int n_threads = 2;
    const int n_bins = neighbor_bins.size(0);
    const dim3 blocks(n_bins);
    const int n_max_neighbors = neighbor_bins.size(1);

    // int i_edges = 2;
    edges[0] = 2;
    // printf("&i_edges = %p\n", (void*) &i_edges);
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
        // n_max_edges_per_bin,
        edges.data<int>(),
        // &i_edges,
        max_size_edges);
    }));
  
    return {edges};
}
