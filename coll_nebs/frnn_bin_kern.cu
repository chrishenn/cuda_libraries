#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>

#include "frnn_bin_kern.h"

// each row of neighbors should contain the 'central' bin at column0, and other bin's bin_ids
// for comparison to the central bin along the same row.
__global__ void frnn_neighbors(
    const   float*  __restrict__   radius_at_each_scabin,
    const   int*    __restrict__   scabin_offsets,
            int*    __restrict__   neighbors,
            int*    __restrict__   neb_counts,

    const   int                    num_binstotal,
    const   int                    num_bins_perimg,
    const   int                    num_imgs,
	const   int                    num_scabin,
	const   int                    neighb_size1
    )
{
    const int thid_glob =       blockIdx.x * blockDim.x + threadIdx.x;
    const int tot_threads =     gridDim.x * blockDim.x;

    for (int i = thid_glob; i < num_binstotal; i += tot_threads)
    {
        int img_idx  =          floorf( i / num_bins_perimg );
        int img_bin_idx =       i - img_idx * num_bins_perimg;

        int scabin_idx = 0;
        int next_scabin_offset = 0;
        int current_scabin_offset = 0;
        for (int j = 0; j < num_scabin; j++)
        {
            current_scabin_offset = scabin_offsets[j];
            if (j+1 <= num_scabin){
                next_scabin_offset = scabin_offsets[j+1];
            } else {
                next_scabin_offset = current_scabin_offset;
            }

            if ( img_bin_idx > current_scabin_offset && img_bin_idx < next_scabin_offset ) {
                scabin_idx = j;
                break;
            }
        }

        // linear-space binning takes place along x-axis only
    	int x_idx = img_bin_idx - current_scabin_offset;
    	int num_xbins = next_scabin_offset - current_scabin_offset;
        int cols_written = 0;

        // I am the 'center' bin at 'i'; my bin goes in neighbors[i][0]
        neighbors[i * neighb_size1 + cols_written] = i;
        cols_written++;

        // bin x below; my scale
        if (x_idx > 0) {
            neighbors[i*neighb_size1 + cols_written] = i - 1;
            cols_written++;
        }
        // bin x above; my scale
        if (x_idx < num_xbins-1) {
            neighbors[i*neighb_size1 + cols_written] = i + 1;
            cols_written++;
        }


        float bin_radius = radius_at_each_scabin[ scabin_idx ];
        float x_left  = x_idx * bin_radius;
        float x_right = x_left + bin_radius;

        // find neighbors in the scale below
        if (scabin_idx > 0) {

            int offset_of_bin_below =   scabin_offsets[ scabin_idx - 1 ];
            float lower_bin_radius =    radius_at_each_scabin[ scabin_idx - 1];

            float neighbor_left_limit  = x_left - bin_radius;
            float neighbor_right_limit = x_right + bin_radius;

            int leftmost_bin_below =    floorf(neighbor_left_limit / lower_bin_radius);
            int rightmost_bin_below =   floorf(neighbor_right_limit / lower_bin_radius);

            int num_xbins_below =  current_scabin_offset - offset_of_bin_below;

            if (leftmost_bin_below < 0) { leftmost_bin_below = 0; }
            if (rightmost_bin_below > (num_xbins_below - 1))  { rightmost_bin_below = num_xbins_below - 1; }

            for (int k = leftmost_bin_below; k <= rightmost_bin_below; k++) {
                assert(cols_written < neighb_size1);

                neighbors[i*neighb_size1 + cols_written] = img_idx * num_bins_perimg + offset_of_bin_below + k;
                cols_written++;
            }
        }

        // find neighbors in the scale above
        if (scabin_idx < num_scabin-1) {

            int offset_of_bin_above =   scabin_offsets[ scabin_idx + 1 ];
            float upper_bin_radius =    radius_at_each_scabin[ scabin_idx + 1 ];

            float neighbor_left_limit  = x_left - upper_bin_radius;
            float neighbor_right_limit = x_right + upper_bin_radius;

            int leftmost_bin_above =    floorf(neighbor_left_limit / upper_bin_radius);
            int rightmost_bin_above =   floorf(neighbor_right_limit / upper_bin_radius);

            int num_xbins_above = offset_of_bin_above - current_scabin_offset;

            if (leftmost_bin_above < 0) { leftmost_bin_above = 0; }
            if (rightmost_bin_above > (num_xbins_above - 1))  { rightmost_bin_above = num_xbins_above - 1; }

            for (int k = leftmost_bin_above; k <= rightmost_bin_above; k++) {
                assert(cols_written < neighb_size1);

                neighbors[i*neighb_size1 + cols_written] = img_idx * num_bins_perimg + offset_of_bin_above + k;
                cols_written++;
            }
        }

        neb_counts[i] = cols_written;

        // fill remaining in this thread's row with -1
        for (int end = cols_written; end < neighb_size1; end++){
            neighbors[ i * neighb_size1 + end ] = -1;
        }
    }
}

// prune redundant bin_id pairs from rows of neighbors. Update neb_counts for correct block launches in frnn_bipart_kern.
__global__ void frnn_neighbors_prune(
    int* __restrict__ neighbors,
    int* __restrict__ neb_counts,

    const int num_binstotal,
    const int neighb_size1
){
    for (int phase = 0; phase < 2; phase++) {

        for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < num_binstotal; row += blockDim.x * gridDim.x) {
            for (int col = 1; col < neighb_size1; col++) {

                int target_row = neighbors[row * neighb_size1 + col];

                if (target_row >= 0 && (target_row+phase)%2 == 0)
                {
                    for (int target_col = 1; target_col < neighb_size1; target_col++) {
                        int old = atomicCAS( &neighbors[target_row*neighb_size1 + target_col], row, -1);

                        if (old == row) { atomicSub( &neb_counts[target_row], 1); }
                    }
                }

            }
        }

    __syncthreads();
    }
}

// shuffle valid bin_ids to lowest open column in neighbors
// each thread reads a row of void bin_ids from neighbors into a local buffer; writes that row;
//   fills remaining in the row with -1.
__global__ void frnn_neighbors_shuffle(
    int* __restrict__ neighbors,

    const int num_binstotal,
    const int neighb_size1
){
    extern __shared__ int buffer[];

    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < num_binstotal; row += blockDim.x * gridDim.x) {

        int buffered = 0;

        for (int col = 1; col < neighb_size1; col++) {

            int curr = neighbors[row * neighb_size1 + col];
            if (curr >= 0)
            {
                buffer[threadIdx.x*neighb_size1 + buffered] = curr;
                buffered++;
            }
        }

        int col;
        for (col = 0; col <= buffered; col++) {

            neighbors[row * neighb_size1 + col + 1] = buffer[threadIdx.x*neighb_size1 + col];
        }
        for (int end = col; end < neighb_size1; end++) {
            
            neighbors[row * neighb_size1 + end] = -1;
        }
    }
}

// count the number of points in each bin. Grid initializes bin_counts to 0 before counting.  
// write_count gives a global counter for threads to write to bid2ptid
__global__ void frnn_bin_counts(
    const   float* __restrict__  pts,
    const   long*   __restrict__  img_ids,
    const   float* __restrict__  radius_at_each_scabin,
    const   int*   __restrict__  scabin_offsets,
            int*   __restrict__  bin_counts,
            int*   __restrict__  bid2ptid,
            int*   __restrict__  write_count,

    const   float                scale_radius,
    const   float                xmin,
    const   float                scalemin,
    const   int                  pts_size0,
    const   int                  pts_size1,
    const   int                  num_bins_perimg,
    const   int                  num_binstotal
)
{
    int thid_grid = blockIdx.x * blockDim.x + threadIdx.x;
    int tot_threads = gridDim.x * blockDim.x;

    for (int i = thid_grid; i < num_binstotal; i += tot_threads)
    {
        bin_counts[i] = 0;
    }
    __syncthreads();

    for (int i = thid_grid; i < pts_size0; i += tot_threads)
    {
        float x =     pts[i * pts_size1 + 1];
        float scale = pts[i * pts_size1 + 4];

        int scabin_id =         floorf( (log(scale)-scalemin) / scale_radius);
        float radius_scabin =   radius_at_each_scabin[scabin_id];
        int xbin_id =           floorf( (x - xmin) / radius_scabin );

        int img_id = img_ids[i];
        int bin_id = img_id * num_bins_perimg + scabin_offsets[scabin_id] + xbin_id;

        atomicAdd( &bin_counts[bin_id], 1 );

        int write_i = atomicAdd( &write_count[0], 2 );
        bid2ptid[write_i + 0] = bin_id;
        bid2ptid[write_i + 1] = i;
    }
}

// build frnn_ptids. bin_wcounts give global counters to write to a given bin's space in bin_ptids. 
__global__ void frnn_ptids(
    const   int*   __restrict__   bin_counts,
    const   int*   __restrict__   bin_offsets,
    const   int*   __restrict__   bid2ptid,
            int*   __restrict__   bin_ptids,
            int*   __restrict__   bin_wcounts,

    const   int                   pts_size0
    )
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < pts_size0; i += blockDim.x * gridDim.x)
    {
        int bid =  bid2ptid[ i*2 + 0 ];
        int ptid = bid2ptid[ i*2 + 1 ];

        int boffset = bin_offsets[ bid ];

        int bwrites = atomicAdd( &bin_wcounts[bid], 1 );

        bin_ptids[boffset + bwrites] = ptid;
    }
}

// cpu-bound kernel launchers
void frnn_bin_kern::frnn_neighbors_launch(
    float* radius_at_each_scabin,
    int* scabin_offsets,
    int* neighbors,
    int* neb_counts,

    int num_binstotal,
    int num_bins_perimg,
    int num_imgs,
    int num_scabin,
    int neighb_size1,

    dim3 blocks,
    dim3 threads
){
    frnn_neighbors<<<blocks, threads>>>(
        radius_at_each_scabin,
        scabin_offsets,
        neighbors,
        neb_counts,

        num_binstotal,
        num_bins_perimg,
        num_imgs,
        num_scabin,
        neighb_size1
    );
}

void frnn_bin_kern::frnn_neighbors_prune_launch(
    int* neighbors,
    int* neb_counts,

    int num_binstotal,
    int neighb_size1,

    dim3 blocks,
    dim3 threads
){
    frnn_neighbors_prune<<<blocks, threads>>>(
        neighbors,
        neb_counts,

        num_binstotal,
        neighb_size1
    );

    int loc_threads = 64;
    size_t shared = loc_threads * neighb_size1 * sizeof(int);

    frnn_neighbors_shuffle<<<blocks, loc_threads, shared>>>(
        neighbors,

        num_binstotal,
        neighb_size1
    );

}

void frnn_bin_kern::frnn_bin_counts_launch(
    float*   pts,
    long*     img_ids,
    float*   radius_at_each_scabin,
    int*     scabin_offsets,
    int*     bin_counts,
    int*     bid2ptid,
    int*     write_count,

    float   scale_radius,
    float   xmin,
    float   scalemin,
    int     pts_size0,
    int     pts_size1,
    int     num_bins_perimg,
    int     num_binstotal,

    dim3 blocks,
    dim3 threads
){
    frnn_bin_counts<<<blocks, threads>>>(
        pts,
        img_ids,
        radius_at_each_scabin,
        scabin_offsets,
        bin_counts,
        bid2ptid,
        write_count,

        scale_radius,
        xmin,
        scalemin,
        pts_size0,
        pts_size1,
        num_bins_perimg,
        num_binstotal
    );
}

void frnn_bin_kern::frnn_ptids_launch(
    int*      bin_counts,
    int*      bin_offsets,
    int*      bid2ptid,
    int*      bin_ptids,
    int*      bin_wcounts,

    int       pts_size0,

    dim3 blocks,
    dim3 threads
){
    frnn_ptids<<<blocks, threads>>>(
        bin_counts,
        bin_offsets,
        bid2ptid,
        bin_ptids,
        bin_wcounts,

        pts_size0
    );
}
