#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>

#include "frnn_bin_kern.h"


__global__ void frnn_neighbors(
    const   float*  __restrict__   radius_at_each_scabin,
    const   int*    __restrict__   scabin_offsets,
            int*    __restrict__   neighbors,
            int*                   neb_counts,

    const   int                  num_binstotal,
    const   int                  num_bins_perimg,
    const   int                  num_imgs,
	const   int                  num_scabin,
	const   int                  max_neighbors
    )
{
    const int threadid_grid =   blockIdx.x * blockDim.x + threadIdx.x;
    const int tot_threads =     gridDim.x * blockDim.x;


    for (int i = threadid_grid; i < num_binstotal; i += tot_threads)
    {
        if (i >= num_binstotal){ break; }


        int img_idx  =      floorf( i / num_imgs );
        int i_imlocal =     i - img_idx * num_bins_perimg;

        int scabin_idx = 0;
        int next_scabin_offset;
        int current_scabin_offset;
        for (int j = 0; j < num_scabin; j++)
        {
            current_scabin_offset = scabin_offsets[j];
            if (j+1 < num_scabin){
                next_scabin_offset =  scabin_offsets[j+1];
            } else {
                next_scabin_offset = current_scabin_offset;
            }

            if ((i_imlocal > current_scabin_offset) && (i_imlocal < next_scabin_offset)) {
                scabin_idx = j;
                break;
            }
        }


    	int x_idx = i_imlocal - current_scabin_offset;
    	int num_xbins = next_scabin_offset - current_scabin_offset;
        int neighbor_count = 0;

        // my bin goes in neighbors[i][0]
        neighbors[i * max_neighbors + neighbor_count] = i;
        neighbor_count++;

        // bin x below; my scale
        if (x_idx > 0) {
            neighbors[i*max_neighbors + neighbor_count] = i - 1;
            neighbor_count++;
        }
        // bin x above; my scale
        if (x_idx < num_xbins) {
            neighbors[i*max_neighbors + neighbor_count] = i + 1;
            neighbor_count++;
        }


        float bin_radius = radius_at_each_scabin[ scabin_idx ];
        float x_left  = x_idx * bin_radius;
        float x_right = x_left + bin_radius;

        // compute neighbors in the scale below
        if (scabin_idx > 0) {

            int offset_of_bin_below = scabin_offsets[ scabin_idx - 1 ];
            float lower_bin_radius = radius_at_each_scabin[ scabin_idx - 1];

            float neighbor_left_limit  = x_left - bin_radius;
            float neighbor_right_limit = x_right + bin_radius;

            int leftmost_bin_below = floorf(neighbor_left_limit / lower_bin_radius);
            int rightmost_bin_below = floorf(neighbor_right_limit / lower_bin_radius);

            int num_xbins_below =  current_scabin_offset - offset_of_bin_below;

            if (leftmost_bin_below < 0) { leftmost_bin_below = 0; }
            if (rightmost_bin_below > (num_xbins_below - 1))  { rightmost_bin_below = num_xbins_below - 1; }

            for (int k = leftmost_bin_below; k < rightmost_bin_below + 1; k++) {
                if (neighbor_count >= max_neighbors){ printf("FRNN_BIN ERROR: TOO MANY NEIGHBORS FOR ALLOCATED WIDTH\n"); }

                neighbors[i*max_neighbors + neighbor_count] = img_idx * num_bins_perimg + offset_of_bin_below + k;
                neighbor_count++;
            }
        }

        // compute neighbors in the scale above
        if (scabin_idx < num_scabin-1) {

            int offset_of_bin_above = scabin_offsets[ scabin_idx + 1 ];
            float upper_bin_radius = radius_at_each_scabin[ scabin_idx + 1 ];

            float neighbor_left_limit  = x_left - upper_bin_radius;
            float neighbor_right_limit = x_right + upper_bin_radius;

            int leftmost_bin_above = floorf(neighbor_left_limit / upper_bin_radius);
            int rightmost_bin_above = floorf(neighbor_right_limit / upper_bin_radius);

            int num_xbins_above = offset_of_bin_above - current_scabin_offset;

            if (leftmost_bin_above < 0) { leftmost_bin_above = 0; }
            if (rightmost_bin_above > (num_xbins_above - 1))  { rightmost_bin_above = num_xbins_above - 1; }

            for (int k = leftmost_bin_above; k < rightmost_bin_above + 1; k++) {
                if (neighbor_count >= max_neighbors){ printf("FRNN_BIN ERROR: TOO MANY NEIGHBORS FOR ALLOCATED WIDTH\n"); }

                neighbors[i*max_neighbors + neighbor_count] = img_idx * num_bins_perimg + offset_of_bin_above + k;
                neighbor_count++;
            }
        }

        neb_counts[threadid_grid] = neighbor_count;

        for (int end = neighbor_count; end < max_neighbors; end++){
            neighbors[ i * max_neighbors + end ] = -1;
        }
    }
}


__global__ void frnn_bin_counts(
    const   float*   __restrict__     pts,
    const   int*     __restrict__     img_ids,
    const   float*   __restrict__     radius_at_each_scabin,
    const   int*     __restrict__     scabin_offsets,
            int*     __restrict__     bin_counts,
            int*     __restrict__     bid2ptid,
            int*     __restrict__     write_count,

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
        if (i >= num_binstotal) { break; }
        bin_counts[i] = 0;
    }
    __syncthreads();

    for (int i = thid_grid; i < pts_size0; i += tot_threads)
    {
        if (i >= pts_size0){ break; }

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
        if (i >= pts_size0){ break; }

        int bid =  bid2ptid[ i*2 + 0 ];
        int ptid = bid2ptid[ i*2 + 1 ];

        int boffset = bin_offsets[ bid ];

        int bwrites = atomicAdd( &bin_wcounts[bid], 1 );

        bin_ptids[boffset + bwrites] = ptid;
    }
}



void frnn_bin_kern::frnn_neighbors_launch(
    float* radius_at_each_scabin,
    int* scabin_offsets,
    int* neighbors,
    int* neb_counts,

    int num_binstotal,
    int num_bins_perimg,
    int num_imgs,
    int num_scabin,
    int max_neighbors,

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
        max_neighbors
    );
}

void frnn_bin_kern::frnn_bin_counts_launch(
    float*   pts,
    int*     img_ids,
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
