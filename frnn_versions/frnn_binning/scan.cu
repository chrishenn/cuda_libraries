#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <iostream>

#include "scan.h"


#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

__global__ void prescan(int* g_idata, int* g_odata, int* glob_temp, bool redu_flag, int chunk_size, int tot_size)
{
    int thid = threadIdx.x;
    int thid_grid = blockIdx.x * blockDim.x + thid;
    int glob_offset = chunk_size * blockIdx.x;
    bool active = thid <= (tot_size - blockIdx.x * blockDim.x - 1)/2 + 1;

    extern __shared__ int temp[];
    int offset = 1;

    int ai;
    int bi;
    int bankOffsetA;
    int bankOffsetB;

    ai = thid;
    bi = thid + chunk_size/2;
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    if (active) {
        temp[ai + bankOffsetA] = g_idata[ai + glob_offset];
        temp[bi + bankOffsetB] = g_idata[bi + glob_offset];
    }

    int lastval_in;
    if ( active && redu_flag && (thid == blockDim.x-1 || thid_grid == tot_size-1) ){
        lastval_in = g_idata[bi + glob_offset];
    }

    for (int d = chunk_size>>1; d > 0; d >>= 1)
    {
        __syncthreads();
       if (thid < d && active)
       {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[bi] += temp[ai];
       }
       offset *= 2;
    }

    if (thid==0) { temp[chunk_size - 1 + CONFLICT_FREE_OFFSET(chunk_size - 1)] = 0; }

    for (int d = 1; d < chunk_size; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d && active)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    if (active){
        g_odata[ai + glob_offset] = temp[ai + bankOffsetA];
        g_odata[bi + glob_offset] = temp[bi + bankOffsetB];

        if ( redu_flag && (thid == blockDim.x-1 || thid_grid == tot_size-1) ){
            glob_temp[blockIdx.x] = temp[bi + bankOffsetB] + lastval_in;
        }
    }
}


__global__ void broadcast_add(int* g_data, int* glob_temp, int tot_size)
{
    __shared__ int s_incr[1];

    if (blockIdx.x < 1){ return; }

    int thid = threadIdx.x;
    int thid_grid = blockIdx.x * blockDim.x + thid;

    if (thid == 0 ){
        *s_incr = glob_temp[ blockIdx.x ];
    }
    __syncthreads();

    int incr = *s_incr;

    if (thid_grid < tot_size/2){
        g_data[thid_grid*2 + 0] += incr;
        g_data[thid_grid*2 + 1] += incr;
    }
}

unsigned long __host__ power_next(unsigned long v)
{
    unsigned long temp = v--;
    temp |= temp >> 1;
    temp |= temp >> 2;
    temp |= temp >> 4;
    temp |= temp >> 8;
    temp |= temp >> 16;
    temp++;
    return temp;
}



// TODO: only supports 2048*2048 = 4.1M elements
void scan::scan_launch(int* data_in, int* data_out, int tot_size)
{
    const unsigned long n_chunks = floorf( (tot_size-1)/2048 ) + 1;

    if (n_chunks > 1){

        unsigned long temp_size = power_next( n_chunks );

        int* glob_temp;
        cudaMalloc((void **)&glob_temp, temp_size * sizeof(int) );
        cudaMemset(glob_temp + n_chunks, 0, (temp_size-n_chunks)*sizeof(int) );

        const dim3 blocks(n_chunks);
        const dim3 threads(1024);

        const int chunk_size = 2048;
        const size_t shared = chunk_size * sizeof(int);
        prescan<<<blocks, threads, shared>>>(data_in, data_out, glob_temp, true, chunk_size, tot_size);

        const dim3 blocks_temp(1);
        const dim3 threads_temp(temp_size/2);
        const size_t shared_temp = temp_size * sizeof(int);
        prescan<<<blocks_temp, threads_temp, shared_temp>>>(glob_temp, glob_temp, glob_temp, false, temp_size, temp_size);

        this->broadcast_add_launch(data_out, glob_temp, tot_size, blocks, threads);

        cudaFree(glob_temp);
    }
    else {
        unsigned long temp_size = power_next( tot_size );

        const dim3 blocks(1);
        const dim3 threads(temp_size/2);
        size_t shared = temp_size * sizeof(int);

        prescan<<<blocks, threads, shared>>>(data_in, data_out, data_in, false, temp_size, temp_size);
    }
}

void scan::broadcast_add_launch(int* data_out, int* glob_temp, int tot_size, dim3 blocks, dim3 threads)
{
    broadcast_add<<<blocks, threads>>>(data_out, glob_temp, tot_size);
}
