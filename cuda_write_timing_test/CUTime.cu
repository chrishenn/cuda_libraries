#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "CUTime.h"


long get_nanos();

long get_nanos() {
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

// Macro to catch CUDA errors in CUDA runtime calls
void CUDA_SAFE_CALL( cudaError_t call ){
    do {
        cudaError_t err = call;
        if (cudaSuccess != err) {
            fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err) );
            exit(EXIT_FAILURE);
        }
    } while (0);
}

// Macro to catch CUDA errors in kernel launches
void CHECK_LAUNCH_ERROR(){
    do {
        /* Check synchronous errors, i.e. pre-launch */
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",
                     __FILE__, __LINE__, cudaGetErrorString(err) );
            exit(EXIT_FAILURE);
        }
        /* Check asynchronous errors, i.e. kernel failed (ULF) */
        err = cudaDeviceSynchronize();
        if (cudaSuccess != err) {
            fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",
                     __FILE__, __LINE__, cudaGetErrorString( err) );
            exit(EXIT_FAILURE);
        }
    } while (0);
}



__global__ void write_test( int* edges, const int size )
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid >= size){ return; }

    int threads = blockDim.x * gridDim.x;

    for (int i = thid; i < size; i += threads){
        edges[thid] = 23;
    }
}
// __global__ void write_test( int* edges, const int size )
// {
// }

void CUTime::run_time(const int size)
{
    const int THREADS_PER_BLOCK = 256;

    double start, stop, elapsed, mintime;
    start = 0;
    stop = 0;
    elapsed = 0;
    mintime = 0;

    const double INFTY = exp(1000.0);

    int* edges = 0;
    cudaMalloc ((void **)&edges, sizeof(edges[0]) * size);

    dim3 dimBlock (THREADS_PER_BLOCK);
    int threadBlocks = (size + (dimBlock.x - 1)) / dimBlock.x;
    // dim3 dimGrid (threadBlocks);
    dim3 dimGrid (256);

    cudaDeviceSynchronize();
    start = get_nanos();
    write_test<<<dimGrid, dimBlock>>> (edges, size);
    cudaDeviceSynchronize();
    stop = get_nanos();
    elapsed = stop - start;

    double mintime_ms = 1.0e-6 * elapsed;
    double mintime_s = 1.0e-9 * elapsed;
    printf ("write_test: data = %.3e bytes  time = %.3f msec\n", (double)sizeof (edges[0]) * size, mintime_ms);
    printf ("write_test: throughput = %.2f GB/sec\n", (1.0e-9 * sizeof(edges[0]) * size) / mintime_s);
    cudaFree (edges);
    cudaDeviceSynchronize();
}
