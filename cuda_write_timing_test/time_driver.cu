#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>

#include "CUTime.h"


long get_nanos1() {
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

std::vector<torch::Tensor> time_cuda_call(int size)
{
    double in = get_nanos1();

    CUTime* TIME = new CUTime();
    TIME->run_time(size);

    auto options = torch::TensorOptions()
      .dtype(torch::kInt32)
      .layout(torch::kStrided)
      .device(torch::kCUDA, 0);

    torch::Tensor out = torch::zeros({1, 1}, options);

    cudaDeviceSynchronize();
    double out1 = get_nanos1()-in;
    printf("CPU time ext: %.3f msec\n", out1*1e-6);

    return {out};
}
