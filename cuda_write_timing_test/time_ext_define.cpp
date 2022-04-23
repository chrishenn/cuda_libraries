#include <torch/extension.h>

#include <vector>
#include <iostream>
#include <string>

std::vector<torch::Tensor> time_cuda_call(int size);

std::vector<torch::Tensor> time_cuda(int size)
{
    return time_cuda_call(size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("time_ext", &time_cuda, "FRNN (CUDA)");
}
