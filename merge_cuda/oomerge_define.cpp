#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <string>

std::vector<torch::Tensor>oomerge_cuda_call(
    torch::Tensor send_map,
    torch::Tensor n_merge_each,

    int device
    );

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> oomerge_cuda(
  torch::Tensor send_map,
  torch::Tensor n_merge_each,

  int device
    )
{
    CHECK_INPUT(send_map);
    CHECK_INPUT(n_merge_each);

    return oomerge_cuda_call(
      send_map,
      n_merge_each,

      device
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("oomerge_kernel", &oomerge_cuda, "oomerge (CUDA)");
}
