#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <string>

std::vector<torch::Tensor> edge_bin_cuda_call(
    torch::Tensor edges,
    torch::Tensor scores,
    torch::Tensor imgid,

    float keep_frac,
    float threshold
);
    
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> edge_bin_cuda(
    torch::Tensor edges,
    torch::Tensor scores,
    torch::Tensor imgid,

    float keep_frac,
    float threshold
)
{
    CHECK_INPUT(edges);
    CHECK_INPUT(scores);
    CHECK_INPUT(imgid);

    return edge_bin_cuda_call(
        edges,
        scores,
        imgid,

        keep_frac,
        threshold
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("edge_bin_kernel", &edge_bin_cuda, "edge_bin (CUDA)");
}
