#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <string>

std::vector<torch::Tensor> edge_min_cuda_call(
    torch::Tensor edges,
    torch::Tensor scores,
    torch::Tensor imgid,

    torch::Tensor max_binsize,

    float keep_frac
);
    
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> edge_min_cuda(
    torch::Tensor edges,
    torch::Tensor scores,
    torch::Tensor imgid,

    torch::Tensor max_binsize,

    float keep_frac
)
{
    CHECK_INPUT(edges);
    CHECK_INPUT(scores);
    CHECK_INPUT(imgid);
    CHECK_INPUT(max_binsize);

    return edge_min_cuda_call(
        edges,
        scores,
        imgid,

        max_binsize,

        keep_frac
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("edge_min_kernel", &edge_min_cuda, "edge_min (CUDA)");
}
