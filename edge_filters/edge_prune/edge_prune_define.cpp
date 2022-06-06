#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <string>

std::vector<torch::Tensor> edge_prune_cuda_call(
    torch::Tensor edges,
    torch::Tensor drop_edges
);
    
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> edge_prune_cuda(
    torch::Tensor edges,
    torch::Tensor drop_edges
)
{
    CHECK_INPUT(edges);
    CHECK_INPUT(drop_edges);

    return edge_prune_cuda_call(
        edges,
        drop_edges
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("edge_prune_kernel", &edge_prune_cuda, "edge_prune (CUDA)");
}
