#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <string>

std::vector<torch::Tensor> pyramid_bin_cuda_call(
        torch::Tensor scabin,
        torch::Tensor locs,
        torch::Tensor locs_imgid,
        torch::Tensor patch_ids,
        torch::Tensor batch_size
);
    
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> pyramid_bin_cuda(
        torch::Tensor scabin,
        torch::Tensor locs,
        torch::Tensor locs_imgid,
        torch::Tensor patch_ids,
        torch::Tensor batch_size
)
{
    CHECK_INPUT(scabin);
    CHECK_INPUT(locs);
    CHECK_INPUT(locs_imgid);
    CHECK_INPUT(patch_ids);
    CHECK_INPUT(batch_size);

    return pyramid_bin_cuda_call(
             scabin,
             locs,
             locs_imgid,
             patch_ids,
             batch_size
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pyramid_bin_kernel", &pyramid_bin_cuda, "pyramid_bin (CUDA)");
}
