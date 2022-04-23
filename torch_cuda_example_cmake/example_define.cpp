#include <torch/script.h>

#include <vector>
#include <iostream>
#include <string>

// this is the entry point; will call into example_call defined in the example_driver.cu
std::vector<torch::Tensor> example_call(
    torch::Tensor input
);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> example_cuda(
    torch::Tensor input
){
    CHECK_INPUT(input);

    return example_call(
        input
    );
}


TORCH_LIBRARY(example_op, m) {
    m.def("example_kernel", example_cuda);
}
