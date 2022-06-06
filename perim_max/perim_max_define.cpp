#include <torch/script.h>

#include <vector>
#include <iostream>
#include <string>

// this is the entry point; will call into example_call defined in the example_driver.cu
std::vector<torch::Tensor> perim_max_call(
    torch::Tensor imgid,
    torch::Tensor values,
    torch::Tensor batch_size
);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> perim_max_cuda(
    torch::Tensor imgid,
    torch::Tensor values,
    torch::Tensor batch_size
){
    CHECK_INPUT(imgid);
    CHECK_INPUT(values);
    CHECK_INPUT(batch_size);

    return perim_max_call(
        imgid,
        values,
        batch_size
    );
}


TORCH_LIBRARY(perim_max_op, m) {
    m.def("perim_max_kernel", perim_max_cuda);
}
