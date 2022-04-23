#include <torch/script.h>

#include <vector>
#include <iostream>
#include <string>

std::vector<torch::Tensor> segmenter_call(
        torch::Tensor edges,
        torch::Tensor imgid,
        torch::Tensor batch_size,

        torch::Tensor angles,
        torch::Tensor seg_thresh
    );

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> segmenter(
        torch::Tensor edges,
        torch::Tensor imgid,
        torch::Tensor batch_size,

        torch::Tensor angles,
        torch::Tensor seg_thresh
){
    CHECK_INPUT(edges);
    CHECK_INPUT(imgid);
    CHECK_INPUT(batch_size);

    CHECK_INPUT(angles);
    CHECK_INPUT(seg_thresh);

    return segmenter_call(
        edges,
        imgid,
        batch_size,

        angles,
        seg_thresh
    );
}

TORCH_LIBRARY(segmenter_op, m) {
    m.def("segmenter_kernel", segmenter);
}