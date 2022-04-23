#include <torch/script.h>

#include <vector>
#include <iostream>
#include <string>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> write_row_call(
        torch::Tensor rel_vec,
        torch::Tensor write_cols,

        torch::Tensor tex,
        torch::Tensor tex_rt,

        torch::Tensor vecs,
        torch::Tensor rowids
    );

std::vector<torch::Tensor> write_row(
        torch::Tensor rel_vec,
        torch::Tensor write_cols,

        torch::Tensor tex,
        torch::Tensor tex_rt,

        torch::Tensor vecs,
        torch::Tensor rowids
    )
{

    CHECK_INPUT(rel_vec);
    CHECK_INPUT(write_cols);
    CHECK_INPUT(tex);
    CHECK_INPUT(tex_rt);
    CHECK_INPUT(vecs);
    CHECK_INPUT(rowids);

    return write_row_call(
        rel_vec,
        write_cols,

        tex,
        tex_rt,

        vecs,
        rowids
    );
}

TORCH_LIBRARY(row_op, m) {
    m.def("write_row_bind", write_row);
}