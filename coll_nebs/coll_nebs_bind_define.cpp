#include <torch/extension.h>

#include <vector>
#include <iostream>
#include <string>

std::vector<torch::Tensor> coll_nebs_cuda_call(
    torch::Tensor tex,
    torch::Tensor pts,
    torch::Tensor imgid,

    float lin_radius,
    float scale_radius,

    int threshold,
    int coll_iters,
    float dampen_fact,

    int batch_size,
    int img_size,

    int device_id
    );

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> coll_nebs_cuda(
    torch::Tensor tex,
    torch::Tensor pts,
    torch::Tensor imgid,

    float lin_radius,
    float scale_radius,

    int threshold,
    int coll_iters,
    float dampen_fact,

    int batch_size,
    int img_size,

    int device_id
    )
{

    CHECK_INPUT(pts);
    CHECK_INPUT(imgid);

    return coll_nebs_cuda_call(
        tex,
        pts,
        imgid,

        lin_radius,
        scale_radius,

        threshold,
        coll_iters,
        dampen_fact,

        batch_size,
        img_size,

        device_id
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("coll_nebs_kernel", &coll_nebs_cuda, "FRNN (CUDA)");
}
