#include <torch/extension.h>

#include <vector>
#include <iostream>
#include <string>

std::vector<torch::Tensor> frnn_cuda_call(
        torch::Tensor pts,
        torch::Tensor imgid,

        int batch_size,

        float lin_radius,
        float scale_radius
    );

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> frnn_cuda(
        torch::Tensor pts,
        torch::Tensor imgid,

        int batch_size,

        float lin_radius,
        float scale_radius
    )
{

    CHECK_INPUT(pts);
    CHECK_INPUT(imgid);

    return frnn_cuda_call(
             pts,
             imgid,

             batch_size,

             lin_radius,
             scale_radius
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("frnn_kernel", &frnn_cuda, "FRNN (CUDA)");
}
