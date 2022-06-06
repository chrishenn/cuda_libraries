#include <torch/extension.h>

#include "frnn_bin_kern.h"

#include <vector>
#include <iostream>
#include <string>

std::vector<torch::Tensor> frnn_cuda_call(
    torch::Tensor pts,
    torch::Tensor img_ids,
    torch::Tensor radius_at_each_scabin,
    torch::Tensor scabin_offsets,

    int num_imgs,
    int num_bins_perimg,
    int device_id,
    float radius,
    float scale_radius,
    float xmin,
    float scalemin
    );


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> frnn_cuda(
    torch::Tensor pts,
    torch::Tensor img_ids,
    torch::Tensor radius_at_each_scabin,
    torch::Tensor scabin_offsets,

    int num_imgs,
    int num_bins_perimg,
    int device_id,
    float radius,
    float scale_radius,
    float xmin,
    float scalemin
    )
{

    CHECK_INPUT(pts);
    CHECK_INPUT(img_ids);
    CHECK_INPUT(radius_at_each_scabin);
    CHECK_INPUT(scabin_offsets);

    return frnn_cuda_call(
        pts,
        img_ids,
        radius_at_each_scabin,
        scabin_offsets,

        num_imgs,
        num_bins_perimg,        
        device_id,
        radius,
        scale_radius,
        xmin,
        scalemin
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("frnn_kernel", &frnn_cuda, "FRNN (CUDA)");
}
