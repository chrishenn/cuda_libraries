#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <string>

std::vector<torch::Tensor> grid_bin_cuda_call(
    torch::Tensor counts,
    torch::Tensor offsets,
    torch::Tensor binid,
    torch::Tensor scores,
    torch::Tensor o_ids,    
    
    float keep_frac,
    int n_bins,
    
    int batch_size,
    int locality_size,
    
    int device   
    );
    
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> grid_bin_cuda(
    torch::Tensor counts,
    torch::Tensor offsets,
    torch::Tensor binid,
    torch::Tensor scores,
    torch::Tensor o_ids,
    
    float keep_frac,
    int n_bins,
    
    int batch_size,
    int locality_size,
        
    int device  
    )
{
    CHECK_INPUT(counts);
    CHECK_INPUT(offsets);
    CHECK_INPUT(binid);
    CHECK_INPUT(scores);
    CHECK_INPUT(o_ids);

    return grid_bin_cuda_call(
        counts,
        offsets,
        binid,
        scores,
        o_ids,
        
        keep_frac,
        n_bins,

        batch_size,
        locality_size,
            
        device  
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grid_bin_kernel", &grid_bin_cuda, "grid_bin (CUDA)");
}
