#include <ATen/ATen.h>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <math.h>
#include <stdio.h>





__global__ void frnn_cuda_forward_kernel(
    const torch::PackedTensorAccessor<int,      2,torch::RestrictPtrTraits,size_t> neighbor_bins,

    const torch::PackedTensorAccessor<float,    2,torch::RestrictPtrTraits,size_t> pts,

    const torch::PackedTensorAccessor<int,      1,torch::RestrictPtrTraits,size_t> pt_idxs,

    const torch::PackedTensorAccessor<int,      1,torch::RestrictPtrTraits,size_t> first_pt_idxs,

    const torch::PackedTensorAccessor<float,    0,torch::RestrictPtrTraits,size_t> radius,

    const torch::PackedTensorAccessor<float,    0,torch::RestrictPtrTraits,size_t> scale_radius,

    const int n_max_neighbors,
    const int n_bins,

    torch::PackedTensorAccessor<int,      2,torch::RestrictPtrTraits,size_t> edges,

    torch::PackedTensorAccessor<int,      0,torch::RestrictPtrTraits,size_t> i_edges
    )
{


}





// This naive implementation makes use of the packed accessor structure provided by the pytorch API.
// Not used going forward due to compatibility issues.
std::vector<torch::Tensor> frnn_cuda_forward(
        torch::Tensor neighbor_bins,
        torch::Tensor pts,
        torch::Tensor pt_idxs,
        torch::Tensor first_pt_idxs,
        torch::Tensor radius,
        torch::Tensor scale_radius )
{

    auto rad = radius.item();

    printf("rad: %i", find);


    const int n_threadsx = 16;
    const int n_threadsy = 16;
    //const int n_threads = 1024;

    const int n_edges0 = 10000000;
    const int n_edges1 = 2;


    //const dim3 blocks((n_edgesx*n_edgesy + n_threads - 1) / n_threads);

    const int blocks = 8192;
    const dim3 threads(n_threadsx, n_threadsy);

    const auto n_bins = neighbor_bins.size(0);
    const auto n_max_neighbors = neighbor_bins.size(1);

    auto edges = torch::zeros({n_edges0, n_edges1}, torch::kInt32).sub_(1);
    auto i_edges = torch::zeros(1, torch::kInt32);


    frnn_cuda_forward_kernel<<<blocks, threads>>>(

        neighbor_bins.packed_accessor<int,      2,torch::RestrictPtrTraits,size_t>(),

        pts.packed_accessor<float,              2,torch::RestrictPtrTraits,size_t>(),

        pt_idxs.packed_accessor<int,            1,torch::RestrictPtrTraits,size_t>(),

        first_pt_idxs.packed_accessor<int,      1,torch::RestrictPtrTraits,size_t>(),

        radius.packed_accessor<float,           0,torch::RestrictPtrTraits,size_t>(),

        scale_radius.packed_accessor<float,     0,torch::RestrictPtrTraits,size_t>(),

        n_max_neighbors,
        n_bins,

        edges.packed_accessor<int,              2,torch::RestrictPtrTraits,size_t>(),

        i_edges.packed_accessor<int,            0,torch::RestrictPtrTraits,size_t>() );

    return {edges, i_edges, neighbor_bins, pts, pt_idxs, first_pt_idxs, radius, scale_radius};
}











