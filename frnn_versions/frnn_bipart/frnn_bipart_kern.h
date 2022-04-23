#ifndef _FRNN_BIPART_KERN
#define _FRNN_BIPART_KERN

#include <torch/types.h>

class frnn_bipart_kern {
  public:

    void frnn_bipart_kern_launch(

        int* neighbors,
        int* bin_counts,
        int* bin_offsets,
        int* bin_ptids,
        torch::Tensor pts,

        int* glob_counts,
        long* edges,

        float radius,
        float scale_radius,

        int max_binsize,
        int bin_stride,

        int neighb_size1,
        int pts_size1,

        dim3 blocks,
        dim3 threads,
        long shared
    );
};
#endif
