#ifndef _NEBS
#define _NEBS

#include <torch/types.h>

class frnn_bin_kern {
  public:
    void frnn_neighbors_launch(
            int* neighbors,
            int* neb_counts,

            float* scabin_linrads,
            int* n_xbin_eachscale,
            int* n_xbin_eachscale_excume,
            int* glob_to_scabin,
            int* glob_to_im_i,

            float xmin,
            float xmax,

            int n_scabin,
            int n_xbin,
            int n_binstotal,
            int batch_size,
            int neighb_size1,

            dim3 blocks,
            dim3 threads
    );

    void frnn_neighbors_prune_launch(
        int* neighbors,
		int* neb_counts,

		int num_binstotal,
		int neighb_size1,

		dim3 blocks,
		dim3 threads
      );

    void frnn_bin_counts_launch(
        torch::Tensor   pts,
        long*     img_ids,
        float*   radius_at_each_scabin,
        int*     scabin_offsets,
        int*     bin_counts,
        int*     bid2ptid,
        int*     write_count,

        float    scale_radius,
        float    xmin,
        float    scalemin,
        int      pts_size0,
        int      pts_size1,
        int      num_bins_perimg,
        int      num_binstotal,
        int pts_x_col,
        int pts_scale_col,

        dim3 blocks,
        dim3 threads
      );

    void frnn_ptids_launch(
        int*      bin_counts,
        int*      bin_offsets,
        int*      bid2ptid,
        int*      bin_ptids,
        int*      bin_wcounts,

        int       pts_size0,

        dim3 blocks,
        dim3 threads
      );

  };
#endif
