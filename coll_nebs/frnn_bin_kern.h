#ifndef _NEBS
#define _NEBS

class frnn_bin_kern {
  public:
    void frnn_neighbors_launch(
        float* radius_at_each_scabin,
        int* scabin_offsets,
        int* neighbors,
        int* neb_counts,

        int num_binstotal,
        int num_bins_perimg,
        int num_imgs,
        int num_scabin,
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
        float*   pts,
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
