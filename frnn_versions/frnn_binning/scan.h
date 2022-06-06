#ifdef __CUDACC__
#ifndef _SCAN

#define _SCAN

class scan {
  public:
    void scan_launch(
        int* data_in,
        int* data_out,
        int tot_size
    );

    void broadcast_add_launch(
        int* data_out,
        int* glob_temp,
        int tot_size,

        dim3 blocks,
        dim3 threads);
};

#endif
#endif
