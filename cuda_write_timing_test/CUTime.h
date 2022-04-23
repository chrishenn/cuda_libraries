#ifdef __CUDACC__
#ifndef _TIME
#define _TIME

class CUTime {
  public:
    void run_time(const int size);
  };

#endif
#endif
