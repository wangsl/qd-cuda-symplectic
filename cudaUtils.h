
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <chrono>
#include <thread>

#define _CUDA_FREE_(x) if(x) { checkCudaErrors(cudaFree(x)); x = 0; }

#define _CUDA_STREAM_CREATE_(x) {					\
    if(!x) {								\
      x = (cudaStream_t *) malloc(sizeof(cudaStream_t));		\
      insist(x);							\
      checkCudaErrors(cudaStreamCreate(x));				\
    }									\
  }			       

#define _CUDA_EVENT_CREATE_(x) {					\
    if(!x) {								\
      x = (cudaEvent_t *) malloc(sizeof(cudaEvent_t));			\
      insist(x);							\
      checkCudaErrors(cudaEventCreateWithFlags(x, cudaEventDisableTiming)); \
    }									\
  }

#define _CUDA_STREAM_DESTROY_(x) {		\
    if(x) {					\
      checkCudaErrors(cudaStreamDestroy(*x));	\
      free(x);					\
      x = 0;					\
    }						\
  }

#define _CUDA_EVENT_DESTROY_(x) {		\
    if(x) {					\
      checkCudaErrors(cudaEventDestroy(*x));	\
      free(x);					\
      x = 0;					\
    }						\
  }

#define _NTHREADS_ 512
#define _POTENTIAL_CUTOFF_ -1.0e+6

namespace cudaUtils {
  
  inline int number_of_blocks(const int n_threads, const int n)
  { return n/n_threads*n_threads == n ? n/n_threads : n/n_threads+1; }

  inline int n_devices()
  {
    int _n_devices = -1;
    checkCudaErrors(cudaGetDeviceCount(&_n_devices));
    return _n_devices;
  }
  
  void device_memory_usage();

  void cufft_work_size(const cufftHandle &plan, const char *type = 0);
}

inline char *time_now()
{
  std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
  std::time_t time_now  = std::chrono::system_clock::to_time_t(now);
  char *time = std::ctime(&time_now); 
  char *pch = strchr(time, '\n');
  if(pch) pch[0] = '\0';
  return time;
}

#endif /* CUDA_UTILS_H */
  

