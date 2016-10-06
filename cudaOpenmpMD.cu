
#include <iostream>
#include <helper_cuda.h>

#include "cudaOpenmpMD.h"
#include "cudaUtils.h"
#include "matlabUtils.h"

inline static void divide_into_chunks(const int n, const int m, int *chunks)
{
  for(int i = 0; i < m; i++) chunks[i] = n/m;
  for(int i = 0; i < n-n/m*m; i++) chunks[i]++;
  int s = 0; for(int i = 0; i < m; i++) s += chunks[i];
  insist(s == n);
}

CUDAOpenmpMD::CUDAOpenmpMD() :
  _n_devices(0)
{ 
  setup_n_devices();
}

CUDAOpenmpMD::~CUDAOpenmpMD() 
{ 
  reset_devices();
}

void CUDAOpenmpMD::setup_n_devices()
{
  if(_n_devices) return;
  
  checkCudaErrors(cudaGetDeviceCount(&_n_devices));

  if(n_devices() == 1)
    std::cout << " There is 1 GPU card" << std::endl;
  else
    std::cout << " There are " << n_devices() << " GPU cards" << std::endl;
}

void CUDAOpenmpMD::devices_synchoronize()
{
  for(int i_dev = 0; i_dev < n_devices(); i_dev++) {
    checkCudaErrors(cudaSetDevice(i_dev));
    checkCudaErrors(cudaDeviceSynchronize());
  }
}

void CUDAOpenmpMD::devices_memory_usage() const
{
  for(int i_dev = 0; i_dev < n_devices(); i_dev++) {
    checkCudaErrors(cudaSetDevice(i_dev));
    cudaUtils::gpu_memory_usage();
  }
}

void CUDAOpenmpMD::reset_devices()
{
  for(int i_dev = 0; i_dev < n_devices(); i_dev++) {
    checkCudaErrors(cudaSetDevice(i_dev));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaDeviceReset());
  }
}

void CUDAOpenmpMD::test()
{
  devices_memory_usage();
  devices_synchoronize();
}
