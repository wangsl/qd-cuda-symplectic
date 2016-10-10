
#include <iostream>
#include <helper_cuda.h>
#include <omp.h>

#include "cudaOpenmpMD.h"
#include "cudaUtils.h"
#include "matlabUtils.h"
#include "matlabData.h"

#include "evolutionUtils.h"

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
  setup_wavepackets_on_single_device();
}

CUDAOpenmpMD::~CUDAOpenmpMD() 
{ 
  devices_memory_usage();
  destroy_wavepackets_on_single_device();
  reset_devices();
}

void CUDAOpenmpMD::setup_n_devices()
{
  if(_n_devices) return;

  _n_devices = cudaUtils::n_devices();
  
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
    cudaUtils::device_memory_usage();
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

void CUDAOpenmpMD::setup_wavepackets_on_single_device()
{ 
  insist(n_devices() > 0);
  
  insist(wavepackets_on_single_device.size() == 0);
  wavepackets_on_single_device.resize(n_devices(), 0);

  const int &n = wavepackets_on_single_device.size();

  Vec<int> omegas(n);

  const int &omega_min = MatlabData::wavepacket_parameters()->omega_min;
  const int &omega_max = MatlabData::wavepacket_parameters()->omega_max;
  const int n_omegas = omega_max - omega_min + 1;

  divide_into_chunks(n_omegas, n, omegas);

  std::cout << " Omegas on devices: "; 
  omegas.show_in_one_line();
  
  int omega_start = 0;
  for(int i_dev = 0; i_dev < n; i_dev++) {
    
    checkCudaErrors(cudaSetDevice(i_dev));
    
    const int n_omegas = omegas[i_dev];
    
    wavepackets_on_single_device[i_dev] = 
      new WavepacketsOnSingleDevice(i_dev, omega_start+omega_min, n_omegas);
    
    insist(wavepackets_on_single_device[i_dev]);

    omega_start += n_omegas;
  }
  
  devices_synchoronize();
  devices_memory_usage();
}

void CUDAOpenmpMD::destroy_wavepackets_on_single_device()
{
  const int &n = wavepackets_on_single_device.size();
  for(int i = 0; i < n; i++) {
    if(wavepackets_on_single_device[i]) { 
      delete wavepackets_on_single_device[i];
      wavepackets_on_single_device[i] = 0; 
    }
  }
  wavepackets_on_single_device.resize(0);
}

void CUDAOpenmpMD::test()
{
  for(int L = 0; L < MatlabData::time()->total_steps; L++) {

  omp_set_num_threads(n_devices());
#pragma omp parallel for default(shared)
  for(int i_dev = 0; i_dev < n_devices(); i_dev++)
    wavepackets_on_single_device[i_dev]->test_parallel();
  
  for(int i_dev = 0; i_dev < n_devices(); i_dev++)
    wavepackets_on_single_device[i_dev]->test();

  }
}

