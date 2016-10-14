
#include <iostream>
#include <helper_cuda.h>
#include <omp.h>
#include <cuda_profiler_api.h>

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
  enable_peer_to_peer_access();
}

CUDAOpenmpMD::~CUDAOpenmpMD() 
{ 
  devices_memory_usage();
  destroy_wavepackets_on_single_device();
  disable_peer_to_peer_access();
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

  setup_devices_neighbours();

  setup_device_work_dev_on_devices();

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

void CUDAOpenmpMD::enable_peer_to_peer_access() const
{ 
  if(n_devices() == 1) return;

  std::cout << " Enable peer to peer memory access" << std::endl;
  
  for(int i_dev = 0; i_dev < n_devices(); i_dev++) {
    for(int j_dev = i_dev+1; j_dev < n_devices(); j_dev++) {
      checkCudaErrors(cudaSetDevice(i_dev));
      checkCudaErrors(cudaDeviceEnablePeerAccess(j_dev, 0));
      
      checkCudaErrors(cudaSetDevice(j_dev));
      checkCudaErrors(cudaDeviceEnablePeerAccess(i_dev, 0));
    }
  }
}

void CUDAOpenmpMD::disable_peer_to_peer_access() const
{
  if(n_devices() == 1) return;
  
  std::cout << " Disable peer to peer memory access" << std::endl;
  
  for(int i_dev = 0; i_dev < n_devices(); i_dev++) {
    for(int j_dev = i_dev+1; j_dev < n_devices(); j_dev++) {
      checkCudaErrors(cudaSetDevice(i_dev));
      checkCudaErrors(cudaDeviceDisablePeerAccess(j_dev));
      
      checkCudaErrors(cudaSetDevice(j_dev));
      checkCudaErrors(cudaDeviceDisablePeerAccess(i_dev));
    }
  }
}

void CUDAOpenmpMD::setup_devices_neighbours() const
{
  if(n_devices() == 1) return;
  
  std::cout << " Setup devices neighbours" << std::endl;

  const int &n = wavepackets_on_single_device.size();

  wavepackets_on_single_device[0]->setup_neighbours(0, wavepackets_on_single_device[1]);
  
  for(int i = 1; i < n-1; i++) {
    wavepackets_on_single_device[i]->setup_neighbours(wavepackets_on_single_device[i-1],
						      wavepackets_on_single_device[i+1]);
  }
  
  wavepackets_on_single_device[n-1]->setup_neighbours(wavepackets_on_single_device[n-2], 0);
}

void CUDAOpenmpMD::setup_device_work_dev_on_devices() const
{
  const int &n = wavepackets_on_single_device.size();
  for(int i = 0; i < n; i++)
    wavepackets_on_single_device[i]->setup_device_work_dev_and_copy_streams_events();
}

void CUDAOpenmpMD::test()
{
  
  for(int L = 0; L < MatlabData::time()->total_steps; L++) {

    std::cout << "\n Step: " << L+1 << ", " << time_now() << std::endl;

    checkCudaErrors(cudaProfilerStart());
    
    omp_set_num_threads(n_devices());

    for(int l = 0; l < 6; l++) {
#pragma omp parallel for default(shared)
      for(int i_dev = 0; i_dev < n_devices(); i_dev++)
	wavepackets_on_single_device[i_dev]->test_parallel();
    }
    
    checkCudaErrors(cudaProfilerStop());
    
    devices_synchoronize();

    for(int i_dev = 0; i_dev < n_devices(); i_dev++)
      wavepackets_on_single_device[i_dev]->test_serial();
  }

  std::cout << std::endl;
}
