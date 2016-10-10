
#include <iostream>
#include <helper_cuda.h>

#include "wavepacketson1device.h"
#include "cudaUtils.h"
#include "matlabUtils.h"
#include "matlabData.h"
#include "evolutionUtils.h"

#include "evolutionAux.cu"

__constant__ EvolutionUtils::RadialCoordinate r1_dev;
__constant__ EvolutionUtils::RadialCoordinate r2_dev;

WavepacketsOnSingleDevice::
WavepacketsOnSingleDevice(const int device_index_,
			  const int omega_start_,
			  const int n_omegas_) :
  _device_index(device_index_),
  omega_start(omega_start_),
  n_omegas(n_omegas_),
  potential_dev(0),
  cufft_work_dev(0),
  omega_wavepacket_from_left_device(0),
  omega_wavepacket_from_right_device(0),
  _has_created_cublas_handle(0),
  _has_cufft_plans(0)
{ 
  insist(_device_index >= 0);
  setup_data_on_device();
}

int WavepacketsOnSingleDevice::current_device_index() const
{
  int dev_index = -1;
  checkCudaErrors(cudaGetDevice(&dev_index));
  return dev_index;
}

void WavepacketsOnSingleDevice::setup_device() const
{
  if(current_device_index() != device_index()) 
    checkCudaErrors(cudaSetDevice(device_index()));
}

void WavepacketsOnSingleDevice::setup_data_on_device()
{
  setup_device();

  std::cout << " Setup data on device: " << device_index() << std::endl;

  setup_constant_memory_on_device();

  setup_cublas_handle();
  setup_cufft_plans();

  setup_work_spaces_on_device();
  
  setup_potential_on_device();
  setup_omega_wavepackets();
}

void WavepacketsOnSingleDevice::destroy_data_on_device()
{ 
  setup_device();
  
  std::cout << " Destroy data on device: " << device_index() << std::endl;

  for(int i = 0; i < omega_wavepackets.size(); i++) {
    if(omega_wavepackets[i]) { delete omega_wavepackets[i]; omega_wavepackets[i] = 0; }
  }
  omega_wavepackets.resize(0);

  _CUDA_FREE_(potential_dev);
  _CUDA_FREE_(cufft_work_dev);
  _CUDA_FREE_(omega_wavepacket_from_left_device);
  _CUDA_FREE_(omega_wavepacket_from_right_device);
  
  destroy_cublas_handle();
  destroy_cufft_plans();
}

void WavepacketsOnSingleDevice::setup_potential_on_device()
{
  if(potential_dev) return;

  std::cout << " Allocate and copy potential on device: " << current_device_index() << std::endl;
  
  const double *potential = MatlabData::potential();
  insist(potential);
  
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;
  
  checkCudaErrors(cudaMalloc(&potential_dev, n1*n2*n_theta*sizeof(double)));
  insist(potential_dev);
  
  checkCudaErrors(cudaMemcpyAsync(potential_dev, potential, n1*n2*n_theta*sizeof(double),
				  cudaMemcpyHostToDevice));
}

void WavepacketsOnSingleDevice::setup_cublas_handle()
{
  if(_has_created_cublas_handle) return;

  std::cout << " Setup cuBLAS handle on device: " << current_device_index() << std::endl;
  
  insist(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);
  
  _has_created_cublas_handle = 1;
}

void WavepacketsOnSingleDevice::destroy_cublas_handle()
{ 
  if(!_has_created_cublas_handle) return;
  
  std::cout << " Destroy cuBLAS handle on device: " << current_device_index() << std::endl;

  insist(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);

  _has_created_cublas_handle = 0;
}

void WavepacketsOnSingleDevice::setup_cufft_plans()
{
  if(_has_cufft_plans) return;

  std::cout << " Setup cuFFT handles on device: " << current_device_index() << std::endl;

  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;

  /* wavepacket psi is from Matlab, which is column-major format, while cuFFT is using row-major format
   * so to switch dimensions, after D2Z FFT, the output data is { n2, n1/2+1 }, 
   * it is still column-major format
   */
  const int dims [] = { n2, n1 };
  
  insist(cufftPlanMany(&cufft_plan_D2Z, 2, const_cast<int *>(dims), 
		       NULL, 1, n1*n2,
		       NULL, 1, n1*n2,
		       CUFFT_D2Z, n_theta) == CUFFT_SUCCESS);

  cudaUtils::cufft_work_size(cufft_plan_D2Z, "D2Z");

  insist(cufftPlanMany(&cufft_plan_Z2D, 2, const_cast<int *>(dims), 
		       NULL, 1, n1*n2,
		       NULL, 1, n1*n2,
		       CUFFT_Z2D, n_theta) == CUFFT_SUCCESS);

  cudaUtils::cufft_work_size(cufft_plan_Z2D, "Z2D");
  
  _has_cufft_plans = 1;
}

void WavepacketsOnSingleDevice::destroy_cufft_plans()
{ 
  if(!_has_cufft_plans) return;

  std::cout << " Destroy cuFFT handles on device: " << current_device_index() << std::endl;

  insist(cufftDestroy(cufft_plan_D2Z) == CUFFT_SUCCESS);
  insist(cufftDestroy(cufft_plan_Z2D) == CUFFT_SUCCESS);

  _has_cufft_plans = 0;
}

void WavepacketsOnSingleDevice::setup_omega_wavepackets()
{
  insist(omega_wavepackets.size() == 0);
  
  omega_wavepackets.resize(n_omegas, 0);

  insist(cufft_work_dev);
  
  for(int i = 0; i < n_omegas; i++) {
    omega_wavepackets[i] = new OmegaWavepacket(i+omega_start, potential_dev,
					       cublas_handle, cufft_plan_D2Z, cufft_plan_Z2D,
					       cufft_work_dev);
    insist(omega_wavepackets[i]);
  }
}

void WavepacketsOnSingleDevice::setup_constant_memory_on_device()
{
  std::cout << " Setup constant memory on device: " << current_device_index() << std::endl;

  EvolutionUtils::copy_radial_coordinate_to_device(r1_dev, MatlabData::r1());
  EvolutionUtils::copy_radial_coordinate_to_device(r2_dev, MatlabData::r2());
}

void WavepacketsOnSingleDevice::setup_work_spaces_on_device()
{
  setup_device();
  
  if(!cufft_work_dev) {
    std::cout << " Setup cuFFT work on device: " << current_device_index() << std::endl;
    const int &n1 = MatlabData::r1()->n;
    const int &n2 = MatlabData::r2()->n;
    const int &n_theta = MatlabData::theta()->n;
    checkCudaErrors(cudaMalloc(&cufft_work_dev, (n1/2+1)*n2*n_theta*2*sizeof(double)));
    insist(cufft_work_dev);
  }

  if(device_index() != 0) {
    if(!omega_wavepacket_from_left_device) {
      std::cout << " Setup wavepacket from left on device: " << current_device_index() << std::endl;
      const int &n1 = MatlabData::r1()->n;
      const int &n2 = MatlabData::r2()->n;
      const int &n_theta = MatlabData::theta()->n;
      checkCudaErrors(cudaMalloc(&omega_wavepacket_from_left_device, n1*n2*n_theta*sizeof(double)));
    }
  }
  
  if(device_index() < cudaUtils::n_devices()-1) {
    if(!omega_wavepacket_from_right_device) {
      std::cout << " Setup wavepacket from right on device: " << current_device_index() << std::endl;
      const int &n1 = MatlabData::r1()->n;
      const int &n2 = MatlabData::r2()->n;
      const int &n_theta = MatlabData::theta()->n;
      checkCudaErrors(cudaMalloc(&omega_wavepacket_from_right_device, n1*n2*n_theta*sizeof(double)));
    }
  }
}

void WavepacketsOnSingleDevice::test_parallel()
{
  setup_device();
  for(int i = 0; i < n_omegas; i++) {
    omega_wavepackets[i]->test_parallel();
  }
}

void WavepacketsOnSingleDevice::test()
{
  setup_device();
  std::cout << " Test on device: " << current_device_index() << std::endl;
  for(int i = 0; i < n_omegas; i++)
    std::cout << " " << omega_wavepackets[i]->wavepacket_module() 
	      << " " << omega_wavepackets[i]->kinetic_energy() 
	      << " " << omega_wavepackets[i]->potential_energy()
	      << std::endl;

}
