
#include <iostream>
#include <helper_cuda.h>

#include "wavepacketson1device.h"
#include "cudaUtils.h"
#include "matlabUtils.h"
#include "matlabData.h"

WavepacketsOnSingleDevice::
WavepacketsOnSingleDevice(const int device_index_,
			  const int omega_start_,
			  const int n_omegas_) :
  _device_index(device_index_),
  omega_start(omega_start_),
  n_omegas(n_omegas_),
  potential_dev(0),
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

  setup_cublas_handle();
  setup_cufft_plans();
  
  setup_potential_on_device();
}

void WavepacketsOnSingleDevice::destroy_data_on_device()
{ 
  setup_device();

  _CUDA_FREE_(potential_dev);
  
  destroy_cublas_handle();
  destroy_cufft_plans();
}

void WavepacketsOnSingleDevice::setup_potential_on_device()
{
  if(potential_dev) return;

  std::cout << " Allocate and copy potential on device: " << current_device_index()  << std::endl;
  
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

  std::cout << " Setup CUBLAS handle on device: " << current_device_index()  << std::endl;
  
  insist(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);
  
  _has_created_cublas_handle = 1;
}

void WavepacketsOnSingleDevice::destroy_cublas_handle()
{ 
  if(!_has_created_cublas_handle) return;
  
  std::cout << " Destroy CUBLAS handle on device: " << current_device_index()  << std::endl;

  insist(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);

  _has_created_cublas_handle = 0;
}

void WavepacketsOnSingleDevice::setup_cufft_plans()
{
  if(_has_cufft_plans) return;

  std::cout << " Setup CUFFT handles on device: " << current_device_index()  << std::endl;

  const int n1 = MatlabData::r1()->n;
  const int n2 = MatlabData::r2()->n;
  const int n_ass_Legs = MatlabData::wavepacket_parameters()->l_max + 1;
  
  const int dims [] = { n2, n1 };
  
  insist(cufftPlanMany(&cufft_plan_D2Z, 2, const_cast<int *>(dims), 
		       NULL, 1, n1*n2,
		       NULL, 1, n1*n2,
		       CUFFT_D2Z, n_ass_Legs) == CUFFT_SUCCESS);

  cudaUtils::cufft_work_size(cufft_plan_D2Z, "D2Z");
  
  insist(cufftPlanMany(&cufft_plan_Z2D, 2, const_cast<int *>(dims), 
		       NULL, 1, n1*n2,
		       NULL, 1, n1*n2,
		       CUFFT_Z2D, n_ass_Legs) == CUFFT_SUCCESS);

  cudaUtils::cufft_work_size(cufft_plan_Z2D, "Z2D");

  _has_cufft_plans = 1;
}

void WavepacketsOnSingleDevice::destroy_cufft_plans()
{ 
  if(!_has_cufft_plans) return;

  std::cout << " Destroy CUFFT handles on device: " << current_device_index()  << std::endl;

  insist(cufftDestroy(cufft_plan_D2Z) == CUFFT_SUCCESS);
  insist(cufftDestroy(cufft_plan_Z2D) == CUFFT_SUCCESS);

  _has_cufft_plans = 0;
}
