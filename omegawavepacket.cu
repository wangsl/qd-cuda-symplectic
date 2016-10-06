
#include <iostream>
#include <helper_cuda.h>

#include "omegawavepacket.h"
#include "cudaUtils.h"
#include "matlabUtils.h"
#include "matlabData.h"

OmegaWavepacket::OmegaWavepacket(const int &omega_,
				 const double * &potential_dev_, 
				 cublasHandle_t &cublas_handle_,
				 cufftHandle &cufft_plan_D2Z_,
				 cufftHandle &cufft_plan_Z2D_
				 ) :
  omega(omega_), 
  potential_dev(potential_dev_),
  cublas_handle(cublas_handle_), 
  cufft_plan_D2Z(cufft_plan_D2Z_),
  cufft_plan_Z2D(cufft_plan_Z2D_),
  weighted_psi_real(0), weighted_psi_imag(0), 
  weighted_psi_real_dev(0), weighted_psi_imag_dev(0)
{ 
  insist(weighted_psi_real && weighted_psi_imag && potential_dev);

  copy_weighted_psi_from_device_to_host();
}

OmegaWavepacket::~OmegaWavepacket()
{
  std::cout << " Destroy OmegaWavepacket, Omega: " << omega << std::endl;

  weighted_psi_real = 0;
  weighted_psi_imag = 0;

  _CUDA_FREE_(weighted_psi_real_dev);
  _CUDA_FREE_(weighted_psi_imag_dev);
}

void OmegaWavepacket::copy_weighted_psi_from_host_to_device()
{
  std::cout << " Copy OmegaWavepacket from host to device, Omega: " << omega << std::endl;

  setup_weighted_psi();
  
  insist(weighted_psi_real && weighted_psi_imag);
  
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;
  
  if(!weighted_psi_real_dev) 
    checkCudaErrors(cudaMalloc(&weighted_psi_real_dev, n1*n2*n_theta*sizeof(double)));
  
  if(!weighted_psi_imag_dev) 
    checkCudaErrors(cudaMalloc(&weighted_psi_imag_dev, n1*n2*n_theta*sizeof(double)));
  
  insist(weighted_psi_real_dev && weighted_psi_imag_dev);
  
  checkCudaErrors(cudaMemcpyAsync(weighted_psi_real_dev, weighted_psi_real, n1*n2*n_theta*sizeof(double), 
				  cudaMemcpyHostToDevice));
  
  checkCudaErrors(cudaMemcpyAsync(weighted_psi_imag_dev, weighted_psi_imag, n1*n2*n_theta*sizeof(double), 
				  cudaMemcpyHostToDevice));
}

void OmegaWavepacket::copy_weighted_psi_from_device_to_host()
{
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;
  
  insist(weighted_psi_real && weighted_psi_imag);
  insist(weighted_psi_real_dev && weighted_psi_imag_dev);
  
  checkCudaErrors(cudaMemcpyAsync(weighted_psi_real, weighted_psi_real_dev, n1*n2*n_theta*sizeof(double), 
				  cudaMemcpyDeviceToHost));
  
  checkCudaErrors(cudaMemcpyAsync(weighted_psi_imag, weighted_psi_imag_dev, n1*n2*n_theta*sizeof(double), 
				  cudaMemcpyDeviceToHost));
}

void OmegaWavepacket::setup_weighted_psi()
{
  if(weighted_psi_real || weighted_psi_imag) return;

  const int &omega_min = MatlabData::wavepacket_parameters()->omega_min;
  Vec<RVec> &weighted_wavepackets_real = MatlabData::wavepacket_parameters()->weighted_wavepackets_real;
  Vec<RVec> &weighted_wavepackets_imag = MatlabData::wavepacket_parameters()->weighted_wavepackets_imag;
  
  const int omega_index = omega - omega_min;
  
  weighted_psi_real = weighted_wavepackets_real[omega_index];
  weighted_psi_imag = weighted_wavepackets_imag[omega_index];
}
