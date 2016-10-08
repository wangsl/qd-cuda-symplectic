
#include <iostream>
#include <helper_cuda.h>

#include "omegawavepacket.h"
#include "cudaUtils.h"
#include "matlabUtils.h"
#include "matlabData.h"

OmegaWavepacket::OmegaWavepacket(int omega_,
				 const double *potential_dev_, 
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
  weighted_psi_dev(0),
  weighted_psi_real_dev(0), weighted_psi_imag_dev(0),
  weighted_associated_legendres_dev(0)
{ 
  insist(potential_dev);

  copy_weighted_psi_from_host_to_device();
  copy_weighted_associated_legendres_to_device();
}

OmegaWavepacket::~OmegaWavepacket()
{
  std::cout << " Destroy OmegaWavepacket, Omega: " << omega << std::endl;

  weighted_psi_real = 0;
  weighted_psi_imag = 0;

  weighted_psi_dev = 0;

  _CUDA_FREE_(weighted_psi_real_dev);
  _CUDA_FREE_(weighted_psi_imag_dev);
  _CUDA_FREE_(weighted_associated_legendres_dev);
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
  if(weighted_psi_real || weighted_psi_imag) {
    insist(weighted_psi_real && weighted_psi_imag);
    return;
  }
  
  const int &omega_min = MatlabData::wavepacket_parameters()->omega_min;
  Vec<RVec> &weighted_wavepackets_real = MatlabData::wavepacket_parameters()->weighted_wavepackets_real;
  Vec<RVec> &weighted_wavepackets_imag = MatlabData::wavepacket_parameters()->weighted_wavepackets_imag;
  
  const int omega_index = omega - omega_min;
  
  weighted_psi_real = weighted_wavepackets_real[omega_index];
  weighted_psi_imag = weighted_wavepackets_imag[omega_index];
}

void OmegaWavepacket::copy_weighted_associated_legendres_to_device()
{
  if(weighted_associated_legendres_dev) return;

  std::cout << " Copy associated Legendres to device, Omega: " << omega << std::endl;
  
  const int &n_theta = MatlabData::theta()->n;
  const int &l_max = MatlabData::wavepacket_parameters()->l_max;
  const int &omega_min = MatlabData::wavepacket_parameters()->omega_min;
  
  const int n_ass_Legs = l_max - omega + 1;
  
  const int omega_index = omega - omega_min;
  
  const Vec<RMat> &Legendres = MatlabData::wavepacket_parameters()->weighted_associated_legendres;
  const RMat &ass_Leg = Legendres[omega_index];
  
  insist(ass_Leg.rows() == n_theta && ass_Leg.columns() == n_ass_Legs);

  checkCudaErrors(cudaMalloc(&weighted_associated_legendres_dev, n_theta*n_ass_Legs*sizeof(double)));
  insist(weighted_associated_legendres_dev);
  
  checkCudaErrors(cudaMemcpyAsync(weighted_associated_legendres_dev, ass_Leg,
				  n_theta*n_ass_Legs*sizeof(double), cudaMemcpyHostToDevice));
}

void OmegaWavepacket::_calculate_wavepacket_module()
{
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;
  
  const double &dr1 = MatlabData::r1()->dr;
  const double &dr2 = MatlabData::r2()->dr;
  
  insist(weighted_psi_dev == weighted_psi_real_dev || weighted_psi_dev == weighted_psi_imag_dev);
  
  double s = 0.0;
  insist(cublasDdot(cublas_handle, n1*n2*n_theta, 
		    weighted_psi_dev, 1,
		    weighted_psi_dev, 1,
		    &s) == CUBLAS_STATUS_SUCCESS);

  s *= dr1*dr2;
  
  if(weighted_psi_dev == weighted_psi_real_dev)
    _wavepacket_module_from_real = s;
  else if(weighted_psi_dev == weighted_psi_imag_dev)
    _wavepacket_module_from_imag = s;
}

void OmegaWavepacket::calculate_wavepacket_module()
{
  weighted_psi_dev = weighted_psi_real_dev;
  _calculate_wavepacket_module();
  
  weighted_psi_dev = weighted_psi_imag_dev;
  _calculate_wavepacket_module();
}
