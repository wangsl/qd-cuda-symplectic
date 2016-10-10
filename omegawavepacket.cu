
#include <iostream>
#include <helper_cuda.h>

#include "omegawavepacket.h"
#include "cudaUtils.h"
#include "matlabUtils.h"
#include "matlabData.h"

#include "evolutionAux.cu"

OmegaWavepacket::OmegaWavepacket(int omega_,
				 const double *potential_dev_, 
				 cublasHandle_t &cublas_handle_,
				 cufftHandle &cufft_plan_D2Z_,
				 cufftHandle &cufft_plan_Z2D_,
				 double * &cufft_work_dev_
				 ) :
  omega(omega_), 
  potential_dev(potential_dev_),
  cublas_handle(cublas_handle_), 
  cufft_plan_D2Z(cufft_plan_D2Z_),
  cufft_plan_Z2D(cufft_plan_Z2D_),
  cufft_work_dev(cufft_work_dev_),
  weighted_psi_real(0), weighted_psi_imag(0), 
  weighted_psi_dev(0),
  weighted_psi_real_dev(0), weighted_psi_imag_dev(0),
  weighted_associated_legendres_dev(0),
  weighted_legendre_psi_dev(0),
  H_weighted_psi_dev(0), H_weighted_legendre_psi_dev(0)
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
  
  _CUDA_FREE_(weighted_legendre_psi_dev);
  _CUDA_FREE_(H_weighted_psi_dev);
  _CUDA_FREE_(H_weighted_legendre_psi_dev);
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

double OmegaWavepacket::dot_product_with_volume_element(const double *x_dev, const double *y_dev) const
{
  insist(x_dev && y_dev);

  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;
  
  const double &dr1 = MatlabData::r1()->dr;
  const double &dr2 = MatlabData::r2()->dr;
  
  double s = 0.0;
  insist(cublasDdot(cublas_handle, n1*n2*n_theta, 
		    x_dev, 1, y_dev, 1, &s) == CUBLAS_STATUS_SUCCESS);
  
  s *= dr1*dr2;

  return s;
}

void OmegaWavepacket::_calculate_wavepacket_module()
{
  const double s = dot_product_with_volume_element(weighted_psi_dev, weighted_psi_dev);
  
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

void OmegaWavepacket::_calculate_kinetic_on_weighted_psi()
{
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;
  
  if(!H_weighted_psi_dev) 
    checkCudaErrors(cudaMalloc(&H_weighted_psi_dev, n1*n2*n_theta*sizeof(double)));
  
  insist(H_weighted_psi_dev);
  
  insist(weighted_psi_dev == weighted_psi_real_dev || weighted_psi_dev == weighted_psi_imag_dev);
  insist(cufft_work_dev);
  
  insist(cufftExecD2Z(cufft_plan_D2Z, (cufftDoubleReal *) weighted_psi_dev,
		      (cufftDoubleComplex *) cufft_work_dev) == CUFFT_SUCCESS);
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, (n1/2+1)*n2*n_theta);
  
  _psi_times_kinetic_energy_<<<n_blocks, n_threads, (n1/2+1+n2)*sizeof(double)>>>
    ((Complex *) cufft_work_dev, (const Complex *) cufft_work_dev, n1, n2, n_theta);
  
  insist(cufftExecZ2D(cufft_plan_Z2D, (cufftDoubleComplex *) cufft_work_dev,
		      (cufftDoubleReal *) H_weighted_psi_dev) == CUFFT_SUCCESS);
  
  const double f = 1.0/(n1*n2);

  insist(cublasDscal(cublas_handle, n1*n2*n_theta, &f, H_weighted_psi_dev, 1) 
	 == CUBLAS_STATUS_SUCCESS);
}

void OmegaWavepacket::_calculate_potential_on_weighted_psi()
{
  insist(H_weighted_psi_dev);
  
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2*n_theta);
  
  _psi_time_potential_energy_<<<n_blocks, n_threads>>>(H_weighted_psi_dev, weighted_psi_dev, potential_dev,
						       n1*n2*n_theta);
}

void OmegaWavepacket::test_parallel()
{
  weighted_psi_dev = weighted_psi_real_dev;
  
  _calculate_kinetic_on_weighted_psi();
  _kinetic_energy_from_real = dot_product_with_volume_element(weighted_psi_dev, H_weighted_psi_dev);

  _calculate_potential_on_weighted_psi();
  _potential_energy_from_real = dot_product_with_volume_element(weighted_psi_dev, H_weighted_psi_dev);
  _potential_energy_from_real -= _kinetic_energy_from_real;
  
  
  weighted_psi_dev = weighted_psi_imag_dev;

  _calculate_kinetic_on_weighted_psi();
  _kinetic_energy_from_imag = dot_product_with_volume_element(weighted_psi_dev, H_weighted_psi_dev);
  
  _calculate_potential_on_weighted_psi();
  _potential_energy_from_imag = dot_product_with_volume_element(weighted_psi_dev, H_weighted_psi_dev);
  _potential_energy_from_imag -= _kinetic_energy_from_imag;
  
  calculate_wavepacket_module();
}
