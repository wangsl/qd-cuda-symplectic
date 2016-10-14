
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
				 cudaStream_t * &computation_stream_,
				 double * &device_work_dev_
				 ) :
  omega(omega_), 
  potential_dev(potential_dev_),
  cublas_handle(cublas_handle_), 
  cufft_plan_D2Z(cufft_plan_D2Z_),
  cufft_plan_Z2D(cufft_plan_Z2D_),
  computation_stream(computation_stream_),
  device_work_dev(device_work_dev_),
  weighted_psi_real(0), weighted_psi_imag(0), 
  weighted_psi_real_dev(0), weighted_psi_imag_dev(0),
  work_dev(0),
  weighted_associated_legendres_dev(0),
  weighted_psi_dev(0), 
  legendre_psi_dev(0),
  T_angle_legendre_psi_dev(0),
  H_weighted_psi_dev(0)
{ 
  insist(potential_dev);
  insist(computation_stream);
  copy_weighted_psi_from_host_to_device();
  copy_weighted_associated_legendres_from_host_to_device();
  setup_work_dev();
}

OmegaWavepacket::~OmegaWavepacket()
{
  std::cout << " Destroy OmegaWavepacket, Omega: " << omega << std::endl;

  weighted_psi_real = 0;
  weighted_psi_imag = 0;

  weighted_psi_dev = 0;
  legendre_psi_dev = 0;
  T_angle_legendre_psi_dev = 0;
  H_weighted_psi_dev = 0;

  _CUDA_FREE_(weighted_psi_real_dev);
  _CUDA_FREE_(weighted_psi_imag_dev);
  _CUDA_FREE_(weighted_associated_legendres_dev);
  _CUDA_FREE_(work_dev);
}

void OmegaWavepacket::setup_weighted_psi_dev(const int part)
{
  if(part == _RealPart_)
    weighted_psi_dev = weighted_psi_real_dev;
  else if(part == _ImagPart_)
    weighted_psi_dev = weighted_psi_imag_dev;
  else
    insist(0);
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

void OmegaWavepacket::copy_weighted_psi_from_device_to_host() const
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

void OmegaWavepacket::copy_weighted_associated_legendres_from_host_to_device()
{
  if(weighted_associated_legendres_dev) return;

  std::cout << " Copy associated Legendres to device, Omega: " << omega;
  
  const int &n_theta = MatlabData::theta()->n;
  const int &l_max = MatlabData::wavepacket_parameters()->l_max;
  const int &omega_min = MatlabData::wavepacket_parameters()->omega_min;
  
  const int n_ass_Legs = l_max - omega + 1;
  
  const int omega_index = omega - omega_min;
  
  const Vec<RMat> &Legendres = MatlabData::wavepacket_parameters()->weighted_associated_legendres;
  const RMat &ass_Leg = Legendres[omega_index];
  
  insist(ass_Leg.rows() == n_theta && ass_Leg.columns() == n_ass_Legs);

  std::cout << ", size: " << n_theta << " " << n_ass_Legs << std::endl;

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
  insist(cublasDdot(cublas_handle, n1*n2*n_theta, x_dev, 1, y_dev, 1, &s) == CUBLAS_STATUS_SUCCESS);
  
  s *= dr1*dr2;
  
  return s;
}

double OmegaWavepacket::
dot_product_with_volume_element_for_legendres(const double *x_dev, const double *y_dev) const
{
  insist(x_dev && y_dev);

  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int n_Legs = MatlabData::wavepacket_parameters()->l_max - omega + 1;
  
  const double &dr1 = MatlabData::r1()->dr;
  const double &dr2 = MatlabData::r2()->dr;

  const double *x = x_dev + omega*n1*n2;
  const double *y = y_dev + omega*n1*n2;
  
  double s = 0.0;
  insist(cublasDdot(cublas_handle, n1*n2*n_Legs, x, 1, y, 1, &s) == CUBLAS_STATUS_SUCCESS);
  
  s *= dr1*dr2;

  return s;
}

void OmegaWavepacket::calculate_wavepacket_module()
{
  _wavepacket_module_from_real = dot_product_with_volume_element(weighted_psi_real_dev,
								 weighted_psi_real_dev);
  
  _wavepacket_module_from_imag = dot_product_with_volume_element(weighted_psi_imag_dev,
								 weighted_psi_imag_dev);
}

void OmegaWavepacket::calculate_radial_kinetic_add_to_H_weighted_psi_dev() const
{
  insist(weighted_psi_dev == weighted_psi_real_dev || weighted_psi_dev == weighted_psi_imag_dev);
  
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;

  double *cufft_tmp_dev = const_cast<double *>(memory_10());
  
  insist(cufftExecD2Z(cufft_plan_D2Z, (cufftDoubleReal *) weighted_psi_dev,
		      (cufftDoubleComplex *) cufft_tmp_dev) == CUFFT_SUCCESS);
  
  const int n_threads = _NTHREADS_;
  int n_blocks = cudaUtils::number_of_blocks(n_threads, (n1/2+1)*n2*n_theta);
  
  _psi_times_kinetic_energy_<<<n_blocks, n_threads, (n1/2+1+n2)*sizeof(double)>>>
    ((Complex *) cufft_tmp_dev, (const Complex *) cufft_tmp_dev, n1, n2, n_theta);
  
  insist(cufftExecZ2D(cufft_plan_Z2D, (cufftDoubleComplex *) cufft_tmp_dev,
		      (cufftDoubleReal *) cufft_tmp_dev) == CUFFT_SUCCESS);
  
  insist(H_weighted_psi_dev == memory_1());

  n_blocks = cudaUtils::number_of_blocks(n_threads, (n1/2+1)*2*n2*n_theta);
  
  _add_T_radial_weighted_psi_to_H_weighted_psi_<<<n_blocks, n_threads>>>(H_weighted_psi_dev,
									 cufft_tmp_dev,
									 n1, n2, n_theta);
}

void OmegaWavepacket::calculate_potential_add_to_H_weighted_psi_dev() const
{
  insist(weighted_psi_dev == weighted_psi_real_dev || weighted_psi_dev == weighted_psi_imag_dev);
  
  insist(H_weighted_psi_dev == memory_1());
  
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;
  
  const int n_threads = _NTHREADS_;
  const int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2*n_theta);
  
  _add_potential_weighted_psi_to_H_weighted_psi_<<<n_blocks, n_threads>>>(H_weighted_psi_dev, weighted_psi_dev, 
									  potential_dev, n1*n2*n_theta);
}

void OmegaWavepacket::setup_work_dev()
{
  if(work_dev) return;
  
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;

  std::cout << " Setup local work dev, Omega: " << omega
	    << " " << n1 << " " << n2 << " " << n_theta << std::endl;

  checkCudaErrors(cudaMalloc(&work_dev, n1*n2*n_theta*sizeof(double)));
  insist(work_dev);
}

void OmegaWavepacket::forward_legendre_transform()
{ 
  insist(weighted_psi_dev == weighted_psi_real_dev || weighted_psi_dev == weighted_psi_imag_dev);
  insist(weighted_associated_legendres_dev);

  legendre_psi_dev = const_cast<double *>(memory_1());

  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;
  const int n_Legs = MatlabData::wavepacket_parameters()->l_max - omega + 1;

  const double zero = 0.0;
  const double one = 1.0;
  
  double *legendre_psi_dev_ = legendre_psi_dev + omega*n1*n2;

  insist(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
		     n1*n2, n_Legs, n_theta, 
		     &one, 
		     weighted_psi_dev, n1*n2,
		     weighted_associated_legendres_dev, n_theta, 
		     &zero,
		     legendre_psi_dev_, n1*n2) == CUBLAS_STATUS_SUCCESS);
}

void OmegaWavepacket::backward_legendre_transform() const
{ 
  insist(weighted_psi_dev == weighted_psi_real_dev || weighted_psi_dev == weighted_psi_imag_dev);
  insist(weighted_associated_legendres_dev);

  insist(legendre_psi_dev == memory_1());
  
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;
  const int n_Legs = MatlabData::wavepacket_parameters()->l_max - omega + 1;
  
  const double zero = 0.0;
  const double one = 1.0;
  
  const double *legendre_psi_dev_ = legendre_psi_dev + omega*n1*n2;
  
  insist(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
		     n1*n2, n_theta, n_Legs,
		     &one, 
		     legendre_psi_dev_, n1*n2,
		     weighted_associated_legendres_dev, n_theta, 
		     &zero,
		     weighted_psi_dev, n1*n2) == CUBLAS_STATUS_SUCCESS);
}

void OmegaWavepacket::T_angle_legendre_psi_to_H_weighted_psi_dev()
{
  insist(T_angle_legendre_psi_dev == memory_10());
  //insist(H_weighted_psi_dev == memory_1());

  H_weighted_psi_dev = const_cast<double *>(memory_1());
  
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;
  const int n_Legs = MatlabData::wavepacket_parameters()->l_max - omega + 1;
  
  const double zero = 0.0;
  const double one = 1.0;

  const double *T_angle_legendre_psi_dev_ = T_angle_legendre_psi_dev + omega*n1*n2;
  
  insist(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
		     n1*n2, n_theta, n_Legs,
		     &one, 
		     T_angle_legendre_psi_dev_, n1*n2,
		     weighted_associated_legendres_dev, n_theta, 
		     &zero,
		     H_weighted_psi_dev, n1*n2) == CUBLAS_STATUS_SUCCESS);
}

void OmegaWavepacket::copy_T_angle_legendre_psi_to_device_work_dev()
{
  insist(T_angle_legendre_psi_dev == memory_0());
  
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &l_max = MatlabData::wavepacket_parameters()->l_max;
  
  checkCudaErrors(cudaMemcpy(device_work_dev, T_angle_legendre_psi_dev, 
			     n1*n2*(l_max+1)*sizeof(double),
			     cudaMemcpyDeviceToDevice));

  T_angle_legendre_psi_dev = const_cast<double *>(memory_10());
}

void OmegaWavepacket::calculate_T_bend_T_sym_add_to_T_angle_legendre_psi_dev()
{ 
  insist(legendre_psi_dev == memory_1());
  
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;
  const int &J = MatlabData::wavepacket_parameters()->J;

  insist(computation_stream);

  T_angle_legendre_psi_dev = const_cast<double *>(memory_0());

  checkCudaErrors(cudaMemsetAsync(T_angle_legendre_psi_dev, 0, n1*n2*n_theta*sizeof(double),
				  *computation_stream));

  const int n_Legs = MatlabData::wavepacket_parameters()->l_max - omega + 1;

  double *T_angle_legendre_psi_dev_ = T_angle_legendre_psi_dev + omega*n1*n2;
  const double *legendre_psi_dev_ = legendre_psi_dev + omega*n1*n2;
  
  const int n_threads = _NTHREADS_;
  int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2*n_Legs);
  
  _add_T_bend_T_sym_to_T_angle_legendre_psi_dev_<<<n_blocks, n_threads, 
    (n1+n2+1)*sizeof(double), *computation_stream>>>(T_angle_legendre_psi_dev_, 
						     legendre_psi_dev_,
						     n1, n2, n_Legs, J, omega);
}

void OmegaWavepacket::calculate_T_asym_add_to_T_angle_legendre_psi_dev(const double *psi_dev, 
								       const int omega1) const
{
  insist(omega1 == omega+1 || omega1 == omega-1);
  
  insist(T_angle_legendre_psi_dev == memory_0());
  
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &J = MatlabData::wavepacket_parameters()->J;
  
  const int omega_ = std::max(omega, omega1);
  
  const int n_Legs = MatlabData::wavepacket_parameters()->l_max - omega_ + 1;

  double *T_angle_legendre_psi_dev_ = T_angle_legendre_psi_dev + omega_*n1*n2;
  const double *legendre_psi_dev_ = psi_dev + omega_*n1*n2;
  
  const int n_threads = _NTHREADS_;
  int n_blocks = cudaUtils::number_of_blocks(n_threads, n1*n2*n_Legs);

  insist(computation_stream);
  
  _add_T_asym_to_T_angle_legendre_psi_dev_<<<n_blocks, n_threads, n1*sizeof(double),
    *computation_stream>>>(T_angle_legendre_psi_dev_, legendre_psi_dev_,
			   n1, n2, n_Legs,
			   J, omega, omega1);
}

void OmegaWavepacket::calculate_H_weighted_psi_dev()
{
  copy_T_angle_legendre_psi_to_device_work_dev();

  backward_legendre_transform();

  T_angle_legendre_psi_to_H_weighted_psi_dev();

  calculate_radial_kinetic_add_to_H_weighted_psi_dev();

  calculate_potential_add_to_H_weighted_psi_dev();

  calculate_energy_and_module();
}

void OmegaWavepacket::calculate_energy_and_module()
{
  insist(weighted_psi_dev == weighted_psi_real_dev || weighted_psi_dev == weighted_psi_imag_dev);
  insist(H_weighted_psi_dev == memory_1());
  
  const double module = dot_product_with_volume_element(weighted_psi_dev, weighted_psi_dev);
  const double energy = dot_product_with_volume_element(weighted_psi_dev, H_weighted_psi_dev);
  
  if(weighted_psi_dev == weighted_psi_real_dev) {
    _wavepacket_module_from_real = module;
    _energy_from_real = energy;
  } else {
    _wavepacket_module_from_imag = module;
    _energy_from_imag = energy;
  }
}

void OmegaWavepacket::test_parallel()
{
  insist(cublasSetStream(cublas_handle, NULL) == CUBLAS_STATUS_SUCCESS);
  
  calculate_wavepacket_module();

  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;

  H_weighted_psi_dev = const_cast<double *>(memory_1());
  checkCudaErrors(cudaMemset(H_weighted_psi_dev, 0, n1*n2*n_theta*sizeof(double)));

  setup_weighted_psi_dev(_RealPart_);

  calculate_radial_kinetic_add_to_H_weighted_psi_dev();
  _kinetic_energy_from_real = dot_product_with_volume_element(weighted_psi_dev, H_weighted_psi_dev);

  calculate_potential_add_to_H_weighted_psi_dev();
  _potential_energy_from_real = dot_product_with_volume_element(weighted_psi_dev, H_weighted_psi_dev);
  _potential_energy_from_real -= _kinetic_energy_from_real;

  H_weighted_psi_dev = const_cast<double *>(memory_1());
  checkCudaErrors(cudaMemset(H_weighted_psi_dev, 0, n1*n2*n_theta*sizeof(double)));

  setup_weighted_psi_dev(_ImagPart_);

  calculate_radial_kinetic_add_to_H_weighted_psi_dev();
  _kinetic_energy_from_imag = dot_product_with_volume_element(weighted_psi_dev, H_weighted_psi_dev);

  calculate_potential_add_to_H_weighted_psi_dev();
  _potential_energy_from_imag = dot_product_with_volume_element(weighted_psi_dev, H_weighted_psi_dev);
  _potential_energy_from_imag -= _kinetic_energy_from_imag;
}
