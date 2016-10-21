
#include <iostream>
#include <helper_cuda.h>

#include "wavepacketson1device.h"
#include "cudaUtils.h"
#include "matlabUtils.h"
#include "matlabData.h"
#include "evolutionUtils.h"

#include "evolutionAux.cu"

/***
 * https://github.com/mohamso/icpads14
 * https://raw.githubusercontent.com/mohamso/icpads14/master/4/omp/src/Async.c
 * DOI: 10.1109/PADSW.2014.7097919
 ***/

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
  device_work_dev(0),
  omega_wavepacket_from_left_device(0),
  omega_wavepacket_from_right_device(0),
  _has_created_cublas_handle(0),
  _has_cufft_plans(0),
  computation_stream(0), computation_event(0),
  copy_to_left_stream(0), copy_to_left_event(0),
  copy_to_right_stream(0), copy_to_right_event(0),
  left(0), right(0)
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

  setup_computation_stream_and_event();

  setup_cublas_handle();
  setup_cufft_plans();

  setup_potential_on_device();
  setup_omega_wavepackets();
}

void WavepacketsOnSingleDevice::destroy_data_on_device()
{ 
  setup_device();
  
  std::cout << " Destroy data on device: " << device_index() << std::endl;

  for(int i = 0; i < omega_wavepackets.size(); i++) 
    if(omega_wavepackets[i]) { delete omega_wavepackets[i]; omega_wavepackets[i] = 0; }
  omega_wavepackets.resize(0);

  _CUDA_FREE_(potential_dev);
  _CUDA_FREE_(device_work_dev);
  
  destroy_cublas_handle();
  destroy_cufft_plans();

  destroy_streams_and_events();

  left = 0;
  right = 0;
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
  
  for(int i = 0; i < n_omegas; i++) {
    omega_wavepackets[i] = new OmegaWavepacket(i+omega_start, potential_dev, 
					       cublas_handle, cufft_plan_D2Z, cufft_plan_Z2D,
					       computation_stream,
					       device_work_dev);
    insist(omega_wavepackets[i]);
  }
}

void WavepacketsOnSingleDevice::setup_constant_memory_on_device()
{
  std::cout << " Setup constant memory on device: " << current_device_index() << std::endl;

  EvolutionUtils::copy_radial_coordinate_to_device(r1_dev, MatlabData::r1());
  EvolutionUtils::copy_radial_coordinate_to_device(r2_dev, MatlabData::r2());
}

void WavepacketsOnSingleDevice::setup_device_work_dev_and_copy_streams_events()
{
  setup_device();

  if(device_work_dev) return;
  
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int &n_theta = MatlabData::theta()->n;
  const int &l_max = MatlabData::wavepacket_parameters()->l_max;
  
  int size = 0;

  if(left) size += n1*n2*(l_max+1);
  if(right) size += n1*n2*(l_max+1);
  
  size = std::max(size, (n1/2+1)*2*n2*n_theta);

  std::cout << " Setup device work on device: " << current_device_index() 
	    << " " << size << " " << size*sizeof(double)/pow(1024.0, 2) << std::endl;

  checkCudaErrors(cudaMalloc(&device_work_dev, size*sizeof(double)));
  insist(device_work_dev);

  int current = 0;

  if(left) {
    if(!omega_wavepacket_from_left_device) {
      std::cout << " Setup wavepacket from left on device: " << current_device_index() << std::endl;
      omega_wavepacket_from_left_device = device_work_dev + current;
      current += n1*n2*(l_max+1);

      if(!copy_to_left_stream) { 
	copy_to_left_stream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
	insist(copy_to_left_stream);
	checkCudaErrors(cudaStreamCreate(copy_to_left_stream));
      }
      
      if(!copy_to_left_event) {
	copy_to_left_event = (cudaEvent_t *) malloc(sizeof(cudaEvent_t));
	insist(copy_to_left_event);
	checkCudaErrors(cudaEventCreateWithFlags(copy_to_left_event, cudaEventDisableTiming));
      }
    }
  }

  if(right) {
    if(!omega_wavepacket_from_right_device) {
      std::cout << " Setup wavepacket from right on device: " << current_device_index() << std::endl;
      omega_wavepacket_from_right_device = device_work_dev + current;
      current += n1*n2*(l_max+1);
      
      if(!copy_to_right_stream) { 
	copy_to_right_stream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
	insist(copy_to_right_stream);
	checkCudaErrors(cudaStreamCreate(copy_to_right_stream));
      }
      
      if(!copy_to_right_event) {
	copy_to_right_event = (cudaEvent_t *) malloc(sizeof(cudaEvent_t));
	insist(copy_to_right_event);
	checkCudaErrors(cudaEventCreateWithFlags(copy_to_right_event, cudaEventDisableTiming));
      }
    }
  }
}

void WavepacketsOnSingleDevice::setup_computation_stream_and_event()
{ 
  if(computation_stream || computation_event) {
    insist(computation_stream && computation_event);
    return;
  }
  
  std::cout << " Setup computation stream and event on device: " << current_device_index() << std::endl;
  
  computation_stream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
  insist(computation_stream);
  checkCudaErrors(cudaStreamCreate(computation_stream));
  
  computation_event = (cudaEvent_t *) malloc(sizeof(cudaEvent_t));
  insist(computation_event);
  checkCudaErrors(cudaEventCreate(computation_event));
}
  
void WavepacketsOnSingleDevice::destroy_streams_and_events()
{ 
  if(cudaUtils::n_devices() == 1) return;

  std::cout << " Destroy streams and events on device: " << device_index() << std::endl;
  
  if(computation_stream) {
    checkCudaErrors(cudaStreamDestroy(*computation_stream));
    free(computation_stream); computation_stream = 0; 
  }

  if(computation_event) {
    checkCudaErrors(cudaEventDestroy(*computation_event));	
    free(computation_event); computation_event = 0; 
  }

  if(copy_to_left_stream) {
    checkCudaErrors(cudaStreamDestroy(*copy_to_left_stream));
    free(copy_to_left_stream); copy_to_left_stream = 0;
  }

  if(copy_to_left_event) {
    checkCudaErrors(cudaEventDestroy(*copy_to_left_event));
    free(copy_to_left_event); copy_to_left_event = 0;
  }
  
  if(copy_to_right_stream) {
    checkCudaErrors(cudaStreamDestroy(*copy_to_right_stream));
    free(copy_to_right_stream); copy_to_right_stream = 0;
  }

  if(copy_to_right_event) {
    checkCudaErrors(cudaEventDestroy(*copy_to_right_event));
    free(copy_to_right_event); copy_to_right_event = 0;
  }
}

void WavepacketsOnSingleDevice::setup_neighbours(const WavepacketsOnSingleDevice *left_, 
						 const WavepacketsOnSingleDevice *right_)
{
  setup_device();
  left = left_; 
  right = right_;
  std::cout << " Neighbours on device: " << current_device_index()
	    << ", pointers: " << this << " " << left << " " << right << std::endl;
}

void WavepacketsOnSingleDevice::
forward_legendre_transform_and_copy_data_to_neighbour_devices(const int part)
{ 
  insist(part == _RealPart_ || part == _ImagPart_);

  setup_device();
 
  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int n_Legs = MatlabData::wavepacket_parameters()->l_max + 1;
  
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->setup_weighted_psi_dev(part);
  
  insist(computation_stream);
  insist(cublasSetStream(cublas_handle, *computation_stream) == CUBLAS_STATUS_SUCCESS);

  // single omega wavepacket

  if(n_omegas == 1) {
    
    OmegaWavepacket *wp = omega_wavepackets[0];
    wp->forward_legendre_transform();
    
    checkCudaErrors(cudaEventRecord(*computation_event, *computation_stream));
    
    if(left) {
      insist(copy_to_left_stream && copy_to_left_event);
      
      checkCudaErrors(cudaStreamWaitEvent(*copy_to_left_stream, *computation_event, 0));
      
      checkCudaErrors(cudaMemcpyPeerAsync(left->omega_wavepacket_from_right_device, left->device_index(),
					  wp->legendre_psi_dev_(), device_index(),
					  n1*n2*n_Legs*sizeof(double), *copy_to_left_stream));
      
      checkCudaErrors(cudaEventRecord(*copy_to_left_event, *copy_to_left_stream));
    }
    
    if(right) {
      insist(copy_to_right_stream && copy_to_right_event);
      
      checkCudaErrors(cudaStreamWaitEvent(*copy_to_right_stream, *computation_event, 0));
      
      checkCudaErrors(cudaMemcpyPeerAsync(right->omega_wavepacket_from_left_device, right->device_index(),
					  wp->legendre_psi_dev_(), device_index(),
					  n1*n2*n_Legs*sizeof(double), *copy_to_right_stream));
      
      checkCudaErrors(cudaEventRecord(*copy_to_right_event, *copy_to_right_stream));
    }
    
    return;
  }
  
  // multiple omege wavepackets

  int wp_start = 0;
  int wp_end = n_omegas;

  if(left) {
    insist(copy_to_left_stream && copy_to_left_event);
    
    OmegaWavepacket *wp = omega_wavepackets[0];
    wp->forward_legendre_transform();
    
    checkCudaErrors(cudaStreamSynchronize(*computation_stream));
    // checkCudaErrors(cudaEventRecord(*computation_event, *computation_stream));

    //checkCudaErrors(cudaStreamWaitEvent(*copy_to_left_stream, *computation_event, 0));
    
    checkCudaErrors(cudaMemcpyPeerAsync(left->omega_wavepacket_from_right_device, left->device_index(),
					wp->legendre_psi_dev_(), device_index(),
					n1*n2*n_Legs*sizeof(double), *copy_to_left_stream));
    
    checkCudaErrors(cudaEventRecord(*copy_to_left_event, *copy_to_left_stream));
    
    wp_start = 1;
  }

  if(right) {
    insist(copy_to_right_stream && copy_to_right_event);
    
    OmegaWavepacket *wp = omega_wavepackets[n_omegas-1];
    wp->forward_legendre_transform();
    
    // checkCudaErrors(cudaEventRecord(*computation_event, *computation_stream));
    // checkCudaErrors(cudaStreamWaitEvent(*copy_to_right_stream, *computation_event, 0));
    
    checkCudaErrors(cudaStreamSynchronize(*computation_stream));
    
    checkCudaErrors(cudaMemcpyPeerAsync(right->omega_wavepacket_from_left_device, right->device_index(),
					wp->legendre_psi_dev_(), device_index(),
					n1*n2*n_Legs*sizeof(double), *copy_to_right_stream));
    
    checkCudaErrors(cudaEventRecord(*copy_to_right_event, *copy_to_right_stream));
    
    wp_end = n_omegas - 1;
  }
  
  for(int i = wp_start; i < wp_end; i++) 
    omega_wavepackets[i]->forward_legendre_transform();
}

void WavepacketsOnSingleDevice::calculate_T_asym_add_to_T_angle_legendre_psi_dev()
{
  setup_device();

  const OmegaWavepacket *wp = 0;

  for(int i = 0; i < n_omegas; i++) {
    if(i > 0) {
      wp = omega_wavepackets[i-1];
      omega_wavepackets[i]->calculate_T_asym_add_to_T_angle_legendre_psi_dev(wp->legendre_psi_dev_(),
									     wp->omega_());
    }
    
    if(i < n_omegas-1) {
      wp = omega_wavepackets[i+1];
      omega_wavepackets[i]->calculate_T_asym_add_to_T_angle_legendre_psi_dev(wp->legendre_psi_dev_(),
									     wp->omega_());
    }
  }

  if(left) {
    insist(left->copy_to_right_stream && left->copy_to_right_event);
    checkCudaErrors(cudaStreamWaitEvent(*computation_stream, *left->copy_to_right_event, 0));
    wp = omega_wavepackets[0];
    wp->calculate_T_asym_add_to_T_angle_legendre_psi_dev(omega_wavepacket_from_left_device,
							 wp->omega_()-1);
  }
  
  if(right) {
    insist(right->copy_to_left_stream && right->copy_to_left_event);
    checkCudaErrors(cudaStreamWaitEvent(*computation_stream, *right->copy_to_left_event, 0));
    wp = omega_wavepackets[n_omegas-1];
    wp->calculate_T_asym_add_to_T_angle_legendre_psi_dev(omega_wavepacket_from_right_device,
							 wp->omega_()+1);
  }
}

void WavepacketsOnSingleDevice::calculate_H_weighted_psi_dev(const int part)
{
  setup_device();
  
  forward_legendre_transform_and_copy_data_to_neighbour_devices(part);

  checkCudaErrors(cudaStreamSynchronize(*computation_stream));

  for(int i = 0; i < n_omegas; i++)
    omega_wavepackets[i]->calculate_T_bend_T_sym_add_to_T_angle_legendre_psi_dev();

  checkCudaErrors(cudaStreamSynchronize(*computation_stream));

  calculate_T_asym_add_to_T_angle_legendre_psi_dev();
  
  checkCudaErrors(cudaDeviceSynchronize());
  insist(cublasSetStream(cublas_handle, NULL) == CUBLAS_STATUS_SUCCESS);
  
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->calculate_H_weighted_psi_dev();
}

void WavepacketsOnSingleDevice::propagate_with_symplectic_integrator(const int i_step)
{
  setup_device();

  calculate_H_weighted_psi_dev(_RealPart_);

  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->propagate_with_symplectic_integrator(i_step);

#pragma omp barrier
  
  calculate_H_weighted_psi_dev(_ImagPart_);
  
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->propagate_with_symplectic_integrator(i_step);
}

void WavepacketsOnSingleDevice::print()
{
  setup_device();
  _module = 0.0;
  _total_energy = 0.0;
  for(int i = 0; i < n_omegas; i++) {
    _module += omega_wavepackets[i]->wavepacket_module();
    _total_energy += omega_wavepackets[i]->energy();
    
    std::cout << " " << omega_wavepackets[i]->omega_()
	      << " " << omega_wavepackets[i]->wavepacket_module()
	      << " " << omega_wavepackets[i]->energy()
	      << std::endl;
  }
}

void WavepacketsOnSingleDevice::copy_weighted_psi_from_device_to_host()
{
  setup_device();
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->copy_weighted_psi_from_device_to_host();
}

void WavepacketsOnSingleDevice::dump_wavepackets() const
{
  setup_device();
  for(int i = 0; i < n_omegas; i++) 
    omega_wavepackets[i]->dump_wavepacket();
}
