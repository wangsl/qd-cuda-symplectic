
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

  setup_streams_and_events();

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
  _CUDA_FREE_(cufft_work_dev);
  _CUDA_FREE_(omega_wavepacket_from_left_device);
  _CUDA_FREE_(omega_wavepacket_from_right_device);
  
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

void WavepacketsOnSingleDevice::setup_cufft_work_dev()
{
  if(!cufft_work_dev) {
    std::cout << " Setup cuFFT work on device: " << current_device_index() << std::endl;
    const int &n1 = MatlabData::r1()->n;
    const int &n2 = MatlabData::r2()->n;
    const int &n_theta = MatlabData::theta()->n;
    checkCudaErrors(cudaMalloc(&cufft_work_dev, (n1/2+1)*n2*n_theta*2*sizeof(double)));
    insist(cufft_work_dev);
  }
}

void WavepacketsOnSingleDevice::setup_omega_wavepackets()
{
  insist(omega_wavepackets.size() == 0);
  
  omega_wavepackets.resize(n_omegas, 0);

  setup_cufft_work_dev();

  for(int i = 0; i < n_omegas; i++) {
    omega_wavepackets[i] = new OmegaWavepacket(i+omega_start, potential_dev,
					       cublas_handle, cufft_plan_D2Z, cufft_plan_Z2D,
					       computation_stream,
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
  if(cudaUtils::n_devices() == 1) return;

  setup_device();
  
  if(left) {
    if(!omega_wavepacket_from_left_device) {
      std::cout << " Setup wavepacket from left on device: " << current_device_index() << std::endl;
      const int &n1 = MatlabData::r1()->n;
      const int &n2 = MatlabData::r2()->n;
      const int &n_theta = MatlabData::theta()->n;
      checkCudaErrors(cudaMalloc(&omega_wavepacket_from_left_device, n1*n2*n_theta*sizeof(double)));
      insist(omega_wavepacket_from_left_device);
      
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
      const int &n1 = MatlabData::r1()->n;
      const int &n2 = MatlabData::r2()->n;
      const int &n_theta = MatlabData::theta()->n;
      checkCudaErrors(cudaMalloc(&omega_wavepacket_from_right_device, n1*n2*n_theta*sizeof(double)));
      insist(omega_wavepacket_from_right_device);
      
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

void WavepacketsOnSingleDevice::setup_streams_and_events()
{ 
  if(computation_stream) return;

  std::cout << " Setup computation stream on event device: " << current_device_index() << std::endl;
  
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

  std::cout << " Destroy streams and events: " << device_index() << std::endl;
  
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
  left = left_; right = right_;
  
  std::cout << " Neighbours on device: " << current_device_index()
	    << ", pointers: " << this << " " << left << " " << right << std::endl;
}

void WavepacketsOnSingleDevice::
forward_legendre_transform_and_copy_data_to_neighbour_devices(const int part)
{ 
  if(cudaUtils::n_devices() == 1) return;
 
  insist(part == _RealPart_ || part == _ImagPart_);
 
  // need to fix the size of omega_wavepacket_from_right_device

  const int &n1 = MatlabData::r1()->n;
  const int &n2 = MatlabData::r2()->n;
  const int n_Legs = MatlabData::wavepacket_parameters()->l_max + 1;
  
  setup_device();
  
  insist(computation_stream);
  insist(cublasSetStream(cublas_handle, *computation_stream) == CUBLAS_STATUS_SUCCESS);

  int wp_start = 0;
  int wp_end = n_omegas;
  
  if(left) {
    insist(copy_to_left_stream && copy_to_left_event);

    OmegaWavepacket *wp = omega_wavepackets[0];
    wp->forward_legendre_transform(part);
    
    checkCudaErrors(cudaEventRecord(*computation_event, *computation_stream));

    checkCudaErrors(cudaStreamWaitEvent(*copy_to_left_stream, *computation_event, 0));
    
    checkCudaErrors(cudaMemcpyPeerAsync(left->omega_wavepacket_from_right_device, left->device_index(),
					wp->legendre_psi_dev_(), device_index(),
					n1*n2*n_Legs*sizeof(double), *copy_to_left_stream));
    
    checkCudaErrors(cudaEventRecord(*copy_to_left_event, *copy_to_left_stream));
    
    wp_start = 1;
  }

  if(right) {
    insist(copy_to_right_stream && copy_to_right_event);
    
    OmegaWavepacket *wp = omega_wavepackets[n_omegas-1];
    wp->forward_legendre_transform(part);

    checkCudaErrors(cudaEventRecord(*computation_event, *computation_stream));

    checkCudaErrors(cudaStreamWaitEvent(*copy_to_right_stream, *computation_event, 0));

    checkCudaErrors(cudaMemcpyPeerAsync(right->omega_wavepacket_from_left_device, right->device_index(),
					wp->legendre_psi_dev_(), device_index(),
					n1*n2*n_Legs*sizeof(double), *copy_to_right_stream));
    
    checkCudaErrors(cudaEventRecord(*copy_to_right_event, *copy_to_right_stream));
    
    wp_end = n_omegas - 1;
  }

  for(int i = wp_start; i < wp_end; i++)
    omega_wavepackets[i]->forward_legendre_transform(part);
}

void WavepacketsOnSingleDevice::test_parallel()
{
  setup_device();
  
  forward_legendre_transform_and_copy_data_to_neighbour_devices(_RealPart_);
  
  for(int i = 0; i < n_omegas; i++)
    omega_wavepackets[i]->backward_legendre_transform(_RealPart_);
  
  checkCudaErrors(cudaDeviceSynchronize());
  
  forward_legendre_transform_and_copy_data_to_neighbour_devices(_ImagPart_);
  
  for(int i = 0; i < n_omegas; i++)
    omega_wavepackets[i]->backward_legendre_transform(_ImagPart_);
  
  checkCudaErrors(cudaDeviceSynchronize());

  for(int i = 0; i < n_omegas; i++) {
    omega_wavepackets[i]->test_parallel();
  }
}

void WavepacketsOnSingleDevice::test_serial()
{
  setup_device();
  std::cout << " Test on device: " << current_device_index() << std::endl;
  for(int i = 0; i < n_omegas; i++)
    std::cout << " " << omega_wavepackets[i]->omega_()
	      << " " << omega_wavepackets[i]->wavepacket_module()
	      << " " << omega_wavepackets[i]->kinetic_energy() 
	      << " " << omega_wavepackets[i]->potential_energy()
	      << std::endl;
}
