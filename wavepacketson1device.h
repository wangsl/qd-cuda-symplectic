
#ifndef WAVEPACKETS_ON_SINGLE_DEVICE
#define WAVEPACKETS_ON_SINGLE_DEVICE

#include <cublas_v2.h>
#include <cufft.h>

#include "omegawavepacket.h"
#include "vecbase.h"

class WavepacketsOnSingleDevice
{
public:
  WavepacketsOnSingleDevice(const int device_index,
			    const int omega_start,
			    const int n_omegas);

  ~WavepacketsOnSingleDevice() { destroy_data_on_device(); }

  void test();
  void test_parallel();

private:

  Vec<OmegaWavepacket *> omega_wavepackets;

  int _device_index;
  int omega_start;
  int n_omegas;

  double *potential_dev;

  double *cufft_work_dev;
  double *omega_wavepacket_from_left_device;
  double *omega_wavepacket_from_right_device;

  cublasHandle_t cublas_handle;
  int _has_created_cublas_handle;

  cufftHandle cufft_plan_D2Z;
  cufftHandle cufft_plan_Z2D;
  int _has_cufft_plans;

  int device_index() const { return _device_index; }
  int current_device_index() const;
  void setup_device() const;

  void setup_data_on_device();
  void destroy_data_on_device();

  void setup_constant_memory_on_device();

  void setup_cublas_handle();
  void destroy_cublas_handle();

  void setup_cufft_plans();
  void destroy_cufft_plans();

  void setup_potential_on_device();
  void setup_omega_wavepackets();

  void setup_work_spaces_on_device();
};

#endif /* WAVEPACKETS_ON_SINGLE_DEVICE */
