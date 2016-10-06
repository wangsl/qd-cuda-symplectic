
#ifndef OMEGA_WAVEPACKET_H
#define OMEGA_WAVEPACKET_H

#include <cublas_v2.h>
#include <cufft.h>

class OmegaWavepacket
{
public:

  OmegaWavepacket(const int &omega,
		  const double * &potential_dev, 
		  cublasHandle_t &cublas_handle,
		  cufftHandle &cufft_plan_D2Z,
		  cufftHandle &cufft_plan_Z2D
		  );

  ~OmegaWavepacket();
  
private:

  double *weighted_psi_real;
  double *weighted_psi_imag;
  
  const int &omega;

  const double * &potential_dev;
  double *weighted_psi_real_dev;
  double *weighted_psi_imag_dev;

  cublasHandle_t &cublas_handle;
  cufftHandle &cufft_plan_D2Z;
  cufftHandle &cufft_plan_Z2D;

  void setup_weighted_psi();
  
  void copy_weighted_psi_from_host_to_device();
  void copy_weighted_psi_from_device_to_host();

};

#endif /* OMEGA_WAVEPACKET_H */
