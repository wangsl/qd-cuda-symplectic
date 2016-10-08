
#ifndef OMEGA_WAVEPACKET_H
#define OMEGA_WAVEPACKET_H

#include <cublas_v2.h>
#include <cufft.h>

class OmegaWavepacket
{
public:

  OmegaWavepacket(int omega,
		  const double *potential_dev, 
		  cublasHandle_t &cublas_handle,
		  cufftHandle &cufft_plan_D2Z,
		  cufftHandle &cufft_plan_Z2D
		  );

  ~OmegaWavepacket();

  void calculate_wavepacket_module();

  double wavepacket_module() const { return _wavepacket_module_real + _wavepacket_module_imag; }
  
private:

  double *weighted_psi;

  double *weighted_psi_real;
  double *weighted_psi_imag;

  const int omega;

  const double *potential_dev;
  double *weighted_psi_dev;
  double *weighted_psi_real_dev;
  double *weighted_psi_imag_dev;
  double *weighted_associated_legendres_dev;

  cublasHandle_t &cublas_handle;
  cufftHandle &cufft_plan_D2Z;
  cufftHandle &cufft_plan_Z2D;

  double _wavepacket_module_real;
  double _wavepacket_module_imag;

  void setup_weighted_psi();
  void copy_weighted_psi_from_host_to_device();
  void copy_weighted_psi_from_device_to_host();

  void copy_weighted_associated_legendres_to_device();

  void _calculate_wavepacket_module();
};

#endif /* OMEGA_WAVEPACKET_H */
