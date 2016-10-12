
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
		  cufftHandle &cufft_plan_Z2D,
		  cudaStream_t * &computation_stream,
		  double * &cufft_work_dev
		  );

  ~OmegaWavepacket();

  void calculate_wavepacket_module();

  double wavepacket_module() const 
  { return _wavepacket_module_from_real + _wavepacket_module_from_imag; }

  double kinetic_energy() const
  { return _kinetic_energy_from_real + _kinetic_energy_from_imag; }

  double potential_energy() const
  { return _potential_energy_from_real + _potential_energy_from_imag; }

  void test_parallel();

  void forward_legendre_transform(const int part);
  void backward_legendre_transform(const int part);

  const double *legendre_psi_dev_() const { return legendre_psi_dev; }
  const int &omega_() const { return omega; }
  
private:

  double *weighted_psi_real;
  double *weighted_psi_imag;

  const int omega;

  const double *potential_dev;
  
  double *weighted_psi_dev;
  double *weighted_psi_real_dev;
  double *weighted_psi_imag_dev;
  double *weighted_associated_legendres_dev;
						
  double *H_weighted_psi_dev;
  double *H_weighted_legendre_psi_dev;

  double *legendre_psi_dev;

  double * &cufft_work_dev;

  cublasHandle_t &cublas_handle;
  cufftHandle &cufft_plan_D2Z;
  cufftHandle &cufft_plan_Z2D;
  cudaStream_t * &computation_stream;

  double _wavepacket_module_from_real;
  double _wavepacket_module_from_imag;
  
  double _kinetic_energy_from_real;
  double _kinetic_energy_from_imag;

  double _potential_energy_from_real;
  double _potential_energy_from_imag;

  void setup_weighted_psi();
  void copy_weighted_psi_from_host_to_device();
  void copy_weighted_psi_from_device_to_host();

  void copy_weighted_associated_legendres_from_host_to_device();
  
  void setup_legendre_psi_dev();

  //void _calculate_wavepacket_module();

  void _calculate_kinetic_on_weighted_psi();
  void _calculate_potential_on_weighted_psi();

  void cufft_D2Z_for_weighted_psi();
  void cufft_Z2D_for_weighted_psi();

  void forward_legendre_transform();
  void backward_legendre_transform();

  double dot_product_with_volume_element(const double *x_dev, const double *y_dev) const;
};

#endif /* OMEGA_WAVEPACKET_H */
