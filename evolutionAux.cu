
#ifndef EVOLUTION_AUX_H
#define EVOLUTION_AUX_H

#include "evolutionUtils.h"
#include "cudaMath.h"
#include "coriolisUtils.h"

#ifdef printf
#undef printf
#endif

static __global__ void _print_constant_memory_()
{
  printf(" %f %f %f %d\n", r1_dev.left, r1_dev.dr, r1_dev.mass, r1_dev.n);
  printf(" %f %f %f %d\n", r2_dev.left, r2_dev.dr, r2_dev.mass, r2_dev.n);
  for(int i = 0; i < 500; i+=10) 
    printf("%d %18.15f %18.15f\n", i+1, r1_dev.dump[i], r2_dev.dump[i]);
}

static __global__ void _psi_times_kinetic_energy_(Complex *psi_out, const Complex *psi_in, 
						  const int n1, const int n2, const int n_theta)
{
  extern __shared__ double kinetic_data[];
  
  double *kin1 = (double *) kinetic_data;
  double *kin2 = &kin1[n1/2+1];
  
  cudaMath::setup_kinetic_energy_for_fft_nonnegative(kin1, r1_dev.n, r1_dev.n*r1_dev.dr, r1_dev.mass);
  cudaMath::setup_kinetic_energy_for_fft(kin2, r2_dev.n, r2_dev.n*r2_dev.dr, r2_dev.mass);

  __syncthreads();
  
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < (n1/2+1)*n2*n_theta) {
    int i = -1; int j = -1; int k = -1;
    cudaMath::index_2_ijk(index, n1/2+1, n2, n_theta, i, j, k);
    psi_out[index] = (kin1[i] + kin2[j])*psi_in[index];
  }
}

static __global__ void _add_T_radial_weighted_psi_to_H_weighted_psi_(double *HPsi, const double *TRadPsi,
								     const int n1, const int n2, 
								     const int n_theta)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < (n1/2+1)*2*n2*n_theta) {
    int i = -1; int j = -1; int k = -1;
    cudaMath::index_2_ijk(index, (n1/2+1)*2, n2, n_theta, i, j, k);
    if(i < n1) {
      const int index2 = cudaMath::ijk_2_index(n1, n2, n_theta, i, j, k);
      HPsi[index2] += TRadPsi[index]/(n1*n2);
    }
  }
}

static __global__ void _add_T_radial_weighted_psi_to_H_weighted_psi_2_(double *HPsi,
								       const double *TRadPsi,
								       const int n1, const int n2, 
								       const int n_theta)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n1*n2*n_theta) 
    HPsi[index] += TRadPsi[index]/(n1*n2);
}

static __global__ void _add_potential_weighted_psi_to_H_weighted_psi_(double *HPsi, const double *psi,
								      const double *pot, const int n)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n) 
    HPsi[index] += pot[index]*psi[index];
}

static __global__ void _add_T_bend_T_sym_to_T_angle_legendre_psi_dev_(double *TangPsi, const double *psi,
								      const int n1, const int n2, 
								      const int nLegs,
								      const int J, const int omega)
{
  extern __shared__ double rotational_moments[];

  double *I1 = rotational_moments;
  double *I2 = &I1[n1];
  double &Tsym = I2[n2];

  cudaMath::setup_moments_of_inertia(I1, r1_dev.n, r1_dev.left, r1_dev.dr, r1_dev.mass);
  cudaMath::setup_moments_of_inertia(I2, r2_dev.n, r2_dev.left, r2_dev.dr, r2_dev.mass);

  if(threadIdx.x == 0) Tsym = J*(J+1) - 2*omega*omega;

  __syncthreads();
  
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n1*n2*nLegs) {
    int i = -1; int j = -1; int l = -1;
    cudaMath::index_2_ijk(index, n1, n2, nLegs, i, j, l);
    TangPsi[index] += ((I1[i]+I2[j])*l*(l+1) + I1[i]*Tsym)*psi[index];
  }
}

static __global__ void _add_T_asym_to_T_angle_legendre_psi_dev_(double *TangPsi, const double *psi,
								const int n1, const int n2, 
								const int nLegs,
								const int J,
								const int Omega, const int Omega1)
{
  extern __shared__ double I1[];
  
  cudaMath::setup_moments_of_inertia(I1, r1_dev.n, r1_dev.left, r1_dev.dr, r1_dev.mass);

  __syncthreads();
  
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n1*n2*nLegs) {
    int i = -1; int j = -1; int l = -1;
    cudaMath::index_2_ijk(index, n1, n2, nLegs, i, j, l);
    const double c = coriolisUtils::coriolis(J, l, Omega, Omega1);
    TangPsi[index] += I1[i]*c*psi[index];
  }
}

static __global__ void _daxpy_(double *y, const double *x, const double alpha, const double beta,
			       const int n)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n)
    y[index] = alpha*x[index] + beta*y[index];
}


#endif /* EVOLUTION_AUX_H */

