
#ifndef EVOLUTION_AUX_H
#define EVOLUTION_AUX_H

#include "evolutionUtils.h"
#include "cudaMath.h"

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
static __global__ void _psi_time_potential_energy_(double *psi_out, const double *psi_in, const double *pot,
						   const int n)
{
  const int index = threadIdx.x + blockDim.x*blockIdx.x;
  if(index < n)
    psi_out[index] += pot[index]*psi_in[index];
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
    psi_out[index] = psi_in[index]*(kin1[i] + kin2[j]); 
  }
}

#endif /* EVOLUTION_AUX_H */

