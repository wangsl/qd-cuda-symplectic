/* $Id$ */

#include "matlabStructures.h"
#include "matlabUtils.h"
#include "matlabArray.h"
#include "matlabData.h"

void remove_matlab_script_extension(char *script, const char *extension)
{
  insist(script && extension);
  const int len = strlen(script) - strlen(extension);
  if(!strcmp((const char *) script+len, extension)) 
    ((char *) script)[len] = '\0';
}

RadialCoordinate::RadialCoordinate(const mxArray *mx) :
  mx(mx),
  n(*(int *) mxGetData(mx, "n", _mxInt32_)),
  left(*(double *) mxGetData(mx, "left")),
  dr(*(double *) mxGetData(mx, "dr")),
  mass(*(double *) mxGetData(mx, "mass"))
{
  double *d = (double *) mxGetData(mx, "dump");
  if(d) dump = RVec(n, d);
} 

AngleCoordinate::AngleCoordinate(const mxArray *mx) :
  mx(mx),
  n(*(int *) mxGetData(mx, "n", _mxInt32_))
{
  x = RVec(n, (double *) mxGetData(mx, "x"));
  w = RVec(n, (double *) mxGetData(mx, "w"));
}
  
EvolutionTime::EvolutionTime(const mxArray *mx) :
  mx(mx),
  total_steps(*(int *) mxGetData(mx, "total_steps", _mxInt32_)),
  steps(*(int *) mxGetData(mx, "steps", _mxInt32_)),
  time_step(*(double *) mxGetData(mx, "time_step"))
{ }

Options::Options(const mxArray *mx) :
  mx(mx),
  wave_to_matlab(0),

  steps_to_copy_psi_from_device_to_host					\
  (*(int *) mxGetData(mx, "steps_to_copy_psi_from_device_to_host", _mxInt32_)),
  
  potential_cutoff(*(double *) mxGetData(mx, "potential_cutoff")),
  
  calculate_reaction_probabilities
  (*(int *) mxGetData(mx, "calculate_reaction_probabilities", _mxInt32_))
{
  wave_to_matlab = mxGetString(mx, "wave_to_matlab");
  if(wave_to_matlab)
    remove_matlab_script_extension(wave_to_matlab, ".m");
}

Options::~Options()
{
  mx = 0; 
  if(wave_to_matlab) { mxFree(wave_to_matlab); wave_to_matlab = 0; }
}

CRPParameters::CRPParameters(const mxArray *mx) :
  mx(mx),
  n_dividing_surface(*(int *) mxGetData(mx, "n_dividing_surface", _mxInt32_)),
  n_gradient_points(*(int *) mxGetData(mx, "n_gradient_points", _mxInt32_)),
  n_energies(*(int *) mxGetData(mx, "n_energies", _mxInt32_))
{
  energies = RVec(n_energies, (double *) mxGetData(mx, "energies"));
  eta_sq = RVec(n_energies, (double *) mxGetData(mx, "eta_sq"));
  CRP = RVec(n_energies, (double *) mxGetData(mx, "CRP"));
}

WavepacketParameters::WavepacketParameters(const mxArray *mx) :
  mx(mx),
  J(*(int *) mxGetData(mx, "J", _mxInt32_)),
  parity(*(int *) mxGetData(mx, "parity", _mxInt32_)),
  l_max(*(int *) mxGetData(mx, "lMax", _mxInt32_)),
  omega_min(*(int *) mxGetData(mx, "OmegaMin", _mxInt32_)),
  omega_max(*(int *) mxGetData(mx, "OmegaMax", _mxInt32_))
{ 
  insist(omega_min == 0 || omega_min == 1);
  setup_weighted_wavepackets();
  setup_weighted_associated_legendres();
}
  
void WavepacketParameters::setup_weighted_wavepackets()
{ 
  const mxArray *wps_ptr = mxGetField(mx, 0, "weighted_wavepackets");
  insist(wps_ptr);
  
  const MatlabComplexArray wps(wps_ptr);
  
  const size_t *dims = wps.dims();

  insist(wps.n_dims() <= 4);
  
  const int n1 = dims[0];
  const int n2 = dims[1];
  const int n3 = wps.n_dims() > 2 ? dims[2] : 1;
  const int n4 = wps.n_dims() > 3 ? dims[3] : 1;

  std::cout << " Wavepacket size: " << n1 << " " << n2 << " " << n3 << " " << n4 << std::endl;

  if(MatlabData::r1()) insist(n1 == MatlabData::r1()->n);
  if(MatlabData::r2()) insist(n2 == MatlabData::r2()->n);
  if(MatlabData::theta()) insist(n3 == MatlabData::theta()->n);
  insist(n4 == omega_max - omega_min + 1);
  
  const double *wps_real = wps.real();
  const double *wps_imag = wps.imag();

  weighted_wavepackets_real.resize(n4);
  weighted_wavepackets_imag.resize(n4);
  
  for(int i = 0; i < n4; i++) {
    weighted_wavepackets_real[i] = RVec(n1*n2*n3, const_cast<double *>(wps_real+i*n1*n2*n3));
    weighted_wavepackets_imag[i] = RVec(n1*n2*n3, const_cast<double *>(wps_imag+i*n1*n2*n3));
  }
}

void WavepacketParameters::setup_weighted_associated_legendres()
{ 
  const mxArray *ass_legs_ptr = mxGetField(mx, 0, "weighted_associated_legendres");
  
  if(!ass_legs_ptr) return;

  const MatlabArray<double> ass_legs(ass_legs_ptr);
  
  insist(ass_legs.n_dims() == 2 || ass_legs.n_dims() == 3);
  
  const size_t *dims = ass_legs.dims();
  
  const int n1 = dims[0];
  const int n2 = dims[1];
  const int n3 = ass_legs.n_dims() == 3 ? dims[2] : 1;

  if(MatlabData::theta()) {
    insist(n1 == MatlabData::theta()->n);
    insist(MatlabData::theta()->n > l_max+1);
  }
  
  if(omega_min == 0) 
    insist(n2 == l_max+1 && n3 == omega_max+1);
  else if(omega_min == 1)
    insist(n2 == l_max && n3 == omega_max);
  
  std::cout << " Weighted associated Legendres size: " << n1 << " " << n2 << " " << n3 << std::endl;

  weighted_associated_legendres.resize(n3);

  const double *p = ass_legs.data();
  for(int i = 0; i < n3; i++) {
    weighted_associated_legendres[i] = RMat(n1, n2-i, const_cast<double *>(p+i*n1));
    p += n1*n2;
  }
}
