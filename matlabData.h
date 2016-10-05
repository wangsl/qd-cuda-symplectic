
#ifndef MATLAB_DATA_H
#define MATLAB_DATA_H

#include "matlabStructures.h"

namespace MatlabData
{
  const RadialCoordinate *r1();
  void r1(const RadialCoordinate *r);
  
  const RadialCoordinate *r2();
  void r2(const RadialCoordinate *r);

  const AngleCoordinate *theta();
  void theta(const AngleCoordinate *th);

  const double *potential();
  void potential(const double *p);

  inline int dump_wavepacket() { return r1()->dump.size() && r2()->dump.size() ? 1 : 0; }
  
  EvolutionTime *time();
  void time(EvolutionTime *t);
  
  const Options *options();
  void options(const Options *op);

  WavepacketParameters *wavepacket_parameters();
  void wavepacket_parameters(WavepacketParameters *params);

#if 0
  OmegaStates *omega_states();
  void omega_states(OmegaStates *om);
#endif
};

#endif /* MATLAB_DATA_H */
