
#ifndef EVOLUTION_UTILS_H
#define EVOLUTION_UTILS_H

#include "matlabStructures.h"

#define _RealPart_ 1
#define _ImagPart_ 2

#ifdef __NVCC__

#define _DumpMaxSize_ 1024

namespace EvolutionUtils {
  
  struct RadialCoordinate
  {
    double left;
    double dr;
    double mass;
    double dump[_DumpMaxSize_];
    int n;
  };
  
  inline void copy_radial_coordinate_to_device(const RadialCoordinate &r_dev, 
					       const ::RadialCoordinate *r)
  {
    size_t size = 0;
    checkCudaErrors(cudaGetSymbolSize(&size, r_dev));
    insist(size == sizeof(RadialCoordinate));

    RadialCoordinate r_;
    r_.left = r->left;
    r_.dr = r->dr;
    r_.mass = r->mass;
    r_.n = r->n;
    
    insist(r->n <= _DumpMaxSize_);

    memset(r_.dump, 0, _DumpMaxSize_*sizeof(double));
    memcpy(r_.dump, r->dump, r->n*sizeof(double));
    
    checkCudaErrors(cudaMemcpyToSymbol(r_dev, &r_, sizeof(RadialCoordinate)));
  }
}

extern __constant__ EvolutionUtils::RadialCoordinate r1_dev;
extern __constant__ EvolutionUtils::RadialCoordinate r2_dev;

#endif /* __NVCC__ */

#endif /* EVOLUTION_UTILS_H */


