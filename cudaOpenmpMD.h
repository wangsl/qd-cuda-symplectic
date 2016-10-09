
#ifndef CUDA_OPENMP_MD_H
#define CUDA_OPENMP_MD_H

#include "wavepacketson1device.h"
#include "vecbase.h"

class CUDAOpenmpMD
{
public:

  CUDAOpenmpMD();

  ~CUDAOpenmpMD();
  
  void test();

private:
  
  int _n_devices;
  
  Vec<WavepacketsOnSingleDevice *> wavepackets_on_single_device;

  // General device functions
  int n_devices() const { return _n_devices; }

  void setup_n_devices();
  void devices_synchoronize();
  void devices_memory_usage() const;
  void reset_devices();
  
  void setup_wavepackets_on_single_device();
  void destroy_wavepackets_on_single_device();
};

#endif /* CUDA_OPENMP_MD_H */
