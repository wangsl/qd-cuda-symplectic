
#ifndef CUDA_OPENMP_MD_H
#define CUDA_OPENMP_MD_H

class CUDAOpenmpMD
{
public:

  CUDAOpenmpMD();

  ~CUDAOpenmpMD();
  
  void test();

private:
  
  int _n_devices;

  // General device functions
  int n_devices() const { return _n_devices; }
  void setup_n_devices();
  void devices_synchoronize();
  void devices_memory_usage() const;
  void reset_devices();

};

#endif /* CUDA_OPENMP_MD_H */
