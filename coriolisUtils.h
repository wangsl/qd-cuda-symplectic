
#ifndef CORIOLIS_UTILS_H
#define CORIOLIS_UTILS_H

namespace coriolisUtils {

  __device__ __host__ inline int kronecker_delta(const int a, const int b)
  { return a == b ? 1 : 0; }

  // sign should be +1 or -1
  __device__ __host__ inline double lambda(const int a, const int b, const int sign)
  { return a >= b ? sqrt(double(a*(a+1)-b*(b+sign))) : 0; }
  
  __device__ __host__ inline double coriolis(const int J, const int l, const int Omega, const int Omega1)
  {
    if(Omega1 == Omega+1) 
      return sqrt(double(1 + kronecker_delta(Omega, 0)))*lambda(J, Omega, 1)*lambda(l, Omega, 1);
    else if(Omega1 == Omega-1) 
      return sqrt(double(1 + kronecker_delta(Omega, 1)))*lambda(J, Omega, -1)*lambda(l, Omega, -1);
    else
      return 0;
  }
};

#endif /* CORIOLIS_UTILS_H */
