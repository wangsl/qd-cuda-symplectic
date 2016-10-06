
#include <iostream>
#include <mex.h>
#include "matlabUtils.h"

#include "matlabArray.h"
#include "matlabStructures.h"
#include "matlabData.h"
#include "cudaOpenmpMD.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  const int np = std::cout.precision();
  std::cout.precision(14);

  std::cout << "\n Quantum Dynamics Time Evolution with CUDA\n" << std::endl;

  insist(nrhs == 1);

  mxArray *mxPtr = 0;

  mxPtr = mxGetField(prhs[0], 0, "r1");
  insist(mxPtr);
  MatlabData::r1(new RadialCoordinate(mxPtr));

  mxPtr = mxGetField(prhs[0], 0, "r2");
  insist(mxPtr);
  MatlabData::r2(new RadialCoordinate(mxPtr));
  
  mxPtr = mxGetField(prhs[0], 0, "theta");
  insist(mxPtr);
  MatlabData::theta(new AngleCoordinate(mxPtr));
  
  mxPtr = mxGetField(prhs[0], 0, "potential");
  insist(mxPtr);
  MatlabData::potential(MatlabArray<double>(mxPtr).data());

  mxPtr = mxGetField(prhs[0], 0, "time");
  insist(mxPtr);
  MatlabData::time(new EvolutionTime(mxPtr));

  mxPtr = mxGetField(prhs[0], 0, "options");
  insist(mxPtr);
  MatlabData::options(new Options(mxPtr));

  mxPtr = mxGetField(prhs[0], 0, "wavepacket_parameters");
  insist(mxPtr);
  MatlabData::wavepacket_parameters(new WavepacketParameters(mxPtr));

  CUDAOpenmpMD evolCUDA;
  evolCUDA.test();
  
  std::cout.flush();
  std::cout.precision(np);
}
