
#include <iostream>
#include <mex.h>
#include "matlabUtils.h"

#include "matlabArray.h"
#include "matlabStructures.h"
#include "matlabData.h"
//#include "complex.h"

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

  // std::cout << *MatlabData::r1() << std::endl;

  mxPtr = mxGetField(prhs[0], 0, "r2");
  insist(mxPtr);
  MatlabData::r2(new RadialCoordinate(mxPtr));
  
  // std::cout << *MatlabData::r2() << std::endl;

  mxPtr = mxGetField(prhs[0], 0, "theta");
  insist(mxPtr);
  MatlabData::theta(new AngleCoordinate(mxPtr));
  
  // std::cout << *MatlabData::theta() << std::endl;

  mxPtr = mxGetField(prhs[0], 0, "potential");
  insist(mxPtr);
  MatlabData::potential(MatlabArray<double>(mxPtr).data());

  mxPtr = mxGetField(prhs[0], 0, "time");
  insist(mxPtr);
  MatlabData::time(new EvolutionTime(mxPtr));

  std::cout << *MatlabData::time() << std::endl;

  mxPtr = mxGetField(prhs[0], 0, "options");
  insist(mxPtr);
  MatlabData::options(new Options(mxPtr));

  std::cout << *MatlabData::options() << std::endl;

  mxPtr = mxGetField(prhs[0], 0, "wavepacket_parameters");
  insist(mxPtr);
  MatlabData::wavepacket_parameters(new WavepacketParameters(mxPtr));
  
  std::cout << *MatlabData::wavepacket_parameters() << std::endl;

  std::cout.flush();
}
