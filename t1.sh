#!/bin/bash

/share/apps/cuda/7.5.18/bin/nvcc -I/share/apps/mpfr/3.1.2/gnu/include -I/share/apps/cuda/7.5.18/include -I/share/apps/gmp/5.1.3/gnu/include -I/share/apps/gcc/4.8.2/include -I/share/apps/intel/16.0.3/compilers_and_libraries_2016.3.210/linux/mkl/include -I/share/apps/cuda/7.5.18/nvvm/include -I/share/apps/mpc/1.0.2/gnu/include -O3 \
    -I/share/apps/cuda/7.5.18/samples/common/inc -I/share/apps/matlab/2015b/extern/include -gencode arch=compute_37,code=sm_37 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -Xcompiler="-fPIC -pthread -fexceptions -m64 -fopenmp" -prec-div=true -prec-sqrt=true -rdc=true -std=c++11 -dc /home/wang/matlab/qd-cuda-symplectic/cudaOpenmpMD.cu

