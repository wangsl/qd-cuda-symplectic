#!/bin/bash

module purge
module load intel/16.0.3
module load gcc/4.8.2
module load cuda/7.5.18
module load matlab/2015b

export LD_PRELOAD=$LD_PRELOAD:$MKL_LIB/libmkl_intel_ilp64.so:$MKL_LIB/libmkl_core.so:$MKL_LIB/libmkl_intel_thread.so:$INTEL_LIB/libiomp5.so

export LD_PRELOAD=$GCC_LIB/libstdc++.so:$LD_PRELOAD

#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
#export CUDA_VISIBLE_DEVICES="0,1,2"

if [ "$1" == "-matlab" ]; then
    taskset -c 0-19 matlab > /dev/null 2>&1 
elif [ "$1" == "-nodesktop" ]; then
    taskset -c 0-19 matlab -nodesktop -r "HO2main; exit" 2>&1 
else
    #suffix=$(echo $CUDA_VISIBLE_DEVICES | sed -e 's/,//g')
    nvprof \
	--print-summary-per-gpu \
	--profile-from-start off \
	--force-overwrite --export-profile /scratch/wang/cuda-profile/profile-HO2.out \
	matlab -nodisplay -r "HO2main; exit" 2>&1 | tee stdout.log
fi

