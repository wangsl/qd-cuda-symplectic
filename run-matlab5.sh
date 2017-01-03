#!/bin/bash

module purge
module load intel/16.0.3
module load gcc/4.8.2
module load cuda/7.5.18
module load matlab/2015b

export LD_PRELOAD=$LD_PRELOAD:$MKL_LIB/libmkl_intel_ilp64.so:$MKL_LIB/libmkl_core.so:$MKL_LIB/libmkl_intel_thread.so:$INTEL_LIB/libiomp5.so

export LD_PRELOAD=$GCC_LIB/libstdc++.so:$LD_PRELOAD

#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
#export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES="4"

#export OMP_NUM_THREADS=1
#export CUDA_LAUNCH_BLOCKING=1

# for((i=0; i<100; i++)) { taskset -c 0-19 matlab -nodisplay -r "FH2main; exit" } 2>&1 | tee stdout.log; exit

if [ "$1" == "-matlab" ]; then
    taskset -c 0-19 matlab > /dev/null 2>&1 
elif [ "$1" == "-nodesktop" ]; then
    taskset -c 0-19 matlab -nodesktop -r "HO2main; exit" 2>&1 
else
    #export CUDA_VISIBLE_DEVICES="0"; matlab -nodisplay -r "HO2main; exit" > HO2main-1.log 2>&1 &
    
    #export CUDA_VISIBLE_DEVICES="1,2"; matlab -nodisplay -r "HO2main; exit" > HO2main-2.log 2>&1 &
    
    #export CUDA_VISIBLE_DEVICES="3,4,5,6"; matlab -nodisplay -r "HO2main; exit" > HO2main-3.log 2>&1 &

    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"; matlab -nodisplay -r "HO2main; exit" > HO2main-3.log 2>&1 &
    
    wait
    
    exit

    suffix=$(echo $CUDA_VISIBLE_DEVICES | sed -e 's/,//g')
    nvprof \
	--devices 0,1 \
	--print-summary-per-gpu \
	--profile-from-start off \
	--force-overwrite --export-profile /scratch/wang/cuda-profile/profile-$suffix.out \
	matlab -nodisplay -r "FH2main; exit" > stdout.log 2>&1 &
fi

