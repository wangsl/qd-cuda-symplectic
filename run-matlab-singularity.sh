#!/bin/bash

module purge
module load matlab/2017a

export CUDA_VISIBLE_DEVICES=0,1,2

/share/apps/singularity/2.3.1/bin/singularity  \
    exec --bind /run,$(which nvidia-smi) --nv \
    /beegfs/work/public/wang/images/centos-7.3.1611.img vncserver



