#! /bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

export TF_USE_DEEP_CONV2D=0
export TF_CUDNN_USE_AUTOTUNE=0
export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=4096  # 4GB working space

# export CUDA_VISIBLE_DEVICES=GPU-53b9  # This is GPU0.  GPU1: GPU-6f90
# export WORKON_HOME=~/envs
# source /usr/local/bin/virtualenvwrapper.sh
# workon tf0.9
python paleo/profiler.py "$@"
