#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd utils
python build.py build_ext --inplace
cd ../
