#!/usr/bin/env bash
source activate dss-vae
export CUDA_VISIBLE_DEVICES=${1}

echo ${1} "indicate the index of GPU devices"
echo ${2} "indicate the running model"
echo ${3} "indicate the training configuration "
echo ${4} "is the name of experiments"

cd ..
python main.py --dataset PTB --mode test --configs configs/data/PTB.yaml configs/acl/${2}-ptb.yaml configs/train/PTB/${3}.yaml --exp-desc ${4} --eval-func reconstruct