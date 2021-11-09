#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${1}

beam_size=4

if [[ $# -gt 4 ]];then
    beam_size=${5}
fi
cd ..
python3 main.py --data-type raw --mode extract \
    --configs configs/data/Quora-Test.yaml configs/model/${2}.yaml configs/train/Quora/Quora-STVAE.yaml \
    --exp-desc ${3} --eval-func template