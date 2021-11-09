#!/usr/bin/env bash

beam_size=4

if [[ $# -gt 4 ]];then
    beam_size=${5}
fi


CUDA_VISIBLE_DEVICES=${1} python3 main.py --data-type raw --mode test  --eval-func paraphrase \
        --configs configs/data/Quora-Test.yaml configs/model/${2}.yaml configs/train/Quora/STVAE.yaml \
        --exp-desc ${3}--sample-mode ${4} --sample-size ${beam_size}