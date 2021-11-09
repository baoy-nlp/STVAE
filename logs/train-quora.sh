#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=${1} python main.py --mode train \
            --configs configs/data/Quora.yaml configs/model/${2}.yaml configs/train/Quora-${3}.yaml \
            --exp-desc ${4}