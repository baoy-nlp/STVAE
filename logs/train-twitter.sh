#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=${1} python main.py --dataset Twitter --mode train \
                --configs configs/data/Twitter-UP.yaml  configs/model/${2}.yaml configs/train/Twitter-${3}.yaml \
                --exp-desc ${4}