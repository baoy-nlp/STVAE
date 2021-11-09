#!/usr/bin/env bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd ${THIS_DIR}

data_config=/mnt/cephfs_hl/bytetrans/baoyu.nlp/projects/non_auto_gen/configs/model_configs/${1}.yaml
model_config=/mnt/cephfs_hl/bytetrans/baoyu.nlp/projects/non_auto_gen/configs/model_configs/${2}.yaml

echo "Training Syntax VAE"

python3 dss_main.py --base_config ${data_config} --model_config ${model_config} --mode train --exp_name ${3}

