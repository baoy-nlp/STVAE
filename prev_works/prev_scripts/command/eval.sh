#!/usr/bin/env bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd ${THIS_DIR}

DEV_MODEL_DIR=${1}
DATA_DIR=${2}

python3 dss_main.py --mode test \
--load_from ${DEV_MODEL_DIR} \
--test_dir ${DATA_DIR}/${1}
