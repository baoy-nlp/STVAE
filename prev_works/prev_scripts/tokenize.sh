#!/usr/bin/env bash

INPUT_PATH=${1}

if [[ $# -lt 1 ]];then
    echo "=== scripts description: tokenize the source word"
    echo "=== useage:
                bash tokenize.sh <INPUT_PATH> [OUTPUT_PATH]"
    echo "===
                INPUT_PATH should include train.tree.word dev.tree.word test.tree.word"
    exit
fi



if [[ $# -lt 2 ]];then
    OUTPUT_PATH=${INPUT_PATH}
else
    OUTPUT_PATH=${2}
fi

ROOT=/mnt/cephfs_hl/bytetrans/baoyu.nlp/
SCRIPTS_ROOT=${ROOT}/projects/non_auto_gen/dss_vae
cd ${SCRIPTS_ROOT}

python raw_tokenize.py --input_file ${INPUT_PATH}/dev.tree.word --for_parse --is_lower
python raw_tokenize.py --input_file ${INPUT_PATH}/test.tree.word --for_parse --is_lower
python raw_tokenize.py --input_file ${INPUT_PATH}/train.tree.word --for_parse --is_lower