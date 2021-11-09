#!/usr/bin/env bash

INPUT_PATH=${1}

if [[ $# -lt 1 ]];then
    echo "=== scripts description: linearized the tree data"
    echo "=== useage:
                bash tree_process.sh <INPUT_PATH> [OUTPUT_PATH]"
    echo "===
                INPUT_PATH should include train.tree dev.tree test.tree"
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

# convert to sequence format
mode=s2t
python raw_prepare.py --src_file ${INPUT_PATH}/dev.tree --syn_mode ${mode}
python raw_prepare.py --src_file ${INPUT_PATH}/test.tree --syn_mode ${mode}
python raw_prepare.py --src_file ${INPUT_PATH}/train.tree --syn_mode ${mode}

mode=s2b
python raw_prepare.py --src_file ${INPUT_PATH}/dev.tree --syn_mode ${mode}
python raw_prepare.py --src_file ${INPUT_PATH}/test.tree --syn_mode ${mode}
python raw_prepare.py --src_file ${INPUT_PATH}/train.tree --syn_mode ${mode}

mode=s2s
python raw_prepare.py --src_file ${INPUT_PATH}/dev.tree --syn_mode ${mode}
python raw_prepare.py --src_file ${INPUT_PATH}/test.tree --syn_mode ${mode}
python raw_prepare.py --src_file ${INPUT_PATH}/train.tree --syn_mode ${mode}