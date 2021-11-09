#!/usr/bin/env bash

cd /mnt/cephfs_hl/bytetrans/baoyu.nlp/projects/non_auto_gen/dss_vae

python data_prepare.py --source_dir /mnt/cephfs_hl/bytetrans/baoyu.nlp/data/snli-data/snli --dest_dir /mnt/cephfs_hl/bytetrans/baoyu.nlp/data/bin/snli-s2t --tgt s2t --train_size -1 --vocab_freq_cutoff 2 --max_src_len 30
python data_prepare.py --source_dir /mnt/cephfs_hl/bytetrans/baoyu.nlp/data/snli-data/snli --dest_dir /mnt/cephfs_hl/bytetrans/baoyu.nlp/data/bin/snli-s2b --tgt s2b --train_size -1 --vocab_freq_cutoff 2 --max_src_len 30
python data_prepare.py --source_dir /mnt/cephfs_hl/bytetrans/baoyu.nlp/data/snli-data/snli --dest_dir /mnt/cephfs_hl/bytetrans/baoyu.nlp/data/bin/snli-s2s --tgt s2s --train_size -1 --vocab_freq_cutoff 2 --max_src_len 30