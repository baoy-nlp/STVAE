Experiments for Twitter
---

### Data Prepare

Get Raw Data
>clone from Qian et.al.

Tokenize the raw data with tools/tokenize.py
>python tokenize.py --input [] --output [] --syntax --lower

Parse the tokenized data with ZPar_0.7.5
>/dist/zpar.en -oc english-models input_file output_file

Build the unsupervised setting of Twitter with preprocess.py
>python preprocess.py --configs configs/data/Twitter-UP.yaml --mode scratch

### Train

Train Syntax-TVAE
```bash
python main.py --dataset Twitter --mode train --configs configs/data/Twitter-UP.yaml  configs/model/syntax-tvae2-h0.5.yaml configs/train/Twitter-STVAE.yaml --exp-desc EXP0
```
