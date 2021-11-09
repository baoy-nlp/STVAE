Experiments for PTB
---
#### Build the PTB tree dataset
```bash
python preprocess.py --configs configs/data/PTB.yaml --mode scratch
```

*Note*: PTB.yaml indicates the location of the data and the way to preprocess the data. Before performing the experiment, you need to ensure that the PTB data has been placed in the designated folder.


#### Training
```bash
# training the DSS-VAE model with:
python main.py --configs configs/data/PTB.yaml configs/model/PTB-dvae.yaml configs/run/PTB-dvae.yaml --mode train --exp-desc rep-ori
```
