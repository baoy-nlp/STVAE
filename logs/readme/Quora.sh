Experiments for Quora Question Pair
---
#### Download


#### Build the tree dataset
```bash
python preprocess.py --configs configs/data/PTB.yaml --mode scratch
```

**Note**: PTB.yaml indicates the location of the data and the way to preprocess the data. Before performing the experiment, you need to ensure that the PTB data has been placed in the designated folder.


#### Training
```bash
# training the DSS-VAE model with:
python main.py --configs configs/data/PTB.yaml configs/model/PTB-dvae.yaml configs/run/PTB-dvae.yaml --mode train --exp-desc rep-ori
```

#### Configuration

TRAIN
```bash
# train vae
--configs configs/data/ptb-208.yaml configs/run/ptb-vae.yaml configs/model/vae.yaml --mode train --exp-desc baseline3
# train dvae
--configs configs/data/ptb-208.yaml configs/run/ptb-dvae.yaml configs/model/dss-vae.yaml --mode train --exp-desc plain
# train dgvae
--configs configs/data/Quora-GEN.yaml configs/train/Quora-DGVAE.yaml configs/model/dss-gmm-vae.yaml --mode train --exp-desc base-BLEU

# train syntax-cvae
--configs configs/data/Quora-GEN.yaml configs/train/Quora-SCVAE.yaml configs/model/syntax-cvae.yaml --mode train --exp-desc 0.5-1.0-1.0-0.1-0.1-0.1-0.1-1.0-1.0-BLEU
--configs configs/data/Quora-GEN.yaml configs/train/Quora-SCVAE1.yaml configs/model/syntax-cvae.yaml --mode train --exp-desc 0.1-1.0-1.0-0.1-0.1-0.1-0.1-1.0-1.0-BLEU

# train syntax-tvae
--configs configs/data/Quora-GEN.yaml configs/train/Quora-STVAE.yaml configs/model/syntax-tvae.yaml --mode train --exp-desc 0.5-1.0-1.0-0.1-0.1-0.1-0.1-1.0-1.0-BLEU
--configs configs/data/Quora-GEN.yaml configs/train/Quora-STVAE1.yaml configs/model/syntax-tvae.yaml --mode train --exp-desc 0.1-1.0-1.0-0.1-0.1-0.1-0.1-1.0-1.0-BLEU


```

TEST
```bash
# dvae-reconstruct
--configs configs/data/Quora-Paraphrase.yaml configs/model/dss-gmm-vae.yaml configs/train/Quora-DVAE.yaml --mode test --eval-func reconstruct --exp-desc base-BLEU
# dvae-transfer
--configs configs/data/Quora-Paraphrase.yaml configs/model/dss-vae.yaml configs/train/Quora-DVAE.yaml --mode test --exp-desc base-ELBO --eval-func transfer --data-type raw
# dvae-extract-train
--configs configs/data/Quora-Paraphrase.yaml configs/model/dss-vae.yaml configs/train/Quora-DVAE.yaml --mode extract --exp-desc base-ELBO --data-type bin --eval-dataset train
# dvae-extract-valid
--configs configs/data/Quora-Paraphrase.yaml configs/model/dss-vae.yaml configs/train/Quora-DVAE.yaml --mode extract --exp-desc base-ELBO --data-type raw --eval-dataset valid
# dvae-extract-test
--configs configs/data/Quora-Paraphrase.yaml configs/model/dss-vae.yaml configs/train/Quora-DVAE.yaml --mode extract --exp-desc base-ELBO --data-type raw --eval-dataset test
# dvae-search-valid
--configs configs/data/Quora-Paraphrase.yaml configs/model/dss-vae.yaml configs/train/Quora-DVAE.yaml --mode search --search-func generate --eval-topk 200 --eval-m 20 --exp-desc base-ELBO --data-type raw --eval-dataset valid --eval-bs 50
# dvae-paraphrase
--configs configs/data/Quora-Paraphrase.yaml configs/model/dss-vae.yaml configs/train/Quora-DVAE.yaml --mode test --exp-desc base-ELBO --eval-func paraphrase --data-type raw
# dvae-reconstruct
--configs configs/data/Quora-GEN.yaml configs/model/dss-gmm-vae.yaml configs/train/Quora-DGVAE.yaml --mode test --exp-desc n_20-BLEU --eval-func reconstruct
# dvae-transfer
--configs configs/data/Quora-Paraphrase.yaml configs/model/dss-gmm-vae.yaml configs/train/Quora-DGVAE.yaml --mode test --exp-desc base-BLEU --eval-func transfer --data-type raw
# dvae-paraphrase
--configs configs/data/Quora-Paraphrase.yaml configs/model/dss-gmm-vae.yaml configs/train/Quora-DGVAE.yaml --mode test --exp-desc n_20-BLEU --eval-func paraphrase --data-type raw
# dvae-meta-paraphrase
--configs configs/data/Quora-Paraphrase.yaml configs/model/dss-gmm-vae.yaml configs/train/Quora-DGVAE.yaml --mode test --exp-desc n_100-BLEU --eval-func meta-paraphrase --data-type raw
```