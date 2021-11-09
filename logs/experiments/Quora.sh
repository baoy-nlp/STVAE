#!/usr/bin/env bash
Experiments for Quora Question Pair
---

RECONSTRUCT
```bash
# dvae-reconstruct
--configs configs/data/Quora-Paraphrase.yaml configs/model/dss-gmm-vae.yaml configs/train/Quora-DVAE.yaml --mode test --eval-func reconstruct --exp-desc base-BLEU
```

EXTRACT
```bash
# dvae-extract-train
--configs configs/data/Quora-Paraphrase.yaml configs/model/dss-vae.yaml configs/train/Quora-DVAE.yaml --mode extract --exp-desc base-ELBO --data-type bin --eval-dataset train
# dvae-extract-valid
--configs configs/data/Quora-Paraphrase.yaml configs/model/dss-vae.yaml configs/train/Quora-DVAE.yaml --mode extract --exp-desc base-ELBO --data-type raw --eval-dataset valid
# dvae-extract-test
--configs configs/data/Quora-Paraphrase.yaml configs/model/dss-vae.yaml configs/train/Quora-DVAE.yaml --mode extract --exp-desc base-ELBO --data-type raw --eval-dataset test
```

TEST
```bash
# dvae-transfer
--configs configs/data/Quora-Paraphrase.yaml configs/model/dss-vae.yaml configs/train/Quora-DVAE.yaml --mode test --exp-desc base-ELBO --eval-func transfer --data-type raw

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

TRAIN
```bash
# vae
--configs configs/data/ptb-208.yaml configs/run/ptb-vae.yaml configs/model/vae.yaml --mode train --exp-desc baseline3

# dvae
--configs configs/data/ptb-208.yaml configs/run/ptb-dvae.yaml configs/model/dss-vae.yaml --mode train --exp-desc plain

# dgvae
--configs configs/data/Quora-GEN.yaml configs/train/Quora-DGVAE.yaml configs/model/dss-gmm-vae.yaml --mode train --exp-desc base-BLEU

# syntax-cvae
--configs configs/data/Quora-GEN.yaml configs/train/Quora-SCVAE.yaml configs/model/syntax-cvae.yaml --mode train --exp-desc 0.5-1.0-1.0-0.1-0.1-0.1-0.1-1.0-1.0-BLEU
--configs configs/data/Quora-GEN.yaml configs/train/Quora-SCVAE1.yaml configs/model/syntax-cvae.yaml --mode train --exp-desc 0.1-1.0-1.0-0.1-0.1-0.1-0.1-1.0-1.0-BLEU
--configs configs/data/Quora-GEN.yaml configs/train/Quora-SCVAE2.yaml configs/model/syntax-cvae.yaml --mode train --exp-desc 0.5-1.0-5.0-0.5-0.5-0.1-0.1-2.0-1.0-BLEU
--configs configs/data/Quora-GEN.yaml configs/train/Quora-SCVAE3.yaml configs/model/syntax-cvae.yaml --mode train --exp-desc 0.5-1.0-5.0-0.5-0.5-0.1-0.1-1.0-2.0-BLEU

# syntax-tvae
--configs configs/data/Quora-GEN.yaml configs/train/Quora-STVAE.yaml configs/model/syntax-tvae.yaml --mode train --exp-desc 0.5-1.0-1.0-0.1-0.1-0.1-0.1-1.0-1.0-BLEU
--configs configs/data/Quora-GEN.yaml configs/train/Quora-STVAE1.yaml configs/model/syntax-tvae.yaml --mode train --exp-desc 0.1-1.0-1.0-0.1-0.1-0.1-0.1-1.0-1.0-BLEU
--configs configs/data/Quora-GEN.yaml configs/train/Quora-STVAE4.yaml configs/model/syntax-tvae.yaml --mode train --exp-desc 0.5-1.0-5.0-0.5-0.5-0.1-0.1-2.0-1.0-BLEU
--configs configs/data/Quora-GEN.yaml configs/train/Quora-STVAE5.yaml configs/model/syntax-tvae-h0.5.yaml --mode train --exp-desc 0.5-1.0-1.0-0.1-0.1-0.1-0.1-2.0-1.0-BLEU-h0.5 # 5
--configs configs/data/Quora-GEN.yaml configs/train/Quora-STVAE6.yaml configs/model/syntax-tvae-h0.5.yaml --mode train --exp-desc 0.5-1.0-1.0-0.5-0.5-0.5-0.5-1.0-1.0-BLEU-h0.5 # 6
--configs configs/data/Quora-GEN.yaml configs/train/Quora-STVAE.yaml configs/model/syntax-tvae2-h0.5.yaml --mode train --exp-desc 0.5-1.0-1.0-0.1-0.1-0.1-0.1-1.0-1.0-BLEU-h0.5-shadow # 7
--configs configs/data/Quora-GEN.yaml configs/train/Quora-STVAE7.yaml configs/model/syntax-tvae-h0.5.yaml --mode train --exp-desc 0.5-5.0-5.0-1.0-1.0-1.0-1.0-1.0-1.0-BLEU-h0.5 # 8

--configs configs/data/Quora-GEN.yaml configs/train/Quora-STVAE9.yaml configs/model/syntax-tvae2-h0.5.yaml --mode train --exp-desc 0.5-1.0-1.0-0.1-0.1-0.1-0.1-2.0-1.0-BLEU-h0.5-shadow # 9
--configs configs/data/Quora-GEN.yaml configs/train/Quora-STVAEA.yaml configs/model/syntax-tvae2-h0.5.yaml --mode train --exp-desc 0.1-1.0-1.0-0.1-0.1-0.1-0.1-1.0-1.0-BLEU-h0.5-shadow # A
--configs configs/data/Quora-GEN.yaml configs/train/Quora-STVAEB.yaml configs/model/syntax-tvae2-h0.5.yaml --mode train --exp-desc 0.1-1.0-1.0-0.1-0.1-0.1-0.1-2.0-1.0-BLEU-h0.5-shadow # B
--configs configs/data/Quora-GEN.yaml configs/train/Quora-STVAEC.yaml configs/model/syntax-tvae2-h0.5.yaml --mode train --exp-desc 0.1-1.0-1.0-0.1-0.1-0.3-0.3-2.0-1.0-BLEU-h0.5-shadow # C

```

Paraphrase
```bash

# dvae-paraphrase
--data-type raw --mode test --configs configs/data/Quora-Paraphrase.yaml configs/model/dss-gmm-vae.yaml configs/train/Quora-DGVAE.yaml --exp-desc n_20-BLEU --eval-func paraphrase

# dvae-meta-paraphrase
--data-type raw --mode test --configs configs/data/Quora-Paraphrase.yaml configs/model/dss-gmm-vae.yaml configs/train/Quora-DGVAE.yaml --exp-desc n_100-BLEU --eval-func meta-paraphrase

# syntax-cvae-paraphrase
--data-type raw --mode test --configs configs/data/Quora-Paraphrase.yaml configs/model/syntax-cvae.yaml configs/train/Quora-SCVAE.yaml --exp-desc 0.5-1.0-1.0-0.1-0.1-0.1-0.1-1.0-1.0-BLEU --eval-func paraphrase

# syntax-tvae-paraphrase
--data-type raw --mode test --configs configs/data/Quora-Paraphrase.yaml configs/model/syntax-tvae.yaml configs/train/Quora-STVAE.yaml --exp-desc 0.5-1.0-1.0-0.1-0.1-0.1-0.1-1.0-1.0-BLEU --eval-func paraphrase


```




