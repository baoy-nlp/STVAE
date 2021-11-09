# Generating Sentences with *VAE

More experiments details will come soon.

Our implementations include:
- **VAE**: [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349) by Bowman et al. 2015.
- **SyntaxVAE**: [Generating Sentences from Disentangled Syntactic and Semantic Spaces](http://nlp.nju.edu.cn/homepage/Papers/bao_acl19.pdf) by 
Bao et al. 2019.
- **SyntaxTVAE**: _Unsupervised Paraphrasing via Syntactic Template Sampling_ by Bao et al.

## Training
- AutoEncoder:
    ```bash
    python train.py --data-path /home/data_ti4_c/baoy/data/VAE/PTB/splits --exp-dir /home/data_ti4_c/baoy/experiments/checkpoints/PTB/AE-EXP0 --log-dir /home/data_ti4_c/baoy/experiments/logs/PTB/AE-EXP0 --share-vocab --share-enc-dec-embed --output-without-embed --bidir --gpu --batch-size 32 --max-sents 60 --embed-dim 353 --hidden-dim 191 --rnn-type lstm --num-layers 1 --latent-dim 50 --unk-factor 0.38 --kl-x0 2500 --kl-factor 0.0
    ```
- VAE: 
    ```bash
    python train.py --data-path /home/data_ti4_c/baoy/data/VAE/PTB/splits --exp-dir /home/data_ti4_c/baoy/experiments/checkpoints/PTB/VAE-EXP0 --log-dir /home/data_ti4_c/baoy/experiments/logs/PTB/VAE-EXP0 --share-vocab --share-enc-dec-embed --output-without-embed --bidir --gpu --batch-size 32 --max-sents 60 --embed-dim 353 --hidden-dim 191 --rnn-type lstm --num-layers 1 --latent-dim 50 --unk-factor 0.38 --kl-x0 2500 --kl-factor 1.0
    ```
- SyntaxVAE:
    ```bash
    python train.py --data-path /home/data_ti4_c/baoy/data/VAE/PTB/splits --exp-dir /home/data_ti4_c/baoy/experiments/checkpoints/PTB/SVAE-EXP9 --log-dir /home/data_ti4_c/baoy/experiments/logs/PTB/SVAE-EXP9 --share-vocab --share-enc-dec-embed --output-without-embed --bidir --gpu --cls SyntaxVAE --syn-embed-dim 50 --syn-hidden-dim 100 --batch-size 32 --max-sents 60 --embed-dim 353 --hidden-dim 191 --rnn-type lstm --num-layers 1 --latent-dim 50 --unk-factor 0.38 --kl-x0 2500 --kl-factor 1.0 --adv-sem 0.5 --adv-syn 0.5 --rec-sem 0.1 --rec-syn 0.1
    ```
- SyntaxTVAE:
   

## Inference

