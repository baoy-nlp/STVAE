import logging
import os
import random
import shutil
from argparse import ArgumentParser
from time import gmtime, strftime

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torchtext.data import BucketIterator

from syntax_vae import load_checkpoint, save_checkpoint
from syntax_vae.dataset import get_dataset


# from torch.utils.data import DataLoader


def get_options():
    parser = ArgumentParser(description="Train a VAE model")

    # basic file configuration
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--max-sents', type=int, default=1000)
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--exp-dir', type=str, default='.exp')
    parser.add_argument('--log-dir', type=str, default='.log')
    parser.add_argument('--prefix', type=str, default='[time]', help='prefix to denote the model, nothing or [time]')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--task", type=str, default="syntax-vae")
    parser.add_argument('--gpu', action='store_true')

    # Generic VAE model parameters
    parser.add_argument("--seed", type=int, default=931029)
    parser.add_argument('--cls', type=str, default="VAE")
    parser.add_argument('--share-vocab', action='store_true', default=True)
    parser.add_argument('--share-input-output-embed', action='store_true', default=False)
    parser.add_argument('--share-enc-dec-embed', action="store_true", default=False)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=100)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--bidir', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--rnn-dropout', type=float, default=0.1)
    parser.add_argument('--rnn-type', type=str, default='gru')
    parser.add_argument('--output-without-embed', action="store_true", default=False)

    # SyntaxVAE model parameters
    parser.add_argument('--sem-latent-dim', type=int)
    parser.add_argument('--syn-latent-dim', type=int)
    parser.add_argument('--syn-embed-dim', type=int)
    parser.add_argument('--syn-hidden-dim', type=int)
    # parser.add_argument('--syn-share-input-output-embed',action='store_true')
    # parser.add_argument('--syn-output-without-embed',action='store_true')

    # Generic VAE Trainer parameters
    parser.add_argument('--unk-factor', type=float, default=0.5)
    parser.add_argument('--kl-factor', type=float, default=1.0)
    parser.add_argument('--unk-anneal-func', type=str, default='fixed')
    parser.add_argument('--unk-x0', type=int, default=-1)
    parser.add_argument('--unk-k', type=float, default=0.0025)
    parser.add_argument('--kl-anneal-func', type=str, default='logistic')
    parser.add_argument('--kl-x0', type=int, default=2500)
    parser.add_argument('--kl-k', type=int, default=0.0025)
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-decay', type=float, default=0.75)
    parser.add_argument('--patience', type=int, default=5)

    # Syntax VAE Trainer parameters:
    parser.add_argument("--mul-sem", type=float, default=0.5)
    parser.add_argument("--mul-syn", type=float, default=0.5)
    parser.add_argument("--adv-sem", type=float, default=0.0)
    parser.add_argument("--adv-syn", type=float, default=0.0)
    parser.add_argument("--rec-sem", type=float, default=0.0)
    parser.add_argument("--rec-syn", type=float, default=0.0)

    parser.add_argument("--kl-sem-factor", type=float, default=1.0)
    parser.add_argument("--kl-syn-factor", type=float, default=1.0)
    parser.add_argument("--share-mul-adv-out", action="store_true")

    # Training Parameters
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--valid-every', type=int, default=300)
    parser.add_argument('--maximize', action="store", default=False)
    parser.add_argument('--metric', type=str, default="ELBO")

    parser.add_argument("--tau-t", type=float, default=1.0)
    parser.add_argument("--gumbel-h", action="store_true")
    parser.add_argument("--s-factor", type=float, default=1.0)
    parser.add_argument("--t-factor", type=float, default=1.0)

    return parser.parse_args()


def get_path(args):
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    else:
        if not args.debug and len(os.listdir(args.exp_dir)):
            key_in = input("Cover Model? Y or N\n")
            if key_in.upper().startswith("Y"):
                shutil.rmtree(args.exp_dir)
                os.makedirs(args.exp_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    else:
        if not args.debug and len(os.listdir(args.log_dir)):
            key_in = input("Cover Log? Y or N\n")
            if key_in.upper().startswith("Y"):
                shutil.rmtree(args.log_dir)
                os.makedirs(args.log_dir)

    if args.prefix == '[time]':
        args.prefix = strftime("%m.%d_%H.%M.", gmtime())

    args.save_path = os.path.join(args.exp_dir, "checkpoint_last.pt")
    args.best_path = os.path.join(args.exp_dir, "checkpoint_best.pt")
    args.logger_path = os.path.join(args.log_dir, "log")
    return args


def get_batch_size_fn(args):
    if args.batch_size != -1:
        from syntax_vae.dataset import batch_with_sentences
        batch_size = args.batch_size
        batch_size_fn = batch_with_sentences
    else:
        from syntax_vae.dataset import dyn_batch_without_padding
        batch_size = args.max_tokens
        batch_size_fn = dyn_batch_without_padding
    return batch_size, batch_size_fn


def validate(model, trainer, valid_data, logger=None):
    tracker = {"Batch": 0}
    for batch in valid_data:
        model.eval()
        ret = trainer.forward(model, batch)
        tracker = trainer.update_record(tracker, ret)

    if logger is not None:
        logger.info("== Valid {} ==".format(trainer.print_record(tracker, step=trainer.step)))

    return tracker


def _main():
    args = get_options()
    args = get_path(args)

    logger = logging.getLogger("{} {} ".format("Train", args.cls.upper()))
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if not args.debug:
        fh = logging.FileHandler('{}-{}.txt'.format(args.logger_path, args.prefix))
        # fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        # ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trainset, validset, testset = get_dataset(
        task=args.task,
        share_vocab=args.share_vocab,
        vocab_size=30000,
        path=args.data_path,
        splits=("train", "valid", "test"),
        logger=logger,
        length=args.max_sents
    )

    args, model, trainer, optimizer = load_checkpoint(args, args.save_path, trainset, logger=logger)

    logger.info(model)
    if not args.debug:
        tw = SummaryWriter(os.path.join(args.log_dir, "train"))
        vw = SummaryWriter(os.path.join(args.log_dir, "valid"))

    if not args.debug:
        tw.add_text("model", str(model))
        tw.add_text("args", str(args))
        tw.add_text("trainer", str(trainer))

    batch_size, batch_size_fn = get_batch_size_fn(args)

    train_real, valid_real = BucketIterator.splits(
        (trainset, validset), batch_sizes=(batch_size, batch_size),
        device=torch.device("cuda:0") if torch.cuda.is_available() else None,
        batch_size_fn=batch_size_fn, repeat=None, sort_within_batch=True,
    )

    try:
        for epoch in range(1, args.max_epochs + 1):
            train_record = {"Batch": 0}
            for batch in train_real:
                model.train()
                step = trainer.step + 1
                batch_ret = trainer.forward(model, batch, optimizer)
                train_record = trainer.update_record(train_record, batch_ret)

                if step % args.log_every == 0:
                    if not args.debug:
                        for key, val in train_record.items():
                            tw.add_scalar("{}".format(key), val, step)

                    logger.info("Train Batch {} {}".format(step, trainer.print_out(batch_ret)))

                if step % args.valid_every == 0:
                    valid_record = validate(model, trainer, valid_real, logger)
                    if not args.debug:
                        for key, val in valid_record.items():
                            vw.add_scalar("{}".format(key), val, step)

                    val = valid_record[args.metric]
                    is_better = trainer.is_best(val)
                    trainer.update_optimizer(is_better, optimizer)

                    if not args.debug:
                        save_checkpoint(args, args.save_path, model, trainer, optimizer, logger)
                        if is_better:
                            save_checkpoint(args, args.best_path, model, trainer, optimizer, logger)
                            KL = valid_record.get("KL/Loss", 0)
                            NLL = valid_record.get("NLL", 0)
                            if KL > 0:
                                vw.add_scalar("Best/KL", valid_record['KL/Loss'], step)
                            if NLL > 0:
                                vw.add_scalar("Best/NLL", valid_record["NLL"], step)
                        vw.add_scalar("Best/{}".format(args.metric), trainer.best_metric, step)
                    logger.info("Best {}: {:5.2f}".format(args.metric, trainer.best_metric))

            logger.info("== Train {} ==".format(trainer.print_record(train_record, epoch)))


    except KeyboardInterrupt:
        if not args.debug:
            save_checkpoint(args, args.save_path, model, trainer, optimizer, logger)


if __name__ == "__main__":
    _main()
