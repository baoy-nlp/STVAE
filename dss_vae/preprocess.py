"""
    - Extract: extract from Con-Parsed Tree.
    - Build: load raw dataset, build dataset, build vocab.
    Examples:
        python preprocess.py --data-configs configs/data/ptb.yaml --mode extract
        python preprocess.py --data-configs configs/data/ptb.yaml --mode build
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from dss_vae.data import build_datasets
from dss_vae.structure import tree_converter_cls
from dss_vae.utils.configs import yaml_to_args
from dss_vae.utils.files import load_docs, write_docs


def split_word_syntax(infile, outfile, linear_func, tree_type):
    with open(infile, 'r') as fi, open(outfile + ".sent", 'w') as fw, open(outfile + ".{}".format(tree_type),
                                                                           'w') as ft:
        for line in fi:
            words, tree_seq = linear_func(line)
            fw.write(words)
            fw.write("\n")
            ft.write(tree_seq)
            ft.write("\n")


def extract_datasets(data_args):
    if not hasattr(data_args, "target_dir") or data_args.target_dir is None:
        data_args.target_dir = data_args.input_dir

    if not os.path.exists(data_args.target_dir):
        os.makedirs(data_args.target_dir)

    if data_args.tree_type is None:
        data_args.tree_type = list(tree_converter_cls.keys())

    for file_pref in data_args.extract_files:
        if isinstance(data_args.tree_type, list):
            for tree_type in data_args.tree_type:
                linear_func = tree_converter_cls[tree_type].linearizing
                split_word_syntax(
                    infile=os.path.join(data_args.input_dir, "{}.{}".format(file_pref, data_args.suffix)),
                    outfile=os.path.join(data_args.target_dir, file_pref),
                    linear_func=linear_func,
                    tree_type=tree_type
                )
        else:
            tree_type = data_args.tree_type
            linear_func = tree_converter_cls[tree_type].linearizing
            split_word_syntax(
                infile=os.path.join(data_args.input_dir, "{}.{}".format(file_pref, data_args.suffix)),
                outfile=os.path.join(data_args.target_dir, file_pref),
                linear_func=linear_func,
                tree_type=tree_type
            )


def split_datasets(docs, train_size, valid_size=0, test_size=0, random_indices=None, seed=1234):
    if random_indices is None:
        import random
        random.seed(seed)
        origin_indices = list(range(len(docs)))
        random_indices = random.sample(origin_indices, train_size + valid_size + test_size)

    shuffle_docs = [docs[ids] for ids in random_indices]

    train = shuffle_docs[:train_size]
    valid = shuffle_docs[train_size:train_size + valid_size] if valid_size > 0 else []
    test = shuffle_docs[train_size + valid_size:] if test_size > 0 else []

    return train, valid, test, random_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='configs/data/Quora-Split4Para.yaml')
    parser.add_argument('--mode', type=str, default='build')
    configs = parser.parse_args()

    args = yaml_to_args(infile=configs.configs)
    if not hasattr(args, 'mode') or args.mode is None:
        args.mode = configs.mode

    if args.mode.lower() == "build":
        build_datasets(
            train_pref=os.path.join(args.input_dir, args.train_pref),
            valid_pref=os.path.join(args.input_dir, args.valid_pref),
            test_pref=os.path.join(args.input_dir, args.test_pref),
            langs=args.langs,
            max_lens=args.max_lens,
            max_nums=args.max_nums,
            task=args.task,
            destdir=args.destdir,
            vocab_sizes=args.vocab_sizes,
            freq_cutoffs=args.freq_cutoffs,
        )
    elif args.mode.lower() == "extract":
        extract_datasets(args)
    elif args.mode.lower() == "scratch":
        extract_datasets(args)
        build_datasets(
            train_pref=os.path.join(args.target_dir, args.train_pref),
            valid_pref=os.path.join(args.target_dir, args.valid_pref),
            test_pref=os.path.join(args.target_dir, args.test_pref),
            langs=args.langs,
            max_lens=args.max_lens,
            max_nums=args.max_nums,
            task=args.task,
            destdir=args.destdir,
            vocab_sizes=args.vocab_sizes,
            freq_cutoffs=args.freq_cutoffs,
        )
    elif args.mode.lower() == 'split':
        full_docs = load_docs(
            fname=os.path.join(args.origin_dir, args.origin_file)
        )
        train_docs, valid_docs, test_docs, _ = split_datasets(
            docs=full_docs,
            train_size=args.train_size,
            valid_size=args.valid_size,
            test_size=args.test_size,
            seed=args.seed
        )
        if not os.path.exists(os.path.join(args.origin_dir, args.split_dir)):
            os.makedirs(os.path.join(args.origin_dir, args.split_dir), exist_ok=True)
        write_docs(os.path.join(args.origin_dir, args.split_dir, "train{}".format(args.split_suffix)), train_docs)
        if args.valid_size > 0:
            write_docs(os.path.join(args.origin_dir, args.split_dir, "valid{}".format(args.split_suffix)), valid_docs)
        if args.test_size > 0:
            write_docs(os.path.join(args.origin_dir, args.split_dir, "test{}".format(args.split_suffix)), test_docs)
