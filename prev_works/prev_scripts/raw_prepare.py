# coding=utf-8
"""
    process the raw file for bin construct
"""
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

sys.path.append("..")

from prev_works.prev_dss_vae.tools import tree_convert
from prev_works.prev_dss_vae.tools import tree_to_sequence_without_backtrack


def load_docs(fname):
    res = []
    with open(fname, 'r') as data_file:
        for line in data_file:
            line_res = line.strip("\n")
            res.append(line_res)
    return res


def write_docs(fname, docs):
    with open(fname, 'w') as f:
        for doc in docs:
            f.write(str(doc))
            f.write('\n')


def remove_same(docs):
    check = {}
    res = []
    for doc in docs:
        if doc not in check:
            check[doc] = 1
            res.append(doc)
        else:
            pass
    print("same data filter:{}".format(len(docs) - len(res)))
    return res


def ptb_to_s2b(tree_file, rm_same=False):
    with open(tree_file, 'r') as tf:
        s2b_list = []
        for tree_str in tf.readlines():
            if len(tree_str.strip()) > 0:
                if not tree_str.strip().startswith("(TOP"):
                    tree_str = "(TOP " + tree_str.strip() + ")"
                try:
                    src, tgt = tree_to_sequence_without_backtrack(tree_str.strip())
                    s2b_list.append("\t".join([src, tgt]))
                except:
                    print(tree_str.strip())

    if rm_same:
        s2b_list = remove_same(s2b_list)

    return s2b_list


def ptb_to_ae(tree_file):
    with open(tree_file, 'r') as tf:
        ae_list = []
        for tree_str in tf.readlines():
            if len(tree_str.strip()) > 0:
                if not tree_str.strip().startswith("(TOP"):
                    tree_str = "(TOP " + tree_str.strip() + ")"
                try:
                    src, tgt = tree_to_sequence_without_backtrack(tree_str.strip())
                    ae_list.append("\t".join([src, src]))
                except:
                    print(tree_str.strip())

    return ae_list


def make_s2b_dataset(train_file, dev_file=None, test_file=None, tgt_dir=None, rm_same=False):
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    train = ptb_to_s2b(train_file, rm_same)
    write_docs(fname=os.path.join(tgt_dir, "train.s2b"), docs=train)
    if dev_file is not None:
        dev = ptb_to_s2b(dev_file)
        write_docs(fname=os.path.join(tgt_dir, "dev.s2b"), docs=dev)
    if test_file is not None:
        test = ptb_to_s2b(test_file)
        write_docs(fname=os.path.join(tgt_dir, "test.s2b"), docs=test)


def make_ae_dataset(train_file, dev_file=None, test_file=None, tgt_dir=None, rm_same=False):
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    train = ptb_to_ae(train_file)
    write_docs(fname=os.path.join(tgt_dir, "train.ae"), docs=train)
    if dev_file is not None:
        dev = ptb_to_ae(dev_file)
        write_docs(fname=os.path.join(tgt_dir, "dev.ae"), docs=dev)
    if test_file is not None:
        test = ptb_to_ae(test_file)
        write_docs(fname=os.path.join(tgt_dir, "test.ae"), docs=test)


# below is universal version

def ptb_convert_func(tree_input_list, mode="s2t"):
    tree_convert_method = tree_convert(mode=mode)
    word_list = []
    tree_list = []
    for tree_str in tree_input_list:
        if len(tree_str.strip()) > 0:
            if not tree_str.strip().startswith("(TOP"):
                tree_str = "(TOP " + tree_str.strip() + ")"
            try:
                word, tree = tree_convert_method(tree_str.strip())
                word_list.append(word)
                tree_list.append(tree)
            except:
                print(tree_str.strip())
    return [word_list, tree_list]


def ptb_convert(tree_file, out_file=None, mode="s2t", convert_mode="SyntaxVAE"):
    tree_input_list = load_docs(tree_file)
    if out_file is None:
        out_file = tree_file
    tree_output_list = ptb_convert_func(tree_input_list, mode)
    write_docs(fname="{}.{}".format(out_file, mode), docs=tree_output_list[1])

    if not convert_mode == 'convert':
        write_docs(fname="{}.word".format(out_file), docs=tree_output_list[0])


def prepare_syntax_gen(src_tree_file, tgt_tree_file=None, out_file=None):
    src_list = load_docs(src_tree_file)
    if tgt_tree_file is None:
        tgt_tree_file = src_tree_file
    tgt_list = load_docs(tgt_tree_file)
    assert len(src_list) == len(tgt_list), "Do not match of src file and tgt file"
    out_list = ["\t".join([src, tgt]) for src, tgt in zip(src_list, tgt_list)]
    write_docs(fname=out_file, docs=out_list)
    print("finish process to ", out_file)


def unk_replace_for_pair(vocab_file, set_file, out_file):
    """
    replace the pair data's inputs with unk token.
    Args:
        vocab_file:
        set_file:
        out_file:

    Returns:

    """
    from prev_works.prev_dss_vae.structs.vocab import Vocab
    from prev_works.prev_dss_vae.structs.dataset import Dataset
    vocab = Vocab.from_bin_file(vocab_file).src
    examples = Dataset.from_bin_file(set_file).examples
    src_out = []
    tgt_out = []
    for e in examples:
        res = [
            src if not vocab.is_unk(src) else "<unk>"
            for src in e.src
        ]
        src_out.append(" ".join(res))
        tgt_out.append(" ".join(e.tgt))
    out_list = ["\t".join([src, tgt]) for src, tgt in zip(src_out, tgt_out)]
    write_docs(fname=out_file, docs=out_list)


def unk_replace_for_LM(vocab_file, set_file, out_file):
    """
    unk replace for LM data's inputs
    Args:
        vocab_file:
        set_file:
        out_file:

    Returns:

    """
    from prev_works.prev_dss_vae.structs.vocab import Vocab
    from prev_works.prev_dss_vae.structs.dataset import Dataset
    vocab = Vocab.from_bin_file(vocab_file).src
    examples = Dataset.from_bin_file(set_file).examples

    out_list = []

    for example in examples:
        res = [
            token if not vocab.is_unk(token) else "<unk>"
            for token in example.src
        ]
        out_list.append(" ".join(res))
    write_docs(fname=out_file, docs=out_list)


if __name__ == "__main__":
    opt_parser = argparse.ArgumentParser()
    opt_parser.add_argument('--tree_file', dest="tree_file", type=str, help='tree file with Penn TreeBank Format[must]')
    opt_parser.add_argument('--syn_mode', dest="syn_mode", type=str, default="s2b",
                            help="linearized tree format:[s2t,s2b,s2s], default is s2b")
    opt_parser.add_argument("--src_file", dest="src_file", type=str, help="source tree data file")
    opt_parser.add_argument("--tgt_file", dest="tgt_file", type=str, help="paraphrase tree data file")
    opt_parser.add_argument('--out_file', dest="out_file", type=str, help='output path[optional]')
    opt_parser.add_argument('--convert_mode', dest='convert_mode', type=str, default="SyntaxVAE")
    opt_parser.add_argument('--vocab_file', type=str, default="Vocab")
    opt_parser.add_argument("--data_mode", dest="data_mode", type=str, default="SyntaxVAE",
                            choices=['Convert', 'SyntaxVAE', 'NAG', "UNK-PARA", "UNK-LM"],
                            help="data format to be prepare,default is SyntaxVAE, other is NAG")
    opt = opt_parser.parse_args()
    if opt.data_mode == "SyntaxVAE":
        ptb_convert(tree_file=opt.src_file, out_file=opt.out_file, mode=opt.syn_mode, convert_mode=opt.convert_mode)
    elif opt.data_mode == "UNK-PARA":
        unk_replace_for_pair(
            vocab_file=opt.vocab_file,
            set_file=opt.src_file,
            out_file=opt.out_file
        )
    elif opt.data_mode == "UNK-LM":
        unk_replace_for_LM(
            vocab_file=opt.vocab_file,
            set_file=opt.src_file,
            out_file=opt.out_file
        )
