from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.append("..")

from prev_works.prev_dss_vae.structs.dataset import Dataset
from prev_works.prev_dss_vae.structs.vocab import Vocab, VocabEntry


def dataset_infos(train_set, dev_set, test_set, vocab=None):
    def info(data_set):
        _src_vocab = VocabEntry.from_corpus([e.src for e in data_set], )
        _tgt_vocab = VocabEntry.from_corpus([e.tgt for e in data_set], )

        _vocab = Vocab(src=_src_vocab, tgt=_tgt_vocab)
        print('vocabulary %s' % repr(_vocab), file=sys.stdout)

        _source_len = [len(e.src) for e in data_set]
        print('Max source len: {}'.format(max(_source_len)), file=sys.stdout)
        print('Avg source len: %.2f' % (np.average(_source_len)), file=sys.stdout)

        _target_len = [len(e.tgt) for e in data_set]
        print('Max target len: {}'.format(max(_target_len)), file=sys.stdout)
        print('Avg target len: %.2f' % (np.average(_target_len)), file=sys.stdout)

    if vocab is not None:
        print('generated vocabulary %s' % repr(vocab), file=sys.stdout)

    print("Train Set: {}".format(len(train_set)))
    info(train_set)
    print("Valid Set: {}".format(len(dev_set)))
    info(dev_set)
    print("Test Set: {}".format(len(test_set)))
    info(test_set)


def filtering_with_length(dataset, max_src_len=-1, max_tgt_len=-1, max_numbers=-1):
    examples = dataset.examples
    ori_num = len(examples)
    if max_src_len != -1:
        new_examples = []
        for x in examples:
            if len(x.src) < max_src_len:
                new_examples.append(x)
        examples = new_examples
    if max_tgt_len != -1:
        new_examples = []
        for x in examples:
            if len(x.tgt) < max_tgt_len:
                new_examples.append(x)
        examples = new_examples
    if max_numbers != -1:
        from random import sample
        train_idx = sample(range(len(examples)), max_numbers)
        examples = [examples[idx] for idx in train_idx]
    dataset.examples = examples
    pro_num = len(examples)
    print("process from {} -> {}".format(ori_num, pro_num))
    return dataset


def prepare_dataset(
        data_dir, data_dict, tgt_dir, max_src_vocab=16000, max_tgt_vocab=300, vocab_freq_cutoff=1,
        max_src_length=-1, max_tgt_length=-1,
        train_size=-1,
        write_down=True
):
    train_pair = os.path.join(data_dir, data_dict['train'])
    dev_pair = os.path.join(data_dir, data_dict['dev'])
    test_pair = os.path.join(data_dir, data_dict['test'])

    make_dataset(train_pair, dev_pair, test_pair, tgt_dir, max_src_vocab, max_tgt_vocab, vocab_freq_cutoff,
                 max_src_length,
                 max_tgt_length, train_size,
                 write_down)


def make_dataset(
        train_raw, dev_raw=None, test_raw=None, dest_dir=".",
        max_src_vocab=16000,
        max_tgt_vocab=300,
        vocab_freq_cutoff=1,
        max_src_length=-1,
        max_tgt_length=-1,
        train_size=-1,
        write_down=True,
        ext_fields=tuple(),
        instance_type="Plain"
):
    train_set = filtering_with_length(
        Dataset.from_raw_file(train_raw, instance_type),
        max_src_length,
        max_tgt_length,
        max_numbers=train_size)

    # generate vocabulary
    if vocab_freq_cutoff == -1:
        vocab_freq_cutoff = 0
    src_vocab = VocabEntry.from_corpus([e.src for e in train_set], size=max_src_vocab,
                                       freq_cutoff=vocab_freq_cutoff)
    tgt_vocab = VocabEntry.from_corpus([e.tgt for e in train_set], size=max_tgt_vocab,
                                       freq_cutoff=vocab_freq_cutoff)

    sub_vocab_dict = {
        "src": src_vocab,
        "tgt": tgt_vocab
    }
    if len(ext_fields) > 0:
        for sub_vocab_name in ext_fields:
            sub_vocab = VocabEntry.from_corpus([getattr(e, sub_vocab_name) for e in train_set])
            sub_vocab_dict[sub_vocab_name] = sub_vocab
    vocab = Vocab(**sub_vocab_dict)
    print('generated vocabulary %s from %s' % (repr(vocab), train_raw), file=sys.stdout)

    dev_set = filtering_with_length(
        Dataset.from_raw_file(dev_raw, instance_type),
        max_src_length,
        max_tgt_length)

    if test_raw is not None:
        test_set = filtering_with_length(
            Dataset.from_raw_file(test_raw, instance_type),
            max_src_length,
            max_tgt_length)
    else:
        test_set = dev_set

    dataset_infos(train_set, dev_set, test_set)

    if write_down:
        train_file = dest_dir + "/train.bin"
        dev_file = dest_dir + "/dev.bin"
        test_file = dest_dir + "/test.bin"
        vocab_file = dest_dir + "/vocab.bin"

        pickle.dump(train_set.examples, open(train_file, 'wb'))
        pickle.dump(dev_set.examples, open(dev_file, 'wb'))
        pickle.dump(test_set.examples, open(test_file, 'wb'))
        pickle.dump(vocab, open(vocab_file, 'wb'))
        # if 'debug' in data_dict:
        #     debug_set = Dataset.from_raw_file(os.path.join(data_dir, data_dict['debug']))
        #     debug_file = tgt_dir + "/debug.bin"
        #     pickle.dump(debug_set.bin, open(debug_file, 'wb'))


def load_docs(fname):
    res = []
    with open(fname, 'r') as data_file:
        for line in data_file:
            line_res = line.strip("\n")
            res.append(line_res)
    return res


def combine_raw_datset(train_file, dev_file, test_file, dest_dir, src_suffix='word.tok', tgt_suffix="s2t"):
    def _combine(file, split):
        srcs = load_docs("{}.{}".format(file, src_suffix))
        tgts = load_docs("{}.{}".format(file, tgt_suffix))
        with open("{}/{}.raw".format(dest_dir, split), "w") as f:
            for src, tgt in zip(srcs, tgts):
                f.write("{}\t{}".format(src.strip(), tgt.strip()))
                f.write("\n")

    _combine(train_file, split='train')
    _combine(dev_file, split='dev')
    try:
        _combine(test_file, split='test')
    except FileNotFoundError:
        raise RuntimeError("not has test set")

    print('finish combine raw data')


if __name__ == "__main__":
    opt_parser = argparse.ArgumentParser()
    opt_parser.add_argument('--source_dir', dest="source_dir", type=str, help='source dir')
    opt_parser.add_argument('--dest_dir', dest="dest_dir", type=str, help='target dir')
    opt_parser.add_argument('--src', dest="src", type=str, help='src field', default='word.tok')
    opt_parser.add_argument('--tgt', dest="tgt", type=str, help='tgt field')
    opt_parser.add_argument("--train_size", dest="train_size", type=int, default=-1,
                            help="the number of bin select from whole dataset, default is -1, means all")
    opt_parser.add_argument("--max_src_vocab", dest="max_src_vocab", type=int, default=16000,
                            help="source phrase vocab size, default is 16000")
    opt_parser.add_argument("--max_tgt_vocab", dest="max_tgt_vocab", type=int, default=300,
                            help="target phrase vocab size, 300 for parse")
    opt_parser.add_argument("--vocab_freq_cutoff", dest="vocab_freq_cutoff", type=int, default=-1,
                            help="sort freq of word in train set, and cutoff which freq which lower than this value, default is -1")
    opt_parser.add_argument("--max_src_len", dest="max_src_len", type=int, default=-1,
                            help="max length of example 's source input , default is -1")
    opt_parser.add_argument("--max_tgt_len", dest="max_tgt_len", type=int, default=-1,
                            help="max length of example 's target output , default is -1")
    opt_parser.add_argument("--mode", dest="mode", type=str, default="Plain",
                            choices=['Plain', 'PTB', 'SyntaxVAE', 'SyntaxGEN'],
                            help="vocab filed 's mode, default is plain, ")

    opt = opt_parser.parse_args()
    if not os.path.exists(opt.dest_dir):
        os.makedirs(opt.dest_dir)
    combine_raw_datset(
        train_file="{}/train.tree".format(opt.source_dir),
        dev_file="{}/dev.tree".format(opt.source_dir),
        test_file="{}/test.tree".format(opt.source_dir),
        dest_dir=opt.dest_dir,
        src_suffix=opt.src,
        tgt_suffix=opt.tgt,
    )

    make_dataset(
        train_raw="{}/train.raw".format(opt.dest_dir),
        dev_raw="{}/dev.raw".format(opt.dest_dir),
        test_raw="{}/test.raw".format(opt.dest_dir),
        dest_dir=opt.dest_dir,
        max_src_vocab=opt.max_src_vocab,
        max_tgt_vocab=opt.max_tgt_vocab,
        max_src_length=opt.max_src_len,
        max_tgt_length=opt.max_tgt_len,
        vocab_freq_cutoff=opt.vocab_freq_cutoff,
        train_size=opt.train_size,
        instance_type=opt.mode
    )
