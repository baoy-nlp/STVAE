# Data Converter
# for PTB: Tree.txt to (src.txt, syn.txt)
# for Quora: Tree.txt to (src.txt, tgt.txt, syn.txt)

import os
from argparse import ArgumentParser
from nltk.tokenize import TweetTokenizer as Tokenizer
from utils.linearized_tree import ReversibleLinearTree


def tree_convert(fname):
    sentences, trees = [], []
    tokenizer = Tokenizer(preserve_case=False)
    for i, line in enumerate(open(fname)):
        sent, syn = ReversibleLinearTree.linearizing(line.strip())
        sentences.append(" ".join(tokenizer.tokenize(sent)))
        trees.append(syn)
    return sentences, trees


def process(train, valid, test, dest):
    def write_docs(docs, fname):
        with open(fname, "w") as f:
            for doc in docs:
                f.writelines(doc.strip())
                f.write("\n")

    def _process(fname, split):
        sents, trees = tree_convert(fname)
        write_docs(sents, os.path.join(dest, "{}.src.txt".format(split)))
        write_docs(trees, os.path.join(dest, "{}.syn.txt".format(split)))

    if not os.path.exists(dest):
        os.makedirs(dest, exist_ok=False)

    _process(train, split="train")
    _process(valid, split="valid")
    _process(test, split="test")


if __name__ == "__main__":
    parser = ArgumentParser(description="Tree to (src,syn) or (src,tgt,syn)")
    parser.add_argument('--raw-dir', type=str, help="raw data")
    parser.add_argument('--dest-dir', type=str, help="destination dir")
    parser.add_argument('--dataset', type=str, help='dataset name')

    args = parser.parse_args()

    if args.dataset == "Quora":
        unpair_path = os.path.join(args.raw_dir, "full.unpair.con.txt")
        pair_path = os.path.join(args.raw_dir, "full.pair.con.txt")

    if args.dataset == "PTB":
        train_file = "train.clean"
        valid_file = "dev.clean"
        test_file = "test.clean"

        process(
            train=os.path.join(args.raw_dir, "train.clean"),
            valid=os.path.join(args.raw_dir, "dev.clean"),
            test=os.path.join(args.raw_dir, "test.clean"),
            dest=args.dest_dir
        )
