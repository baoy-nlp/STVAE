import argparse
import math
import os

INF = math.inf


def length_filter(fname, lower=0, upper=100, keep_num=INF, spliter="\t"):
    def length(strs):
        return len(strs.split(" "))

    def in_condition(strs):
        parsed_line = strs.split(spliter)
        return sum([lower < length(line) <= upper for line in parsed_line])

    docs = []

    with open(fname, "r") as f:
        while len(docs) < keep_num:
            line = f.readline()
            if not line:
                break
            if in_condition(line):
                docs.append(line)
    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--lower', type=int, default=5)
    parser.add_argument('--upper', type=int, default=100)
    parser.add_argument('--keep-num', type=int, default=200000)

    opt = parser.parse_args()

    assert opt.input is not None, '--input need set'
    if opt.output is None:
        filename, ext = os.path.splitext(opt.input)
        suffix = "L{}-U{}_N{}".format(opt.lower, opt.upper, opt.keep_num)
        opt.output = "{}-{}{}".format(filename, suffix, ext)

    docs = length_filter(opt.input, opt.lower, opt.upper, opt.keep_num)
    with open(opt.output, 'w') as f:
        for doc in docs:
            f.write(doc.strip())
            f.write("\n")
