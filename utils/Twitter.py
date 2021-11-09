import sys

"""
The format of the example in the Twitter's file is :
    ori_sent1
    para_sent1
    ori_sent2
    para_sent2

We use this script to process the data into the form we need, including:
    - split: convert origin file to [ori.file para.file]
    - combine: convert origin line to [ori_sent, para_sent]
    
"""


def load_docs(filename):
    docs = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            docs.append(line.strip())
    return docs


def split(in_file, out_prefix, suffix="txt"):
    docs = load_docs(in_file)
    with open("{}.ori.{}".format(out_prefix, suffix), "w", encoding="utf-8") as ow, \
            open("{}.par.{}".format(out_prefix, suffix), "w", encoding="utf-8") as pw:
        for i, doc in enumerate(docs):
            if i % 2 == 0:
                ow.write(doc.strip())
                ow.write("\n")
            else:
                pw.write(doc.strip())
                pw.write("\n")


def combine():
    pass


def main():
    mode = sys.argv[1]
    in_file = sys.argv[2]
    out_prefix = sys.argv[3]
    out_suffix = sys.argv[4]

    if mode == "split":
        split(in_file, out_prefix, out_suffix)


if __name__ == "__main__":
    main()
