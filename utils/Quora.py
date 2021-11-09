import csv
import sys


def load_docs(filename):
    docs = []
    with open(filename, "r") as f:
        for line in f.readlines():
            docs.append(line.strip())
    return docs


def write_docs(docs, filename):
    with open(filename, "w") as f:
        for doc in docs:
            if len(doc.strip("\n")) != 0:
                f.write(doc.strip())
                f.write("\n")


def load_csv_file(filename, ignore_first_line=False):
    results = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for items in reader:
            if reader.line_num == 1 and ignore_first_line:
                continue
            results.append(items)

    return results


def parse_raw_sentence(load_list):
    res = list(range(len(load_list) * 2))
    end_idx = 0
    for item in load_list:
        res[int(item[1])] = item[3]
        res[int(item[2])] = item[4]

        end_idx = max(int(item[1]), int(item[2]), end_idx)

    for idx, item in enumerate(res):
        if isinstance(item, int) or len(item.strip("\n")) == 0:
            res[idx] = str(idx)

    res = res[:end_idx + 2]

    return res


def parse_test_sentence(load_list):
    res = []

    for item in load_list:
        res.append(item[1])
        res.append(item[2])
    return res


def parse_pair_label(load_list):
    res = []
    for item in load_list:
        # item = item.strip().split("\t")
        res.append("\t".join([item[1], item[2], item[-1]]))
    return res


def get_detection_data(pair_list, sent_list):
    res = []
    for item in pair_list:
        item = item.strip().split("\t")
        res.append("\t".join([sent_list[int(item[0])], sent_list[int(item[1])], item[2]]))
    return res


def get_parallel_data(pair_list, sent_list):
    res = []
    for item in pair_list:
        item = item.strip().split("\t")
        if int(item[2]) == 1:
            res.append("\t".join([sent_list[int(item[0])], sent_list[int(item[1])]]))
    return res


def get_unpaired_data(pair_list, sent_list):
    res = []
    for item in pair_list:
        item = item.strip().split("\t")
        if int(item[2]) == 0:
            res.append("\t".join([sent_list[int(item[0])], sent_list[int(item[1])]]))
    return res


def remove_pos(docs):
    words = []
    tags = []
    for doc in docs:
        tokens = doc.strip().split(" ")
        sentence = " ".join([token[:token.rfind("/")] for token in tokens])
        tag = " ".join([token[token.rfind("/") + 1:] for token in tokens])
        words.append(sentence)
        tags.append(tag)
    return words, tags


def split_pair(docs):
    origins = []
    paras = []
    for doc in docs:
        pairs = doc.strip().split("\t")

        origins.append(pairs[0])
        paras.append(pairs[1])
    return origins, paras


def main():
    mode = sys.argv[1]
    in_file = sys.argv[2]
    out_file = sys.argv[3]
    if mode == "parse_raw_sentence":
        load_list = load_csv_file(in_file, ignore_first_line=True)
        res_list = parse_raw_sentence(load_list)
        write_docs(res_list, out_file)
    elif mode == "parse_test_sentence":
        load_list = load_csv_file(in_file, ignore_first_line=True)
        res_list = parse_test_sentence(load_list)
        write_docs(res_list, out_file)
    elif mode == "parse_pair_label":
        load_list = load_csv_file(in_file, ignore_first_line=True)
        res_list = parse_pair_label(load_list)
        write_docs(res_list, out_file)
    elif mode == "detection":
        sent_file = sys.argv[4]
        sent_list = load_docs(sent_file)
        pair_list = load_docs(in_file)
        res_list = get_detection_data(pair_list, sent_list)
        write_docs(res_list, out_file)
    elif mode == "parallel":
        sent_file = sys.argv[4]
        sent_list = load_docs(sent_file)
        pair_list = load_docs(in_file)
        res_list = get_parallel_data(pair_list, sent_list)
        write_docs(res_list, out_file)
    elif mode == "unpaired":
        sent_file = sys.argv[4]
        sent_list = load_docs(sent_file)
        pair_list = load_docs(in_file)
        res_list = get_unpaired_data(pair_list, sent_list)
        write_docs(res_list, out_file)
    elif mode == "remove-pos":
        docs = load_docs(in_file)
        words, tags = remove_pos(docs)
        write_docs(words, out_file)
        write_docs(tags, sys.argv[4])
    elif mode == 'split-pair':
        docs = load_docs(in_file)
        origins, paras = split_pair(docs)
        write_docs(origins, out_file)
        write_docs(paras, sys.argv[4])
    else:
        print("error mode:{}".format(mode))


if __name__ == "__main__":
    main()
