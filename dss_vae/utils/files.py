def load_docs(fname):
    res = []
    with open(fname, 'r') as data_file:
        for line in data_file:
            line_res = line.strip("\n")
            res.append(line_res)
    return res


def write_docs(fname, docs):
    if isinstance(docs, dict):
        docs = ["{}:{}".format(key, val) for key, val in docs.items()]

    with open(fname, 'w') as f:
        for doc in docs:
            f.write(str(doc).strip())
            f.write('\n')
