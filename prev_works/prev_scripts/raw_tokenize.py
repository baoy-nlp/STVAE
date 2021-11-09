"""
nltk tokenizer for raw data.

"""
import argparse


# import nltk
# nltk.download('punkt')
# from nltk.tokenize import word_tokenize

def bracket_replace(line):
    line = line.strip()
    line = line.replace("(", "-LRB-")
    line = line.replace(")", "-RRB-")
    line = line.replace("[", "-LRB-")
    line = line.replace("]", "-RRB-")
    line = line.replace("{", "-LRB-")
    line = line.replace("}", "-RRB-")

    line = line.replace("-lrb-", "-LRB-")
    line = line.replace("-rrb-", "-RRB-")

    return line


def tokenizer(line, for_parse=False, is_lower=False, language='english'):
    # post = word_tokenize(line.strip(), language=language)
    # post_str = " ".join(post)
    post_str = line.strip()
    if is_lower:
        post_str = post_str.lower()
    if for_parse:
        post_str = bracket_replace(post_str)
    return post_str


def tokenizing(input_file, output_file, for_parse=False, is_lower=False, is_bpe=False, language='english'):
    """ tokenizing the input file to output file """
    process_docs = load_docs(input_file)
    if is_bpe:
        def process(x):
            x = x.replace('@@ ', '')
            if is_lower:
                x = x.lower()
            return x

        with open(output_file, 'w') as f:
            for line in process_docs:
                f.write(str(process(line.strip())))
                f.write("\n")

    else:
        with open(output_file, 'w') as f:
            for line in process_docs:
                f.write(str(tokenizer(line.strip(), for_parse, is_lower, language)))
                f.write("\n")


def load_docs(fname):
    res = []
    with open(fname, 'r') as data_file:
        for line in data_file:
            line_res = line.strip("\n")
            res.append(line_res)
    return res


if __name__ == "__main__":
    opt_parser = argparse.ArgumentParser()
    opt_parser.add_argument('--input_file', dest="input_file", type=str)
    opt_parser.add_argument('--output_file', dest="output_file", type=str)
    opt_parser.add_argument('--for_parse', dest="is_parse", action="store_true", default=False)
    opt_parser.add_argument('--for_bpe', dest="is_bpe", action='store_true', default=False)
    opt_parser.add_argument('--is_lower', dest="is_lower", action="store_true", default=False)
    opt_parser.add_argument('--language', dest='language', type=str, default='language')
    opt = opt_parser.parse_args()

    if opt.output_file is None:
        setattr(opt, 'output_file', "{}.tok".format(opt.input_file))

    tokenizing(
        opt.input_file, opt.output_file, for_parse=opt.is_parse, is_lower=opt.is_lower,
        is_bpe=opt.is_bpe, language=opt.language
    )
