"""
    Moses tokenizer for raw data.
    Examples:
        python tokenize.py --input-file
"""
import argparse
import os


def bracket_normalize(raw):
    raw = raw.strip()
    raw = raw.replace("(", "-LRB-")
    raw = raw.replace(")", "-RRB-")
    raw = raw.replace("[", "-LRB-")
    raw = raw.replace("]", "-RRB-")
    raw = raw.replace("{", "-LRB-")
    raw = raw.replace("}", "-RRB-")
    raw = raw.replace("-lrb-", "-LRB-")
    raw = raw.replace("-rrb-", "-RRB-")
    return raw


def bracket_min_normalize(raw):
    raw = raw.strip()
    raw = raw.replace("(", "（")
    raw = raw.replace(")", "）")
    return raw


def get_moses_tokenizer(lang: str = "en"):
    if lang.startswith("en"):
        from mosestokenizer import MosesTokenizer
        return MosesTokenizer('en')
    else:
        return lambda x: x.strip().split(" ")


def main(input_file, output_file, is_syntax=False, to_lower=False, lang='english'):
    """ tokenizing the input file to output file """
    word_tokenizer = get_moses_tokenizer(lang=lang)

    def tokenize(raw):
        post = word_tokenizer(raw.strip())
        if len(post) > 1 and post[-1] == "?":
            post = post[:-1]
        post_str = " ".join(post).strip()
        if to_lower:
            post_str = post_str.lower()
        if is_syntax:
            # post_str = bracket_normalize(post_str)
            post_str = bracket_min_normalize(post_str)
        return post_str

    with open(input_file, 'r', encoding='utf8') as fr, open(output_file, 'w', encoding='utf-8') as f:
        for line in fr:
            f.write(str(tokenize(line.strip())))
            f.write("\n")


if __name__ == "__main__":
    opt_parser = argparse.ArgumentParser()
    opt_parser.add_argument('--input', type=str)
    opt_parser.add_argument('--output', type=str)
    opt_parser.add_argument('--syntax', action="store_true", default=False)
    opt_parser.add_argument('--lower', action="store_true", default=False)
    opt_parser.add_argument('--lang', type=str, default='en')
    opt = opt_parser.parse_args()

    assert opt.input is not None, '--input need set'
    if opt.output is None:
        import os

        filename, ext = os.path.splitext(opt.input)
        suffix = ""
        if opt.syntax:
            suffix += ".syntax"
        if opt.lower:
            suffix += ".lower"
        opt.output = "{}{}{}".format(filename, suffix, ext)

    if not os.path.exists(os.path.dirname(opt.output)):
        os.makedirs(os.path.dirname(opt.output))

    main(opt.input, opt.output, is_syntax=opt.syntax, to_lower=opt.lower, lang=opt.lang)
