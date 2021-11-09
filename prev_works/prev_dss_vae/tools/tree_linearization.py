"""
for tree_str:(TOP (NP (NNP EDUCATION) (NNPS ADS) (: :)))
Tree Sequence Format
    S2T: NP NNP NNPS : /NP
    S2B: NP NNP NNPS : /BRK
    S2S: NP XX XX : /NP

"""


def tree_sequence_check(s2t_str):
    """
    Example:
        tree_string = "(TOP (S (VP (VBP Do) (RB n't) (VP (VB be) (NP (JJ such) (DT a) (NN pessimist)) (, ,) (NP (NNP Mr.) (NNP Ambassador)))) (. .)))"
        _, tgt = tree_to_s2t(tree_string)
        print(s2t_check(tgt))
        print(s2t_check(tgt + " /NP"))
        print(s2t_check("S " + tgt))
        print(s2t_check("S NP /S" + tgt))
        print(s2t_check("S NP " + tgt + " /S"))
    """
    tokens = s2t_str.strip("\n").split(" ")
    word_format = "{})"
    stack = []
    for item in tokens:
        if not item.startswith("/"):
            stack.append((item, False))
        else:
            query = item[1:]
            has_find = False
            mid_cell = 0
            item_str = ")"
            while len(stack) > 0:
                key = stack.pop(-1)
                if key[0] == query:
                    item_str = "({} {}".format(key[0], item_str)
                    has_find = True
                    break
                else:
                    mid_cell += 1
                    if not key[1]:
                        item_str = "({} {} {}".format(key[0], word_format, item_str)
                    else:
                        item_str = "{} {}".format(key[0], item_str)

            if mid_cell == 0 or not has_find:
                return False

            stack.append((item_str, True))

    if len(stack) > 1:
        return False
    return True


def tree_sequence_fix(s2t, fm):
    tokens = s2t.strip("\n").split(" ")
    stack = []
    res = []
    error_fix = 0

    for item in tokens:
        if not item.startswith("/"):
            if fm.is_tag(item):
                if len(stack) > 0:
                    res.append(item)
                else:
                    error_fix += 1
            else:
                stack.append(item)
                res.append(item)
        else:
            if len(stack) > 0:
                top = stack.pop(-1)
                new_item = "/" + top
                res.append(new_item)
                if new_item == item:
                    error_fix += 1
            else:
                error_fix += 1
    return " ".join(res), error_fix


def tree_to_sequence(tree_str):
    """
    S2T format
    linearized the phrase tree to token sequences.
    Args:
        tree_str:(TOP (NP (NNP EDUCATION) (NNPS ADS) (: :)))

    Return:
        s2t format:
            words: EDUCATION ADS :
            tokens: NP NNP NNPS : /NP
    """
    stack, tokens, words = [], [], []
    for tok in tree_str.strip().split():
        if len(tok) >= 2:
            if tok[0] == "(":
                symbol = tok[1:]
                tokens.append(symbol)
                stack.append(symbol)
            else:
                assert tok[-1] == ")"
                stack.pop()  # Pop the POS-tag.
                while tok[-2] == ")":
                    tokens.append("/" + stack.pop())
                    tok = tok[:-1]
                words.append(tok[:-1])
    return str.join(" ", words), str.join(" ", tokens[1:-1])  # Strip "TOP" tag.


def sequence_to_tree(s2t_str, word_str=None, split_token=" "):
    """ tree sequence str to tree str """
    tokens = s2t_str.strip("\n").split(split_token)
    word_format = "{})"
    stack = []
    for item in tokens:
        if not item.startswith("/"):
            stack.append((item, False))
        else:
            query = item[1:]
            item_str = ")"
            while len(stack) > 0:
                key = stack.pop(-1)
                if key[0] == query:
                    item_str = "({} {}".format(key[0], item_str)
                    break
                else:
                    if not key[1]:
                        item_str = "({} {} {}".format(key[0], word_format, item_str)
                    else:
                        item_str = "{} {}".format(key[0], item_str)
            stack.append((item_str, True))

    if len(stack) > 1:
        raise RuntimeError("Miss Match of Predict")

    res = "(TOP {})".format(stack[0][0])
    res = res.replace(" )", ")")
    if word_str is not None:
        words = word_str.strip("\n").split(" ")
        res = res.format(*words)
    return res


def tree_to_sequence_without_backtrack(tree_str):
    """
    S2B format
    linearized the phrase tree to token sequences.
    Args:
        tree_str:(TOP (NP (NNP EDUCATION) (NNPS ADS) (: :)))

    Return:
        s2b format:
            words: EDUCATION ADS :
            tokens: NP NNP NNPS : /BRK
    """
    words, tokens = tree_to_sequence(tree_str)
    tokens = replace_backtrack(tokens)
    return words, tokens


def tree_to_sequence_without_tags(tree_str):
    """
    S2S format
    linearized the phrase tree to token sequences.
    Args:
        tree_str:(TOP (NP (NNP EDUCATION) (NNPS ADS) (: :)))

    Return:
        s2S format:
            words: EDUCATION ADS :
            tokens: NP XX XX : /NP
    """
    words, tokens = tree_to_sequence(tree_str)
    tokens = replace_tags(tokens)
    return words, tokens


def add_backtrack(s2b, fm):
    """ convert s2b to s2t format"""
    if isinstance(s2b, str):
        tokens = s2b.strip("\n").split(" ")
    elif isinstance(s2b, list):
        tokens = s2b
    else:
        raise RuntimeError("unknown types:", type(s2b))
    stack = []
    res = []
    error_fix = 0

    for item in tokens:
        if not item.startswith("/"):
            if fm.is_tag(item):
                if len(stack) > 0:
                    res.append(item)
                else:
                    error_fix += 1
            else:
                stack.append(item)
                res.append(item)
        else:
            if len(stack) > 0:
                top = stack.pop(-1)
                new_item = "/" + top
                res.append(new_item)
            else:
                error_fix += 1
    return " ".join(res), error_fix


def replace_backtrack(convert_res):
    tokens = convert_res.split(" ")
    right_bracket = "/BRK"
    new_tokens = [right_bracket if token.startswith("/") else token
                  for token in tokens]
    return " ".join(new_tokens)


def replace_tags(convert_res):
    tokens = convert_res.split(" ")
    sym_set = []
    for token in tokens:
        if token.startswith("/"):
            sym_set.append(token[1:])
            sym_set.append(token)
    sub_tag = "XX"
    new_tokens = [token if token in sym_set else sub_tag
                  for token in tokens]
    return " ".join(new_tokens)


def tree_convert(mode):
    return {
        "s2b": tree_to_sequence_without_backtrack,
        "s2t": tree_to_sequence,
        "s2s": tree_to_sequence_without_tags
    }[mode]
