"""
Linearized tree structure
"""


class ReversibleLinearTree(object):
    """
    include forward label, pos-tags, backward label,
    """
    NAME = "s2t"

    @classmethod
    def linearizing(cls, tree_str):
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
        return str.join(" ", words), str.join(" ", tokens[1:-1])

    @classmethod
    def reversing(cls, tokens, words, split=" ", **kwargs):
        tokens = tokens.strip("\n").split(split)
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
        if words is not None:
            words = words.strip("\n").split(" ")
            res = res.format(*words)
        return res


class TagLinearTree(ReversibleLinearTree):
    NAME = "tag"

    @classmethod
    def postprocess(cls, syn_seq):
        tokens = syn_seq.split(" ")
        syntax_sets = []
        for token in tokens:
            if token.startswith("/"):
                syntax_sets.append(token[1:])
                syntax_sets.append(token)
        tag_sequences = [token for token in tokens if token not in syntax_sets]
        return " ".join(tag_sequences)

    @classmethod
    def linearizing(cls, tree_str):
        words, syn_seqs = super().linearizing(tree_str)
        syn_seqs = cls.postprocess(syn_seqs)
        return words, syn_seqs


class NonBacktrackLinearTree(ReversibleLinearTree):
    """
    include forward label, pos-tags, w/o backward label,
    """

    NAME = "s2b"

    @classmethod
    def postprocess(cls, convert_res):
        tokens = convert_res.split(" ")
        right_bracket = "/BRK"
        new_tokens = [right_bracket if token.startswith("/") else token
                      for token in tokens]
        return " ".join(new_tokens)

    @classmethod
    def linearizing(cls, tree_str):
        words, tokens = super().linearizing(tree_str)
        tokens = cls.postprocess(tokens)
        return words, tokens

    @classmethod
    def add_backtrack(cls, tokens, fm):
        if isinstance(tokens, str):
            tokens = tokens.strip("\n").split(" ")
        elif isinstance(tokens, list):
            tokens = tokens
        else:
            raise RuntimeError("unknown types:", type(tokens))
        stack = []
        res = []
        error_fix_count = 0

        for item in tokens:
            if not item.startswith("/"):
                if fm.is_tag(item):
                    if len(stack) > 0:
                        res.append(item)
                    else:
                        error_fix_count += 1
                else:
                    stack.append(item)
                    res.append(item)
            else:
                if len(stack) > 0:
                    top = stack.pop(-1)
                    new_item = "/" + top
                    res.append(new_item)
                else:
                    error_fix_count += 1
        return " ".join(res), error_fix_count

    @classmethod
    def reversing(cls, tokens, words, split=" ", fm=None, **kwargs):
        tokens, error = cls.add_backtrack(tokens, fm)
        return super().reversing(tokens, words, split)


class NonTagLinearTree(ReversibleLinearTree):
    """
    include forward label, w/o pos-tags, backward label,
    """

    NAME = "s2s"

    @classmethod
    def postprocess(cls, convert_res):
        tokens = convert_res.split(" ")
        sym_set = []
        for token in tokens:
            if token.startswith("/"):
                sym_set.append(token[1:])
                sym_set.append(token)
        sub_tag = "XX"
        new_tokens = [token if token in sym_set else sub_tag for token in tokens]
        return " ".join(new_tokens)

    @classmethod
    def linearizing(cls, tree_str):
        words, tokens = super().linearizing(tree_str)
        tokens = cls.postprocess(tokens)
        return words, tokens


class NonTagBacktrackLinearTree(ReversibleLinearTree):
    @classmethod
    def postprocess(cls, convert_res):
        convert_res = NonTagLinearTree.postprocess(convert_res)
        return NonBacktrackLinearTree.postprocess(convert_res)

    @classmethod
    def linearizing(cls, tree_str):
        words, tokens = super().linearizing(tree_str)
        tokens = cls.postprocess(tokens)
        return words, tokens


tree_converter_cls = {
    ReversibleLinearTree.NAME: ReversibleLinearTree,  # s2t
    NonBacktrackLinearTree.NAME: NonBacktrackLinearTree,  # s2b
    NonTagLinearTree.NAME: NonTagLinearTree,  # s2s
    TagLinearTree.NAME: TagLinearTree  # tag
}
