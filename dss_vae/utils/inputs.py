# coding=utf-8
import numpy as np


def word2id(words, vocab):
    if type(words[0]) == list:
        return [[vocab[w] for w in s] for s in words]
    else:
        return [vocab[w] for w in words]


def id2word(ids, vocab):
    if type(ids[0]) == list:
        return [robust_id2word(s, vocab) for s in ids]
    else:
        return robust_id2word(ids, vocab)


def robust_id2word(ids, vocab):
    res = []
    for w in ids:
        if w == vocab.sos_id or w == vocab.pad_id:
            pass
        elif w == vocab.eos_id:
            break
        else:
            res.append(vocab.id2word[w])
    return res


def padding_input(words, pad_token="<pad>", tgt_len=-1):
    if tgt_len == -1:
        tgt_len = max(len(s) for s in words)
    batch_size = len(words)
    seqs = []
    for i in range(batch_size):
        seqs.append(words[i][0:tgt_len] + [pad_token] * (tgt_len - len(words[i])))
    return seqs


def to_target_word(log_prob, vocab):
    _, word_ids = log_prob.sort(dim=-1, descending=True)
    word_ids = word_ids[:, :, 0].data.tolist()
    return [[[id2word(sents, vocab)], [-1]] for sents in word_ids]


def data_to_word(tensor, vocab):
    word_ids = tensor.squeeze(1).data.tolist()
    return [[[id2word(sents, vocab)], [-1]] for sents in word_ids]


def pair_to_inputs(examples, vocab, device=None, use_tgt=True):
    if not isinstance(examples, list):
        examples = [examples]

    word_vocab = getattr(vocab, 'src', vocab)
    origin_batch = [e.src for e in examples]
    origins = word_vocab.process(
        minibatch=origin_batch,
        device=device
    )
    paras = None
    if use_tgt:
        try:
            para_batch = [e.tgt for e in examples]
            paras = word_vocab.process(
                minibatch=para_batch,
                device=device
            )
        except AttributeError:
            paras = None

    return {
        'src': origins,
        'tgt': paras
    }


def str_to_numpy(numpy_str):
    return np.fromstring(numpy_str[1:-1], dtype=np.float32, sep=" ")


def load_whole_numpy(file):
    numpy_array = []
    with open(file, 'r') as f:
        _array = [str_to_numpy(line) for line in f.readlines()]
        numpy_array.extend(_array)
    return np.stack(numpy_array)


def load_whole_tensor(file):
    n = load_whole_numpy(file)
    import torch
    t = torch.from_numpy(n)
    if torch.cuda.is_available():
        t = t.cuda()
    return t
