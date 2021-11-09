from __future__ import print_function

from collections import Counter
from itertools import chain

import torch

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'


class Dictionary(object):

    def __init__(self, append_edge=True):
        self.word2id = dict()
        self.word2id[UNK_TOKEN] = 0
        self.word2id[PAD_TOKEN] = 1
        if append_edge:
            self.word2id[SOS_TOKEN] = 2
            self.word2id[EOS_TOKEN] = 3
        self.id2word = {v: k for k, v in self.word2id.items()}

    def reset_objects(self):
        self.id2word = {v: k for k, v in self.word2id.items()}

    @property
    def pad_id(self):
        return self.word2id[PAD_TOKEN]

    @property
    def unk_id(self):
        return self.word2id[UNK_TOKEN]

    @property
    def sos_id(self):
        return self[SOS_TOKEN]

    @property
    def eos_id(self):
        return self[EOS_TOKEN]

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Dictionary:[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word.get(wid, UNK_TOKEN)

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def is_unk(self, word):
        return word not in self

    @staticmethod
    def from_corpus(corpus, vocab_size=-1, freq_cutoff=0):
        if vocab_size == -1:
            vocab_size = 50000
        word_freq = Counter(chain(*corpus))
        non_singletons = [w for w in word_freq if word_freq[w] > 1]
        print('Total Token Types: %d, Token Types w/ Freq > 1: %d' % (len(word_freq), len(non_singletons)))

        dictionary = Dictionary()
        top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)[:vocab_size]

        for word in top_k_words:
            if len(dictionary) < vocab_size:
                if word_freq[word] >= freq_cutoff:
                    dictionary.add(word)

        dictionary.reset_objects()

        return dictionary

    @staticmethod
    def load(fname):
        word2id = dict()
        with open(fname, 'r') as rf:
            for line in rf.readlines():
                key, value = line.strip().split("\t")
                word2id[key] = int(value)

        dictionary = Dictionary()
        dictionary.word2id = word2id
        dictionary.reset_objects()
        return dictionary

    def save(self, fname, **kwargs):
        with open(fname, 'w') as f:
            for key, value in self.word2id.items():
                line = "\t".join([key, str(value)])
                f.write(line)
                f.write('\n')

    def process(self, minibatch, sequential=True, fix_length=None, pad_first=False, truncate_first=False,
                include_lengths=False, append_edge=True,
                dtype=torch.long, device=None):
        padded = self.pad(minibatch, sequential, fix_length, pad_first, truncate_first, include_lengths, append_edge)
        tensor = self.numericalize(padded, include_lengths, dtype=dtype, sequential=sequential, device=device)
        return tensor

    @classmethod
    def pad(cls, minibatch, sequential, fix_length, pad_first, truncate_first, include_lengths, append_edge=True):
        minibatch = list(minibatch)
        init_token = SOS_TOKEN if append_edge else None
        eos_token = EOS_TOKEN if append_edge else None
        if not sequential:
            return minibatch
        if fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = fix_length + (init_token, eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if pad_first:
                padded.append(
                    [PAD_TOKEN] * max(0, max_len - len(x)) +
                    ([] if init_token is None else [init_token]) +
                    list(x[-max_len:] if truncate_first else x[:max_len]) +
                    ([] if eos_token is None else [eos_token]))
            else:
                padded.append(
                    ([] if init_token is None else [init_token]) +
                    list(x[-max_len:] if truncate_first else x[:max_len]) +
                    ([] if eos_token is None else [eos_token]) +
                    [PAD_TOKEN] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if include_lengths:
            return padded, lengths
        return padded

    def numericalize(self, arr, include_lengths, dtype, sequential, batch_first=True, device=None):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        """
        if include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        lengths = None
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype = dtype, device=device)

        if sequential:
            flatten_arr = []
            for ex in arr:
                _arr = [self[x] for x in ex]
                flatten_arr.extend(_arr)
            batch = len(arr)
            arr = flatten_arr
        else:
            arr = [self[x] for x in arr]
            batch = 1
        # TODO: There is error need check
        var = torch.tensor(arr, dtype=dtype, device=device).view(batch, -1)

        if sequential and not batch_first:
            var.t_()
        if sequential:
            var = var.contiguous()

        if include_lengths:
            return var, lengths
        return var
