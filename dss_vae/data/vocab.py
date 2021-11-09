from __future__ import print_function

import pickle
import sys

from .dictionary import Dictionary


class Vocab(object):
    def __init__(self, **kwargs):
        self.entries = []
        for key, item in kwargs.items():
            assert isinstance(item, Dictionary)
            self.__setattr__(key, item)
            self.entries.append(key)

    def __repr__(self):
        return 'Vocab(%s)' % (', '.join('%s %swords' % (entry, getattr(self, entry)) for entry in self.entries))

    def info(self):
        print(self, file=sys.stdout)

    def save(self, fname, verbose=True):
        if verbose:
            self.info()
            print("=== Saved Vocab in {} ===".format(fname), file=sys.stdout)
        saved_dict = {}
        for entry in self.entries:
            saved_dict[entry] = getattr(self, entry)
        pickle.dump(saved_dict, open(fname, "wb"))

    @staticmethod
    def load(fname, verbose=True):
        saved_dict = pickle.load(open(fname, "rb"))
        vocab = Vocab(**saved_dict)
        if verbose:
            vocab.info()
        return vocab


def load_vocab(fname):
    try:
        vocab = Vocab.load(fname)
    except:
        vocab = Dictionary.load(fname)
    return vocab
