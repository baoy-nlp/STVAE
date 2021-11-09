from __future__ import print_function
from __future__ import print_function

import pickle
import sys

import numpy as np

from .example import task_to_example


class Dataset(object):
    def __init__(self, examples):
        self.examples = examples
        assert len(examples) > 0, "Empty Dataset?"
        self.fields = examples[0].fields

    def _field_info(self, field: str):
        lengths = [len(getattr(e, field)) for e in self.examples]
        print(
            '%s Max Len: %d \tAvg Len: %.2f' % (
                field.upper(),
                max(lengths),
                float(np.average(lengths))
            ),
            file=sys.stdout
        )

    def info(self):
        for field in self.fields:
            self._field_info(field)
        print(self, file=sys.stdout)

    def __getitem__(self, item):
        return self.examples[item]

    def __repr__(self):
        return "= Dataset[{}]\tSize:{} =".format("-".join(self.fields).upper(), len(self.examples))

    def __sizeof__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def __len__(self):
        return len(self.examples)

    def save(self, fname):
        self.info()
        pickle.dump(self.examples, open(fname, 'wb'))
        print("=== Saved Dataset in {} ===".format(fname), file=sys.stdout)

    @classmethod
    def load(cls, fname, allow_repeat=True):
        examples = pickle.load(open(fname, 'rb'))
        res_examples = []
        repeat_count = 0
        if not allow_repeat:
            singleton_keys = {}
            for example in examples:
                singleton_key = str(example)
                if singleton_key not in singleton_keys:
                    singleton_keys[singleton_key] = True
                    res_examples.append(example)
                else:
                    repeat_count += 1
        if repeat_count > 0:
            print("remove repeat:{}".format(repeat_count), file=sys.stdout)
            dataset = cls(res_examples)
        else:
            dataset = cls(examples)
        dataset.info()
        return dataset

    @classmethod
    def from_raw_iterable(cls, iterable, example_type='DSS-VAE'):
        load_func = task_to_example[example_type].build
        examples = [load_func(line, fields=task_to_fields[example_type]) for line in iterable]
        return cls(examples)

    @classmethod
    def from_raw_file(cls, fname, example_type="DSS-VAE"):
        with open(fname, "r") as f:
            return cls.from_raw_iterable(f, example_type)


task_to_fields = {
    'RAW': ['src'],
    'DSS-VAE': ['src', 'tgt'],
    "Para-VAE": ['src', 'tgt', 'ref'],
    'Paraphrase': ['src', 'tgt', 'label'],
}
