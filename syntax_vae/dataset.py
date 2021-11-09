import os
from contextlib import ExitStack

import six
import torch
from torchtext.data import Example, Field
from torchtext.datasets import TranslationDataset


class GenericDataset(TranslationDataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path=None, exts=None, fields=None,
                 load_dataset=False, prefix='', examples=None, **kwargs):
        if examples is None:
            assert len(exts) == len(fields), 'N parallel dataset must match'
            self.N = len(fields)

            paths = tuple(os.path.expanduser(path + x) for x in exts)
            if load_dataset and (os.path.exists(path + '.processed.{}.pt'.format(prefix))):
                examples = torch.load(path + '.processed.{}.pt'.format(prefix))
            else:
                examples = []
                with ExitStack() as stack:
                    files = [stack.enter_context(open(fname)) for fname in paths]
                    for lines in zip(*files):
                        lines = [line.strip() for line in lines]
                        if not any(line == '' for line in lines):
                            examples.append(Example.fromlist(lines, fields))
                if load_dataset:
                    torch.save(examples, path + '.processed.{}.pt'.format(prefix))

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, exts, fields, path=None, root='.data',
               train='train', validation='val', test='test', **kwargs):
        if path is None:
            path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)
        return train_data, val_data, test_data


class WrapperField(Field):
    @property
    def sos_id(self):
        return self.vocab.stoi[self.init_token]

    @property
    def eos_id(self):
        return self.vocab.stoi[self.eos_token]

    @property
    def unk_id(self):
        return self.vocab.stoi[self.unk_token]

    @property
    def pad_id(self):
        return self.vocab.stoi[self.pad_token]

    def __len__(self):
        return len(self.vocab)

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab.stoi.get(x, self.unk_id) for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi.get(x, self.unk_id) for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)
        else:
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explicitly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [numericalization_func(x) if isinstance(x, six.string_types)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

        var = torch.tensor(arr, dtype=self.dtype, device=device)

        if self.sequential and not self.batch_first:
            var.t_()
        if self.sequential:
            var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var


def get_dataset(task, path, share_vocab=False, vocab_size=30000, length=1000, min_freq=1,
                splits=("train", "valid", "test"),
                logger=None):
    exts_dict = {
        "vae": ("src",),
        "syntax-vae": ("src", "syn"),
        "paraphrase": ("src", "tgt"),
        "paraphrase-syn": ("src", "tgt", "syn"),
    }

    src = WrapperField(init_token='<INIT>', eos_token="<EOS>", unk_token="<UNK>", pad_token="<PAD>",
                       batch_first=False)
    tgt, syn = None, None
    ext_names = exts_dict[task]

    fields = [src]

    if "tgt" in ext_names:
        tgt = WrapperField(init_token='<INIT>', eos_token="<EOS>", unk_token="<UNK>", pad_token="<PAD>",
                           batch_first=False) if not share_vocab else src
        fields.append(tgt)

    if "syn" in ext_names:
        syn = WrapperField(init_token='<INIT>', eos_token="<EOS>", unk_token="<UNK>", pad_token="<PAD>",
                           batch_first=False)

        fields.append(syn)
    prefix = "-".join(ext_names)
    prefix += "_{}".format(length)
    fields = tuple([(name, field) for name, field in zip(ext_names, fields)])
    exts = tuple([".{}.txt".format(name) for name in ext_names])

    trainset, validset, testset = GenericDataset.splits(
        exts=exts, fields=fields, path=path, train=splits[0], validation=splits[1], test=splits[2],
        load_dataset=True, prefix=prefix,
        filter_pred=lambda x: len(x.src) < length
    )

    vocab_path = path + "/" + "vocab_{}_{}.pt".format(prefix, vocab_size)

    if os.path.exists(vocab_path):
        vocabs = torch.load(vocab_path)
        if logger is not None:
            logger.info("Load vocab from {}".format(vocab_path))
    else:
        if trainset is None:
            raise RuntimeError("no trainset used for initial the vocab")

        src.build_vocab(trainset, min_freq=min_freq, max_size=vocab_size)
        vocabs = [src.vocab]
        if tgt is not None:
            tgt.build_vocab(trainset, min_freq=min_freq, max_size=vocab_size)
            vocabs.append(tgt.vocab)
        if syn is not None:
            syn.build_vocab(trainset, min_freq=min_freq, max_size=vocab_size)
            vocabs.append(syn.vocab)
        torch.save(vocabs, vocab_path)

        if logger is not None:
            logger.info("Save vocab to {}".format(vocab_path))

    for vocab, (name, field) in zip(vocabs, fields):
        field.vocab = vocab

    return trainset, validset, testset


def dyn_batch_with_padding(new, i, sofar):
    prev_max_len = sofar / (i - 1) if i > 1 else 0
    return max(len(new.src), prev_max_len) * i


def dyn_batch_without_padding(new, i, sofar):
    return sofar + len(new.src)


def batch_with_sentences(new, i, sofar):
    return sofar + 1
