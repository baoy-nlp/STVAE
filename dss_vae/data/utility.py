import os

from .dataset import Dataset
from .dictionary import Dictionary
from .vocab import Vocab


def load_docs(fname):
    res = []
    with open(fname, 'r') as data_file:
        for line in data_file:
            line_res = line.strip("\n")
            res.append(line_res)
    return res


def _combine_load_dataset(*fnames):
    """ combine source, target, other structure... """
    raw_datas = [load_docs(fname) for fname in fnames]
    return ["\t".join(raw_data) for raw_data in zip(*raw_datas)]


def _filter_dataset_with_field(dataset: Dataset, field='src', max_len=-1, max_num=-1):
    """ filter data set by condition on specific field """
    examples = dataset.examples
    origin_number = len(examples)
    if max_len > -1:
        res_examples = []
        for e in examples:
            if len(getattr(e, field)) < max_len:
                res_examples.append(e)
        examples = res_examples

    if max_num > -1:
        from random import sample
        train_idx = sample(range(len(examples)), max_num)
        examples = [examples[idx] for idx in train_idx]
    dataset.examples = examples
    print("filtering key:{} with max_len: {}, max_num:{}".format(field, max_len, max_num))
    print("filtering number from {} -> {} ".format(origin_number, len(dataset.examples)))
    return dataset


def _build_vocab_for_field(dataset, field='src', vocab_size=-1, freq_cutoff=-1):
    """ build vocab for specific field """
    corpus = [getattr(e, field) for e in dataset]
    vocab = Dictionary.from_corpus(corpus, vocab_size=vocab_size, freq_cutoff=freq_cutoff)
    return vocab


def _filter_dataset(dataset, max_lens=-1, max_nums=-1):
    fields = dataset.fields
    if not isinstance(max_lens, list):
        max_lens = [max_lens] * len(fields)
    if not isinstance(max_nums, list):
        max_nums = [max_nums] * len(fields)
    for field, max_len, max_num in zip(fields, max_lens, max_nums):
        dataset = _filter_dataset_with_field(dataset, field, max_len, max_num)
    return dataset


def _build_vocab(dataset, vocab_sizes=-1, freq_cutoffs=-1):
    """ build vocab for dataset """
    fields = dataset.fields
    if not isinstance(vocab_sizes, list):
        vocab_sizes = [vocab_sizes] * len(fields)
    if not isinstance(freq_cutoffs, list):
        freq_cutoffs = [freq_cutoffs] * len(fields)
    vocabs = [
        _build_vocab_for_field(dataset, field=field, vocab_size=vocab_size, freq_cutoff=freq_cutoff) for
        field, vocab_size, freq_cutoff in zip(
            dataset.fields,
            vocab_sizes,
            freq_cutoffs
        )
    ]
    entries = {
        field: vocab for field, vocab in zip(fields, vocabs)
    }

    return Vocab(**entries)


def _build_paraphrase_vocab(dataset, vocab_sizes=-1, freq_cutoffs=-1):
    """ build vocab for dataset """
    fields = dataset.fields
    if not isinstance(vocab_sizes, list):
        vocab_sizes = [vocab_sizes] * len(fields)
    if not isinstance(freq_cutoffs, list):
        freq_cutoffs = [freq_cutoffs] * len(fields)

    syn_vocab = _build_vocab_for_field(dataset, field=dataset.fields[1], vocab_size=vocab_sizes[1],
                                       freq_cutoff=freq_cutoffs[1])

    word_corpus = [getattr(e, "src") for e in dataset]
    ref_corpus = [getattr(e, "ref") for e in dataset]

    word_corpus.extend(ref_corpus)
    word_vocab = Dictionary.from_corpus(word_corpus, vocab_size=vocab_sizes[0], freq_cutoff=freq_cutoffs[0])

    entries = {
        'src': word_vocab,
        'tgt': syn_vocab,
        'ref': word_vocab
    }

    return Vocab(**entries)


def _build_dataset(
        fname: str,
        langs=('sent', 's2b'),
        max_lens=-1,
        max_nums=-1,
        task='DSS-VAE',
        destdir=None,
        save_dataset=True,
        save_vocab=False,
        vocab_sizes=(16000, 300),
        freq_cutoffs=-1,
        desc='data',
):
    field_fnames = ["{}.{}".format(fname, lang) for lang in langs]
    rawset = _combine_load_dataset(*field_fnames)
    dataset = Dataset.from_raw_iterable(
        iterable=rawset,
        example_type=task,
    )
    dataset = _filter_dataset(dataset, max_lens, max_nums)
    if save_dataset:
        if destdir is not None:
            if not os.path.exists(destdir):
                os.makedirs(destdir, exist_ok=False)
            # fname = fname.strip().split("/")[-1]
            fname = os.path.join(destdir, desc)
        data_prefix = "-".join(langs)
        dataset.save(fname="{}-{}.bin".format(fname, data_prefix))

    if save_vocab:
        if task == "DSS-VAE":
            vocab = _build_vocab(dataset, vocab_sizes, freq_cutoffs)
        else:
            vocab = _build_paraphrase_vocab(dataset, vocab_sizes, freq_cutoffs)
        if destdir is not None:
            if not os.path.exists(destdir):
                os.makedirs(destdir, exist_ok=False)
        else:
            destdir = os.path.dirname(fname)
        lang_prefix = "-".join(langs)
        vocab.save(fname=os.path.join(destdir, "vocab_{}.bin".format(lang_prefix)))


def build_datasets(
        train_pref='train',
        valid_pref='valid',
        test_pref=None,
        langs=('sent', 's2b'),
        max_lens=-1,
        max_nums=-1,
        task='DSS-VAE',
        destdir=None,
        vocab_sizes=(16000, 300),
        freq_cutoffs=-1
):
    print("=== Build Train Set ===")
    _build_dataset(
        train_pref,
        langs=langs,
        max_lens=max_lens,
        max_nums=max_nums,
        task=task,
        destdir=destdir,
        save_vocab=True,
        vocab_sizes=vocab_sizes,
        freq_cutoffs=freq_cutoffs,
        desc='train'
    )
    if valid_pref is not None:
        print("=== Build Valid Set ===")
        _build_dataset(
            valid_pref,
            langs=langs,
            max_lens=max_lens,
            max_nums=max_nums,
            task=task,
            destdir=destdir,
            desc='dev'
        )
    if test_pref is not None:
        print("=== Build Test Set ===")
        _build_dataset(
            test_pref,
            langs=langs,
            max_lens=max_lens,
            max_nums=max_nums,
            task=task,
            destdir=destdir,
            desc='test'
        )


def _load_dataset(fname, langs=('sent', 's2b')):
    data_prefix = "-".join(langs)
    data_path = "{}-{}.bin".format(fname, data_prefix)
    dataset = Dataset.load(data_path)
    return dataset


def _load_raw_dataset(fnames, task='DSS-VAE', max_lens=-1, max_nums=-1):
    rawset = _combine_load_dataset(*fnames)
    dataset = Dataset.from_raw_iterable(
        iterable=rawset,
        example_type=task
    )
    dataset = _filter_dataset(dataset, max_lens, max_nums)
    return dataset


def is_include_train(mode):
    return mode in ['train', 'extract']


def load_datasets(datadir, langs, vocab_file=None, mode='train'):
    lang_prefix = "-".join(langs)
    if vocab_file is None:
        vocab_file = os.path.join(datadir, "vocab_{}.bin".format(lang_prefix))
    vocab = Vocab.load(fname=vocab_file)
    trainset, validset, testset = None, None, None
    try:
        if is_include_train(mode):
            trainset = _load_dataset(fname=os.path.join(datadir, 'train'), langs=langs)
        validset = _load_dataset(fname=os.path.join(datadir, 'dev'), langs=langs)
        testset = _load_dataset(fname=os.path.join(datadir, 'test'), langs=langs)
    except FileNotFoundError:
        print("Some dataset may not be included")

    return trainset, validset, testset, vocab


def load_raw_datasets(args):
    vocab = Vocab.load(fname=args.vocab_file)
    trainset, validset, testset = None, None, None
    try:
        if is_include_train(args.mode):
            trainset = _load_raw_dataset(
                args.train_files,
                task=args.task,
                max_lens=args.max_lens,
                max_nums=args.max_nums
            )
        validset = _load_raw_dataset(
            args.valid_files,
            task=args.task,
            max_lens=args.max_lens,
            max_nums=args.max_nums
        )
        testset = _load_raw_dataset(
            args.test_files,
            task=args.task,
            max_lens=args.max_lens,
            max_nums=args.max_nums
        )
    except FileNotFoundError:
        print("Some dataset may not be included")

    return trainset, validset, testset, vocab


def prepare_dataset(args):
    data_type = getattr(args, 'data_type', 'bin')
    train, valid, test, _vocab = load_datasets(
        datadir=args.destdir,
        langs=args.langs,
        vocab_file=getattr(args, 'vocab_file', None),
        mode=args.mode
    ) if data_type == 'bin' else load_raw_datasets(args)
    loaded_datasets = {
        'train': train,
        'valid': valid,
        'test': test,
    }
    from dss_vae.data.iterator import make_data_iterators
    _datasets = make_data_iterators(
        datasets=loaded_datasets,
        batch_size=args.batch_size,
        sort_key=args.sort_key,
        mode=args.mode,
        eval_batch_size=args.eval_batch_size,
        update_freq=args.update_freq,
        num_gpus=args.num_gpus
    )
    return _datasets, _vocab
