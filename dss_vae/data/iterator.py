from torchtext.data import iterator, batch


class SortedBucketIterator(iterator.BucketIterator):
    """
        :dataset:
        :batch_size:
        :sort_key: None
        :device: None
        :batch_size_fn: None
        :train: True
        :repeat: False
        :shuffle: None
        :sort: None
        :sort_within_batch: None
    """

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                yield minibatch
            if not self.repeat:
                return


class UnsortedBucketIterator(object):
    def __init__(self, dataset, batch_size, batch_size_fn=None):
        super(UnsortedBucketIterator, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.batches = None

    def create_batches(self):
        return batch(self.dataset, self.batch_size, self.batch_size_fn)

    def init_epoch(self):
        self.batches = self.create_batches()

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                yield minibatch
            return


def dyn_batch_with_padding(new, i, sofar):
    prev_max_len = sofar / (i - 1) if i > 1 else 0
    return max(len(new), prev_max_len) * i


def dyn_batch_without_padding(new, i, sofar=0):
    return sofar + len(new)


def batch_add_one(new, i, sofar=0):
    return sofar + 1


def make_data_iterators(datasets, batch_size, sort_key, mode='train', eval_batch_size=-1, update_freq=1, num_gpus=-1):
    trainset = datasets.get('train', None)
    validset = datasets.get('valid', None)
    testset = datasets.get('test', None)

    batch_size_fn = dyn_batch_without_padding if batch_size > 100 else batch_add_one
    if batch_size == 1:
        batch_size_fn = None
    batch_size = int(batch_size / update_freq)
    batch_size = batch_size if batch_size > 1 else 1

    eval_batch_size = batch_size if eval_batch_size < 0 else eval_batch_size

    train_flag = mode == "train"
    repeat_shuffle_flag = not num_gpus > 1  # indicate: distributed training

    train_iterator = SortedBucketIterator(
        dataset=trainset,
        batch_size=batch_size,
        sort_key=sort_key,
        batch_size_fn=batch_size_fn,
        repeat=train_flag,
        shuffle=repeat_shuffle_flag,
    ) if trainset is not None else None

    valid_iterator = UnsortedBucketIterator(
        dataset=validset,
        batch_size=eval_batch_size,
        batch_size_fn=batch_size_fn,
    ) if validset is not None else None

    test_iterator = UnsortedBucketIterator(
        dataset=testset,
        batch_size=eval_batch_size,
        batch_size_fn=batch_size_fn,
    ) if testset is not None else None

    return {
        'train': train_iterator,
        'valid': valid_iterator,
        'test': test_iterator
    }
