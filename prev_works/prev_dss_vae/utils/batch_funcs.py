# MIT License

# Copyright (c) 2018 the NJUNLP groups.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Author baoyu.nlp   
# Time 2019-01-22 15:55

import numpy as np

from .tensor_ops import get_float_tensor
from .tensor_ops import get_long_tensor


def batch_iter(examples, batch_size, shuffle=False):
    index_arr = np.arange(len(examples))
    if shuffle:
        np.random.shuffle(index_arr)

    batch_num = int(np.ceil(len(examples) / float(batch_size)))
    for batch_id in range(batch_num):
        batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
        batch_examples = [examples[i] for i in batch_ids]

        yield batch_examples


def batchify(data_set, data_name, batch_size):
    sents, trees = None, None
    if data_name == 'train':
        idxs, tags, stags, arcs, distances, sents, trees = data_set.train
    elif data_name == 'valid':
        idxs, tags, stags, arcs, distances, _, _ = data_set.valid
    elif data_name == 'test':
        idxs, tags, stags, arcs, distances, _, _ = data_set.test
    else:
        raise RuntimeError('need a correct data_name')

    assert len(idxs) == len(distances)
    assert len(idxs) == len(tags)

    bachified_idxs, bachified_tags, bachified_stags, bachified_arcs, bachified_dsts = [], [], [], [], []
    bachified_sents, bachified_trees = [], []
    for i in range(0, len(idxs), batch_size):
        if i + batch_size >= len(idxs):
            continue

        if sents is not None:
            bachified_sents.append(sents[i: i + batch_size])
            bachified_trees.append(trees[i: i + batch_size])

        extracted_idxs = idxs[i: i + batch_size]
        extracted_tags = tags[i: i + batch_size]
        extracted_stags = stags[i: i + batch_size]

        extracted_arcs = arcs[i: i + batch_size]
        extracted_dsts = distances[i: i + batch_size]

        longest_idx = max([len(i) for i in extracted_idxs])
        longest_arc = longest_idx - 1

        minibatch_idxs, minibatch_tags, minibatch_stags, minibatch_arcs, minibatch_dsts, = [], [], [], [], []
        for idx, tag, stag, arc, dst in zip(extracted_idxs, extracted_tags, extracted_stags,
                                            extracted_arcs, extracted_dsts):
            padded_idx = idx + [-1] * (longest_idx - len(idx))
            padded_tag = tag + [0] * (longest_idx - len(tag))
            padded_stag = stag + [0] * (longest_idx - len(stag))

            padded_arc = arc + [0] * (longest_arc - len(arc))
            padded_dst = dst + [0] * (longest_arc - len(dst))

            minibatch_idxs.append(padded_idx)
            minibatch_tags.append(padded_tag)
            minibatch_stags.append(padded_stag)

            minibatch_arcs.append(padded_arc)
            minibatch_dsts.append(padded_dst)

        minibatch_idxs = get_long_tensor(minibatch_idxs)
        minibatch_tags = get_long_tensor(minibatch_tags)
        minibatch_stags = get_long_tensor(minibatch_stags)

        minibatch_arcs = get_long_tensor(minibatch_arcs)
        minibatch_dsts = get_float_tensor(minibatch_dsts)

        bachified_idxs.append(minibatch_idxs)
        bachified_tags.append(minibatch_tags)
        bachified_stags.append(minibatch_stags)

        bachified_arcs.append(minibatch_arcs)
        bachified_dsts.append(minibatch_dsts)

    if sents is not None:
        return bachified_idxs, bachified_tags, bachified_stags, bachified_arcs, \
               bachified_dsts, bachified_sents, bachified_trees
    return bachified_idxs, bachified_tags, bachified_stags, bachified_arcs, bachified_dsts


def get_syntax(extracted_tags, extracted_arcs, extracted_dsts, longest_idx):
    longest_arc = longest_idx - 1
    minibatch_tags, minibatch_arcs, minibatch_dsts, = [], [], [],
    for idx, tag, stag, arc, dst in zip(extracted_tags, extracted_arcs, extracted_dsts):
        padded_tag = tag + [0] * (longest_idx - len(tag))
        padded_arc = arc + [0] * (longest_arc - len(arc))
        padded_dst = dst + [0] * (longest_arc - len(dst))

        minibatch_tags.append(padded_tag)
        minibatch_arcs.append(padded_arc)
        minibatch_dsts.append(padded_dst)

    minibatch_tags = get_long_tensor(minibatch_tags)
    minibatch_arcs = get_long_tensor(minibatch_arcs)
    minibatch_dsts = get_float_tensor(minibatch_dsts)

    return {
        'tags': minibatch_tags,
        "arcs": minibatch_arcs,
        "dsts": minibatch_dsts
    }
