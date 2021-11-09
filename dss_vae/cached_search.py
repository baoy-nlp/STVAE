import numpy as np
import torch
import torch.nn.functional as F

from .utils.inputs import str_to_numpy


def file_batcher(filename, batch_size=1000):
    with open(filename, "r") as f:
        batch = []
        while True:
            line = f.readline()
            if not line:
                break
            batch.append(line)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if len(batch) >= 0:
            yield batch


def batch_to_tensor(batch):
    numpy_array = np.stack([str_to_numpy(numpy_str) for numpy_str in batch])
    t = torch.from_numpy(numpy_array)
    if torch.cuda.is_available():
        t = t.cuda()
    return t


def compute_score(query, key):
    batch_size = query.size(0)
    key_extend = key.unsqueeze(0).expand(batch_size, -1, -1)
    query_extend = query.unsqueeze(1)
    cosine_score = F.cosine_similarity(query_extend, key_extend, 2)
    return 1 - cosine_score


def combine_query(current_ids, current_scores, new_scores, new_start_ids, topk):
    batch_size, candidate_size = new_scores.size()

    candidate_ids = (torch.arange(candidate_size) + new_start_ids).view(1, -1).expand(batch_size, -1)

    if current_ids is not None:
        new_ids = torch.cat([current_ids, candidate_ids])
    else:
        new_ids = candidate_ids.cuda() if torch.cuda.is_available() else candidate_ids

    if current_scores is not None:
        new_scores = torch.cat([current_scores, new_scores])

    topk_scores, topk_ids = new_scores.topk(topk)

    return topk_scores, new_ids.gather(1, topk_ids)


def search_candidate(query_file, key_file, out_file, topk=100):
    with open(out_file, "w") as f:
        for input_batch in file_batcher(query_file, 10):
            query_tensor = batch_to_tensor(input_batch)
            start_ids = 0
            current_ids, current_scores = None, None
            for key_batch in file_batcher(key_file, topk * 10):
                key_tensor = batch_to_tensor(key_batch)
                new_scores = compute_score(query_tensor, key_tensor)
                current_scores, current_ids = combine_query(current_ids, current_scores, new_scores, start_ids, topk)
                start_ids += len(key_batch)
            for ids in current_ids.cpu().detach().numpy():
                ids = np.array2string(ids, max_line_width=300)
                f.write(ids)
                f.write("\n")
    return out_file


def filter_candidate(query_file, key_file, candidate_file, out_file, iter_size=10):
    for query_batch, candidate in zip(file_batcher(query_file, iter_size), file_batcher(candidate_file, iter_size)):
        pass
