import time

import gc
import torch

from dss_vae.model_utils import update_detached_logs


class VAEGenerator(object):
    def __init__(self, model, vocab=None, args=None):
        self.model = model
        self.vocab = vocab
        self.args = args

    def reconstruct(self, testset, **kwargs):
        eval_time = 0
        model = self.model
        predicts, reference = [], []
        for batch in testset:
            torch.cuda.empty_cache()
            gc.collect()
            start_time = time.time()
            predict_ids = model.predict(examples=batch, to_word=False)
            eval_time += (time.time() - start_time)
            predict = model.to_word(predict_ids)
            predicts.extend(predict)
            ref = [" ".join(e.src) for e in batch]
            reference.extend(ref)

        return reference, predicts, eval_time

    def elbo(self, iterator, criterion, step=-1):
        eval_dict = {}
        eval_time = 0
        model = self.model
        for batch in iterator:
            torch.cuda.empty_cache()
            gc.collect()
            start_time = time.time()
            input_var = model.example_to_input(batch)
            forward_ret = criterion(model, input_var, step=step)
            eval_time += (time.time() - start_time)
            eval_dict = update_detached_logs(forward_ret, eval_dict)
        eval_dict['EVAL TIME'] = eval_time
        return eval_dict

    def generate(self, sample_size):
        eval_time = 0
        model = self.model
        predicts = []
        cum_size = 0
        batch_size = 32
        while cum_size < sample_size:
            torch.cuda.empty_cache()
            gc.collect()
            start_time = time.time()
            predict_ids = model.generating_from_prior(batch_size, to_word=False)
            eval_time += (time.time() - start_time)
            predict = model.to_word(predict_ids)
            predicts.extend(predict)
            cum_size += batch_size
        return predicts[:sample_size], eval_time

    def paraphrasing(self, testset, **kwargs):
        eval_time = 0
        model = self.model
        predicts, reference = [], []
        for batch in testset:
            torch.cuda.empty_cache()
            gc.collect()
            start_time = time.time()
            predict_ids = model.generating_from_posterior(examples=batch, to_word=False)
            eval_time += (time.time() - start_time)
            predict = model.to_word(predict_ids)
            predicts.extend(predict)
            ref = [" ".join(e.src) for e in batch]
            reference.extend(ref)
        return reference, predicts, eval_time
