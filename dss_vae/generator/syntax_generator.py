import time

import gc
import torch

from dss_vae.utils.inputs import pair_to_inputs
from .vae_generator import VAEGenerator


class SyntaxGenerator(VAEGenerator):

    def syntactic_transferring(self, testset):
        """ for syntax-vae and its variants """
        device = torch.device(0) if torch.cuda.is_available() else None
        sum_time = 0
        model = self.model
        vocab = model.vocab
        predicts, ref_ori, ref_tgt = [], [], []
        for batch in testset:
            torch.cuda.empty_cache()
            gc.collect()
            inputs = pair_to_inputs(batch, vocab=vocab, device=device)
            start_time = time.time()
            predict_ids = model.transfer_predict(inputs, trans_key='syn', to_word=False)
            sum_time += (time.time() - start_time)

            predict = model.to_word(predict_ids)
            predicts.extend(predict)

            ref_syn = [" ".join(e.src) for e in batch]
            ref_ori.extend(ref_syn)

            ref_sem = [" ".join(e.tgt) for e in batch]
            ref_tgt.extend(ref_sem)

        return ref_ori, ref_tgt, predicts, sum_time

    def paraphrasing(self, testset, keys="syn", mode='posterior', sample_size=1, **kwargs):
        test_time = 0
        model = self.model
        results, reference_ori, reference_tgt = [], [], []
        for batch in testset:
            torch.cuda.empty_cache()
            gc.collect()
            start_time = time.time()
            predict_ids = model.generating_for_paraphrase(
                batch,
                to_word=False,
                keys=keys,
                mode=mode,
                sample_size=sample_size
            )  # batch_size * sample_size,
            test_time += (time.time() - start_time)
            res = model.to_word(predict_ids)
            results.extend(res)
            reference_ori.extend([" ".join(e.src) for e in batch])
            reference_tgt.extend([" ".join(e.tgt) for e in batch])

        if sample_size > 1:
            predict_lists = [[] for _ in range(len(reference_ori))]
            for ids, predict in enumerate(results):
                predict_lists[ids // sample_size].append(predict)
            results = predict_lists

        return reference_ori, reference_tgt, results, test_time

    def gmm_paraphrasing(self, testset, **kwargs):
        """ meta paraphrasing for GMM models """
        ref_ori, ref_tgt, predicts, eval_time = self.paraphrasing(testset, keys='meta-syn')
        num_components = len(predicts) // len(ref_ori)
        predict_lists = [[] for _ in range(len(ref_ori))]
        for ids, predict in enumerate(predicts):
            predict_lists[ids // num_components].append(predict)
        return ref_ori, ref_tgt, predict_lists, eval_time
