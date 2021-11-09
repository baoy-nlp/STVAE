import torch
import torch.nn as nn
import torch.nn.functional as F

from .tricks import Trick


class VAECriterion(nn.Module):
    def __init__(self, args, vocab):
        super(VAECriterion, self).__init__()
        self.args = args
        self.tricky = Trick(
            kl_anneal_funcs=args.kl_anneal_funcs,
            wd_anneal_funcs=args.wd_anneal_funcs,
            unk=vocab.unk_id,
            pad=vocab.pad_id,
            sos=vocab.sos_id,
            eos=vocab.eos_id
        )
        self.pad_id = vocab.pad_id
        self.k = args.k
        self.x0 = args.x0

        self.unk_rate = args.unk_rate
        self.norm_by_word = getattr(args, 'norm_by_word', False)
        self.kl_factor = getattr(args, 'kl_factor', 1.0)
        self.NLL = torch.nn.NLLLoss(ignore_index=self.pad_id, reduction="sum")

    def kl_weight(self, step):
        return self.tricky.get_kl_weight(step, self.k, self.x0)

    @classmethod
    def kl_divergence(cls, mean, logv):
        return -0.5 * ((1 + logv - mean.pow(2) - logv.exp()).sum())

    @classmethod
    def cross_entropy(cls, logits, tgt_var, pad_ids=-1, logp=True):
        """

        :param logits: batch_size, seq_len, vocab
        :param tgt_var: batch_size, seq_len
        :param pad_ids: ids
        :param logp: bool, default is True
        :return: batch_size, seq_len
        """
        batch_size, seq_len, _ = logits.size()
        log_prob = logits.log_softmax(dim=-1) if not logp else logits
        loss = F.cross_entropy(
            input=log_prob.contiguous().view(batch_size * seq_len, -1),
            target=tgt_var.contiguous().view(-1),
            ignore_index=pad_ids,
            reduction='none'
        ).contiguous().view(batch_size, seq_len)
        return loss

    def reconstruction(self, scores, tgt_var, norm_by_word=False):
        """
        :param scores: seq_len, batch_size, vocab_size
        :param tgt_var: batch_size, seq_len+1
        :param norm_by_word:
        :return: batched nll loss
        """
        if scores is None:
            return 0.0
        batch_size = tgt_var.size(0)
        if scores.size(1) == batch_size:
            scores = scores.contiguous().permute(1, 0, 2)

        # loss = self.cross_entropy(
        #     logits=scores,
        #     tgt_var=target,
        #     pad_ids=self.pad_id,
        #     logp=False
        # ).sum(dim=-1)

        # if norm_by_word:
        #     tgt_len = target.ne(self.pad_id).sum(dim=-1).float()
        #     loss = loss / tgt_len

        # return loss.sum()

        log_prob = scores.log_softmax(dim=-1).view(-1, scores.size(-1))

        nll_loss = self.NLL(log_prob,tgt_var.contiguous().view(-1))
        return nll_loss

    def word_dropout(self, input_var, step):
        unk_drop_p = self.tricky.get_wd_ratio(alpha=self.unk_rate, step=step, k=self.k, x0=self.x0)
        feed_var = self.tricky.get_dropped_inputs(inputs=input_var, dropoutr=unk_drop_p)
        return feed_var

    def forward(self, model, input_var, step: int = -1, **kwargs):
        feed_var = self.word_dropout(input_var, step)
        ret = model(input_var, feed_var)
        batch_size = ret['batch_size']

        NLL_loss = self.reconstruction(ret['scores'], input_var[:, 1:], norm_by_word=self.norm_by_word) / batch_size
        KL_loss = self.kl_divergence(ret['mean'], ret['logv']) / batch_size

        beta = self.kl_weight(step) * self.kl_factor
        beta_KL = KL_loss * beta

        loss = beta_KL + NLL_loss

        return {
            'KL': KL_loss,
            'Beta': beta,
            'Beta KL': beta_KL,
            'NLL': NLL_loss,
            'ELBO': KL_loss + NLL_loss,
            'Beta ELBO': loss,
            'Loss': loss,
        }

    def printable(self):
        return ['ELBO', 'NLL', 'KL', 'Beta']
