import torch

from .vae_criterion import VAECriterion


class SyntaxVAECriterion(VAECriterion):
    def __init__(self, args, vocab):
        vocab = getattr(vocab, args.vocab_key_words, vocab)
        super(SyntaxVAECriterion, self).__init__(args, vocab)
        self.kl_sem = args.kl_sem
        self.kl_syn = args.kl_syn

        self.sem_to_sem = args.mul_sem
        self.syn_to_syn = args.mul_syn

        self.sem_to_syn = args.adv_sem
        self.syn_to_sem = args.adv_syn

        self.rec_sem = args.rec_sem
        self.rec_syn = args.rec_syn

    def forward(self, syntax_vae, inputs, step: int = -1, adv_training=False):
        input_var = inputs['src']
        feed_var = self.word_dropout(input_var, step)
        ret = syntax_vae(input_var, feed_var)
        ret.update(**inputs)

        if adv_training:
            # compute the forward loss for training of the adversary
            dis_adv_sem_loss, dis_adv_syn_loss = self.adv(syntax_vae, ret, detach=True)
            dis_rec_sem_loss, dis_rec_syn_loss = self.rec(syntax_vae, ret, detach=True)
            ret['sem_loss'] = dis_adv_sem_loss + dis_rec_sem_loss
            ret['syn_loss'] = dis_adv_syn_loss + dis_rec_syn_loss
            return ret

        with torch.no_grad():
            # compute the adversarial loss of Syntax-VAE 's forward pass
            sem_to_syn, syn_to_sem = self.adv(syntax_vae, ret)
            sem_to_sen, syn_to_sen = self.rec(syntax_vae, ret)
            adv_sem = (sem_to_syn * self.sem_to_syn + sem_to_sen * self.rec_sem)
            adv_syn = (syn_to_sem * self.syn_to_sem + syn_to_sen * self.rec_syn)
            extra_loss = {
                'adv_sem': sem_to_syn,
                'adv_syn': syn_to_sem,
                'rec_sem': sem_to_sen,
                'rec_syn': syn_to_sen,
                'ADV Loss': adv_sem + adv_syn
            }

        # compute the multi-task loss of Syntax-VAE 's forward pass
        mul_sem_loss, mul_syn_loss = self.multi(syntax_vae, ret)
        extra_loss['mul_sem'] = mul_sem_loss
        extra_loss['mul_syn'] = mul_syn_loss
        extra_loss['MUL Loss'] = mul_sem_loss * self.sem_to_sem + mul_syn_loss * self.syn_to_syn

        ret_dict = self.vae_loss(ret, step)
        ret_dict.update(**extra_loss)
        ret_dict['Loss'] = ret_dict['Loss'] + extra_loss['MUL Loss'] - extra_loss['ADV Loss']
        return ret_dict

    def vae_loss(self, input_ret, step: int = -1):
        """ as a normal var with two disentangled latent variable """
        input_var = input_ret['src']
        bs = input_ret['batch_size']
        sem_kl = self.sem_distribution_loss(input_ret)
        syn_kl = self.syn_distribution_loss(input_ret)
        kl = (sem_kl * self.kl_sem + syn_kl * self.kl_syn) / (self.kl_sem + self.kl_syn)

        beta = self.kl_weight(step) * self.kl_factor
        beta_kl = kl * beta
        nll_loss = self.reconstruction(
            input_ret['scores'],
            input_var.contiguous()[:, 1:],
            norm_by_word=self.norm_by_word
        ) / bs

        return {
            'Sem KL': sem_kl,
            'Syn KL': syn_kl,
            'KL': kl,
            'Beta': beta,
            'Beta KL': beta_kl,
            'NLL': nll_loss,
            'ELBO': kl + nll_loss,
            'Beta ELBO': beta_kl + nll_loss,
            'Loss': beta_kl + nll_loss,
        }

    @classmethod
    def bow(cls, logits, tgt_var, pad_ids, logp=True, norm_by_word=False):
        log_prob = logits.log_softmax(dim=-1) if not logp else logits
        loss = -log_prob.gather(1, tgt_var)  # batch_size, seq_len
        mask = tgt_var.ne(pad_ids).float()
        loss = loss * mask
        if norm_by_word:
            loss = loss.sum(dim=-1)
            tgt_len = mask.sum(dim=-1)
            loss = loss / tgt_len
        return loss.sum()

    def multi(self, syntax_vae, input_ret):
        forward_ret = syntax_vae.mul_forward(input_ret, self.sem_to_sem, self.syn_to_syn)
        mul_sem_score = forward_ret['mul_sem']
        sem = self.bow(
            mul_sem_score,
            tgt_var=input_ret['src'].contiguous()[:, 1:],
            pad_ids=self.pad_id,
            logp=True
        ) if mul_sem_score is not None else 0.0

        syn = self.reconstruction(
            forward_ret['mul_syn'],
            tgt_var=input_ret['syn'].contiguous()[:, 1:]
        )
        bs = input_ret['batch_size']
        return sem / bs, syn / bs

    def adv(self, syntax_vae, input_ret, detach=True):
        forward_ret = syntax_vae.adv_forward(input_ret, self.sem_to_syn, self.syn_to_sem, detach=detach)

        syn_var = input_ret['syn'].contiguous()[:, 1:]
        adv_sem = self.reconstruction(forward_ret['adv_sem'], tgt_var=syn_var)

        adv_syn_score = forward_ret['adv_syn']
        adv_syn = self.bow(
            adv_syn_score,
            input_ret['src'].contiguous()[:, 1:],
            pad_ids=self.pad_id,
            logp=True
        ) if adv_syn_score is not None else 0.0
        bs = input_ret['batch_size']
        return adv_sem / bs, adv_syn / bs

    def rec(self, syntax_vae, input_ret, detach=True):
        output = syntax_vae.rec_forward(input_ret, self.rec_sem, self.rec_syn, detach=detach)
        words = input_ret['src'].contiguous()[:, 1:]
        rec_sem_loss = self.reconstruction(output['rec_sem'], tgt_var=words)
        rec_syn_loss = self.reconstruction(output['rec_syn'], tgt_var=words)
        batch_size = input_ret['batch_size']
        return rec_sem_loss / batch_size, rec_syn_loss / batch_size

    def sem_distribution_loss(self, input_ret):
        bs = input_ret['batch_size']
        return self.kl_divergence(input_ret['sem_mean'], input_ret['sem_logv']) / bs

    def syn_distribution_loss(self, input_ret):
        bs = input_ret['batch_size']
        return self.kl_divergence(input_ret['syn_mean'], input_ret['syn_logv']) / bs

    def printable(self):
        return super().printable() + ['Sem KL', 'Syn KL']
