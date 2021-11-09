import torch.nn.functional as F

from .syntax_gmvae_criterion import SyntaxGMMVAECriterion


class SyntaxTVAECriterion(SyntaxGMMVAECriterion):
    def __init__(self, args, vocab):
        vocab = getattr(vocab, args.vocab_key_words, vocab)
        SyntaxGMMVAECriterion.__init__(self, args, vocab)
        self.c_div = getattr(args, "c_div", 0.0)

    def syn_distribution_loss(self, input_ret):
        bs = input_ret['batch_size']

        z_loss = self.kl_divergence(
            input_ret['syn_mean'], input_ret['syn_logv'].log()
        ) / bs

        c_loss = self.c_prior_loss(
            input_ret['syn_logits'], input_ret['syn_prob_cat']
        ) / bs - self.c_term

        return z_loss, c_loss

    def vae_loss(self, input_ret, step: int = -1):
        input_var = input_ret['src']
        bs = input_ret['batch_size']
        sem_kl = self.kl_divergence(input_ret['sem_mean'], input_ret['sem_logv']) / bs
        z_prior_loss, c_prior_loss = self.syn_distribution_loss(input_ret)

        syn_kl = z_prior_loss * self.z_weight + c_prior_loss * self.c_weight

        kl = (sem_kl * self.kl_sem + syn_kl * self.kl_syn) / (self.kl_sem + self.kl_syn)

        beta = self.kl_weight(step) * self.kl_factor
        beta_kl = kl * beta

        nll_loss = self.reconstruction(
            input_ret['scores'],
            input_var.contiguous()[:, 1:],
            norm_by_word=self.norm_by_word
        ) / bs

        if self.c_div:
            code = input_ret["syn_code"]
            c_div_loss = 1.0 + F.cosine_similarity(
                code[:, None, :], code[None, :, :], dim=-1
            )
            div_loss = c_div_loss.sum() / c_div_loss.size(0)
        else:
            div_loss = 0.0

        ret = {
            'Sem KL': sem_kl,
            'Syn KL': syn_kl,
            'KL': kl,
            'Z-Loss': z_prior_loss,
            'C-Loss': c_prior_loss,
            "DIV-Loss": div_loss,
            'Beta': beta,
            'Beta KL': beta_kl,
            'NLL': nll_loss,
            'ELBO': kl + nll_loss,
            'Beta ELBO': beta_kl + nll_loss,
            'Loss': beta_kl + nll_loss + div_loss * self.c_div,
        }

        return ret

    def printable(self):
        return ["Loss", 'ELBO', 'NLL', 'KL', 'Sem KL', 'Z-Loss', 'C-Loss', "DIV-Loss", 'Beta']
