from .gmvae_criterion import GMMVAECriterion
from .syntax_vae_criterion import SyntaxVAECriterion


class SyntaxGMMVAECriterion(SyntaxVAECriterion, GMMVAECriterion):
    def __init__(self, args, vocab):
        vocab = getattr(vocab, args.vocab_key_words, vocab)
        GMMVAECriterion.__init__(self, args, vocab)
        SyntaxVAECriterion.__init__(self, args, vocab)

    def syn_distribution_loss(self, input_ret):
        bs = input_ret['batch_size']
        z = input_ret['syn_z']
        mu, var = input_ret['syn_mean'], input_ret['syn_var']
        z_mu, z_var = input_ret['syn_z_mean'], input_ret['syn_z_var']
        c_logits, c_prob = input_ret['syn_logits'], input_ret['syn_prob_cat']
        z_prior_loss = self.z_prior_loss(z, mu, var, z_mu, z_var) / bs
        c_prior_loss = self.c_prior_loss(c_logits, c_prob) / bs - self.c_term
        return z_prior_loss, c_prior_loss

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

        return {
            'Sem KL': sem_kl,
            'Syn KL': syn_kl,
            'KL': kl,
            'Z-Loss': z_prior_loss,
            'C-Loss': c_prior_loss,
            'Beta': beta,
            'Beta KL': beta_kl,
            'NLL': nll_loss,
            'ELBO': kl + nll_loss,
            'Beta ELBO': beta_kl + nll_loss,
            'Loss': beta_kl + nll_loss,
        }

    def forward(self, model, inputs, step: int = -1, adv_training=False):
        self.gumbel_decay(model, step)
        return SyntaxVAECriterion.forward(self, model, inputs, step, adv_training)

    def printable(self):
        return ['ELBO', 'NLL', 'KL', 'Beta', 'Sem KL', 'Z-Loss', 'C-Loss']
