from .syntax_vae_criterion import SyntaxVAECriterion
from .utils import gaussian_kl_divergence


class SyntaxCVAECriterion(SyntaxVAECriterion):
    def __init__(self, args, vocab):
        vocab = getattr(vocab, args.vocab_key_words, vocab)
        self.eps = 1e-8
        SyntaxVAECriterion.__init__(self, args, vocab)

    def syn_distribution_loss(self, input_ret):
        bs = input_ret['batch_size']
        z = input_ret['syn_z']
        mu, var = input_ret['syn_mean'], input_ret['syn_var']
        z_mu, z_var = input_ret['syn_z_mean'], input_ret['syn_z_var']
        z_prior_loss = gaussian_kl_divergence(z, mu, var, z_mu, z_var, self.eps) / bs
        return z_prior_loss
   