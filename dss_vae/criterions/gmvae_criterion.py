import numpy as np
import torch.nn.functional as F

from .vae_criterion import VAECriterion


class GMMVAECriterion(VAECriterion):
    def __init__(self, args, vocab):
        super(GMMVAECriterion, self).__init__(args, vocab)

        self.eps = 1e-8
        # append for GMM-VAE

        self.z_weight = args.z_weight
        self.c_weight = args.c_weight
        self.r_weight = args.r_weight

        # gumbel
        self.init_temp = args.init_temp
        self.decay_temp = args.decay_temp
        self.min_temp = args.min_temp
        self.decay_temp_rate = args.decay_temp_rate

        self.decay_every = args.decay_every
        self.verbose = args.verbose

        self.c_term = np.log(1.0 / args.num_components)

    def log_normal(self, x, mu, var):
        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * (np.log(2.0 * np.pi) + var.log() + (x - mu).pow(2) / var).sum(-1)

    def z_prior_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.sum()

    def c_prior_loss(self, w_posterior, w_prior):
        log_q = F.log_softmax(w_posterior + self.eps, dim=-1)
        return (w_prior * log_q).sum()

    def forward(self, model, input_var, step: int = -1, **kwargs):
        self.gumbel_decay(model, step)  # decay gumbel

        feed_var = self.word_dropout(input_var, step)
        input_ret = model(input_var, feed_var)
        bs = input_ret['batch_size']
        z = input_ret['z']
        mu, var = input_ret['mean'], input_ret['var']
        z_mu, z_var = input_ret['z_mean'], input_ret['z_var']
        c_logits, c_prob = input_ret['logits'], input_ret['prob_cat']
        rec_loss = self.reconstruction(input_ret['scores'], input_var[:, 1:], norm_by_word=self.norm_by_word) / bs
        z_prior_loss = self.z_prior_loss(z, mu, var, z_mu, z_var) / bs
        c_prior_loss = self.c_prior_loss(c_logits, c_prob) / bs - self.c_term
        beta = self.kl_weight(step) * self.kl_factor
        loss = self.r_weight * rec_loss + (self.z_weight * z_prior_loss + self.c_weight * c_prior_loss) * beta

        return {
            'Z-Loss': z_prior_loss,
            'C-Loss': c_prior_loss,
            'Beta': beta,
            'NLL': rec_loss,
            'ELBO': rec_loss + z_prior_loss + c_prior_loss,
            'Loss': loss,
        }

    def gumbel_decay(self, model, step):
        if step % self.decay_every != 0:
            return

        epoch = step // self.decay_every

        if self.decay_temp == 1 and model.training:
            model.gumbel_temp = np.maximum(self.init_temp * np.exp(-self.decay_temp_rate * epoch), self.min_temp)
            if self.verbose == 1:
                print("Gumbel Temperature: %.3lf" % model.gumbel_temp)

    def printable(self):
        return ['ELBO', 'NLL', 'Z-Loss', 'C-Loss', 'Beta']
