import numpy as np


def log_normal(x, mu, var, eps):
    if eps > 0.0:
        var = var + eps
    return -0.5 * (np.log(2.0 * np.pi) + var.log() + (x - mu).pow(2) / var).sum(-1)


def gaussian_kl_divergence(z, posterior_mean, posterior_var, prior_mean, prior_var, eps):
    loss = log_normal(z, posterior_mean, posterior_var, eps) - log_normal(z, prior_mean, prior_var, eps)
    return loss.sum()

