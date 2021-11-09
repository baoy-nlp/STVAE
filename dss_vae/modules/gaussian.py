import torch
import torch.nn as nn

from .utils import reparameterize


class GaussianLatent(nn.Module):
    """
    vanilla VAE 's latent variable
    """

    def __init__(self, input_dim, latent_dim, hidden_dim=512, rec_dim=None):
        super(GaussianLatent, self).__init__()
        rec_dim = rec_dim if rec_dim is not None else input_dim
        self.in_features = input_dim
        self.latent_dim = latent_dim
        self.mean = nn.Linear(self.in_features, self.latent_dim)
        self.logv = nn.Linear(self.in_features, self.latent_dim)

        self.reconstructor = nn.ModuleList([
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rec_dim),
            nn.Sigmoid()
        ])

    def forward(self, inputs, is_sampling=True):
        mean, logv = self.posterior(inputs)  # used for compute KL-Loss
        z = reparameterize(mean, logv, is_logv=True) if is_sampling else mean

        rec = self.reconstruct(z)
        return {
            "mean": mean,
            "logv": logv,
            "z": z,
            'rec': rec
        }

    def posterior(self, hidden):
        mean = self.mean(hidden)
        logv = self.logv(hidden)
        return mean, logv

    def prior(self, batch_size):
        z = torch.randn([batch_size, self.latent_dim])
        if torch.cuda.is_available():
            z = z.cuda()

        return z

    def reconstruct(self, z):
        for layer in self.reconstructor:
            z = layer(z)
        return z

    def sampling(self, pos_ret, mode="prior", beam_size=1):
        """

        :param pos_ret: mean, logv, batch_size
        :param mode:
        :param beam_size:
        :return:
        """
        batch_size = pos_ret['batch_size']
        if mode == "prior":
            return self.prior(batch_size * beam_size)
        elif mode == "posterior":
            if beam_size > 1 or pos_ret["z"] is None:
                mean, logv = pos_ret['mean'], pos_ret['logv']
                return reparameterize(mean, logv, is_logv=True, sample_size=beam_size)
            return pos_ret["z"]
        elif mode == "map":
            return pos_ret['mean']
        else:
            raise RuntimeError("mode for sampling is not valid")
