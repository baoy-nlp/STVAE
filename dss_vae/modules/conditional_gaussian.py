import torch
import torch.nn as nn
import torch.nn.init as init

from .gaussian import GaussianLatent
from .layers import Gaussian
from .utils import reparameterize


class Posterior(nn.Module):
    """
    posterior: infer s, infer delta s, infer sem
    """

    def __init__(self, input_dim, latent_dim, hidden_dim=512):
        super(Posterior, self).__init__()

        # posterior for z, condition on inputs and t
        self.z_nets = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Gaussian(hidden_dim, latent_dim)
        ])

    def infer_z(self, x):
        """
        infer z from z
        """
        output = x
        for layer in self.z_nets:
            output = layer(output)
        return output

    def forward(self, x):
        mean, var, z = self.infer_z(x)

        output = {
            'mean': mean,
            'var': var,
            'z': z,
        }
        return output


class ConditionalPrior(nn.Module):
    """
    uniform prior for c
    gaussian prior for z
    """

    def __init__(self, input_dim, latent_dim, hidden_dim=512, rec_dim=None):
        super(ConditionalPrior, self).__init__()

        rec_dim = rec_dim if rec_dim is not None else input_dim
        self.latent_dim = latent_dim

        self.z_priors = nn.ModuleList([
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Gaussian(hidden_dim, latent_dim)
        ])

        self.reconstructor = nn.ModuleList(
            [
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, rec_dim),
                nn.Sigmoid()
            ]
        )

    def reconstruct_to_input(self, z):
        output = z
        for layer in self.reconstructor:
            output = layer(output)
        return output

    def forward_z(self, s, sample_size=1):
        output = s
        num_layers = len(self.z_priors)
        for i, layer in enumerate(self.z_priors):
            if i != num_layers - 1:
                output = layer(output)
            else:
                output = layer(output, is_logv=False, sample_size=sample_size)
        mean, var, z = output
        return mean, var, z

    def forward(self, s, sample_size=1):
        mean, var, z = self.forward_z(s, sample_size)
        x_rec = self.reconstruct_to_input(z)
        output = {'rec': x_rec, "z_mean": mean, "z_var": var, "z_prior": z}
        return output


class ConditionalGaussian(GaussianLatent):
    """
           sem: semantic, syn: syntactic
    prior            p(syn*|sem)
    posterior         q(syn*|x)
    """

    def __init__(self, input_dim, latent_dim, hidden_dim=512):
        super(GaussianLatent, self).__init__()
        self.in_features = input_dim
        self.latent_dim = latent_dim

        self.posterior_net = Posterior(input_dim, latent_dim, hidden_dim)
        self.prior_net = ConditionalPrior(input_dim, latent_dim, hidden_dim)

        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def reconstruct(self, z):
        return self.prior_net.reconstruct_to_input(z)

    def posterior(self, x, s=None, **kwargs):
        posterior_ret = self.posterior_net(x)
        prior_ret = self.prior_net(s)

        output = posterior_ret
        for key, value in prior_ret.items():
            output[key] = value
        return output

    def prior(self, s, beam_size=1):
        output = self.prior_net(s, sample_size=beam_size)
        output['z'] = output['z_prior']
        return output

    def sampling(self, pos_ret, mode="prior", beam_size=1):
        """
        :param pos_ret: mean, var, s, z
        :param mode:
        :param beam_size:
        :return:
        """
        if mode == "prior":
            s = pos_ret["s"]
            return self.prior(s, beam_size)["z"]
        elif mode == "posterior":
            if beam_size > 1 or pos_ret["z"] is None:
                mean, var = pos_ret["mean"], pos_ret["var"]
                return reparameterize(mean, var, is_logv=False, sample_size=beam_size)
            return pos_ret["z"]
        elif mode == "map":
            return pos_ret['mean']
        else:
            raise RuntimeError("mode for sampling is not valid")
