import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .gaussian import GaussianLatent
from .layers import GumbelSoftmax, Gaussian

"""
prior: w, z
    w ~ N(0,I)
    c ~ Cat(1/K)

common:
    x ~ Mixture of Gaussian Network's Model + c

posterior:
    w ~ \phi_w(y): y -> w
    x ~ \phi_x(y): y -> x

GMM Latent:
    y -> x

"""


class Posterior(nn.Module):
    def __init__(self, input_dim, latent_dim, num_components, hidden_dim=512):
        super(Posterior, self).__init__()

        # q(c|x)
        self.inference_qyx = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            GumbelSoftmax(hidden_dim, num_components)
        ])

        # q(z|c,x)
        self.inference_qzyx = nn.ModuleList([
            nn.Linear(input_dim + num_components, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Gaussian(hidden_dim, latent_dim)
        ])

    # q(c|x)
    def c_posterior(self, x, temperature, hard):
        num_layers = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                # last layer is gumbel softmax
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x

    # q(z|c,x)
    def z_posterior(self, x, c):
        concat = torch.cat((x, c), dim=1)
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat

    def forward(self, x, temperature=1.0, hard=0):
        # q(c|x)
        logits, prob, c = self.c_posterior(x, temperature, hard)

        # q(z|x,c)
        z_mean, z_var, z = self.z_posterior(x, c)

        output = {
            'mean': z_mean,
            'var': z_var,
            'z': z,
            'logits': logits,
            'prob_cat': prob,
            'categorical': c
        }
        return output


class Prior(nn.Module):
    def __init__(self, input_dim, latent_dim, num_components, hidden_dim=512, rec_dim=None):
        super(Prior, self).__init__()
        if rec_dim is None:
            rec_dim = input_dim

        # p(z|c)
        self.y_mu = nn.Linear(num_components, latent_dim)
        self.y_var = nn.Linear(num_components, latent_dim)

        # p(x|z)
        self.reconstructor = nn.ModuleList([
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rec_dim),
            nn.Sigmoid()
        ])

    # p(z|c)
    def z_prior(self, c):
        y_mu = self.y_mu(c)
        y_var = F.softplus(self.y_var(c))
        return y_mu, y_var

    # p(x|z)
    def reconstruct(self, z):
        for layer in self.reconstructor:
            z = layer(z)
        return z

    def forward(self, post_z, post_c):
        # p(z|c)
        z_mean, z_var = self.z_prior(post_c)  # prior z

        # p(x|z)
        x_rec = self.reconstruct(post_z)

        output = {
            'z_mean': z_mean,
            'z_var': z_var,
            'rec': x_rec
        }
        return output


class GaussianMixtureLatent(GaussianLatent):
    def __init__(self, input_dim, latent_dim, num_components, hidden_dim=512):
        super(GaussianLatent, self).__init__()
        self.num_components = num_components
        self.in_features = input_dim
        self.latent_dim = latent_dim

        self.posterior_net = Posterior(input_dim, latent_dim, num_components, hidden_dim)
        self.prior_net = Prior(input_dim, latent_dim, num_components, hidden_dim)

        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def posterior(self, hidden, temperature=1.0, hard=0):
        x = hidden.view(hidden.size(0), -1)
        inf_ret = self.posterior_net(x, temperature, hard)
        z, c = inf_ret['z'], inf_ret['categorical']
        gen_ret = self.prior_net(z, c)

        output = inf_ret
        for key, value in gen_ret.items():
            output[key] = value
        return output

    def reconstruct(self, z):
        return self.prior_net.reconstruct(z)

    def prior(self, batch_size):
        categorical = F.one_hot(torch.randint(0, self.num_components, (batch_size,)), self.num_components).float()
        if torch.cuda.is_available():
            categorical = categorical.cuda()

        mean, var = self.prior_net.z_prior(categorical)
        noise = torch.randn_like(var)
        std = torch.sqrt(var)
        z = mean + noise * std
        rec = self.prior_net.reconstruct(z)

        output = {"z": z, "rec": rec}
        return output

    def meta_prior(self, batch_size):
        categorical = F.one_hot(torch.Tensor(range(self.num_components)).long(), self.num_components).float()
        if torch.cuda.is_available():
            categorical = categorical.cuda()

        categorical = categorical.unsqueeze(0).expand(batch_size, -1, -1).contiguous().view(
            batch_size * self.num_components, -1)

        mean, var = self.prior_net.z_prior(categorical)
        noise = torch.randn_like(var)
        std = torch.sqrt(var)
        z = mean + noise * std
        rec = self.prior_net.reconstruct(z)

        output = {"z": z, "rec": rec}
        return output

    def sampling(self, pos_ret, mode="prior", beam_size=1):
        """

        :param pos_ret: mean, var, s, z
        :param mode:
        :param beam_size:
        :return:
        """
        pass
