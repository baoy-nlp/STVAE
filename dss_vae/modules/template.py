import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Categorical

from .gaussian import GaussianLatent
from .layers import GumbelSoftmax, Gaussian
from .utils import reparameterize


class Posterior(nn.Module):
    """
    posterior: infer s, infer delta s, infer sem
    """

    def __init__(self, input_dim, latent_dim, num_components, hidden_dim=512):
        super(Posterior, self).__init__()

        # posterior for t
        self.t_nets = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            GumbelSoftmax(hidden_dim, num_components)
        ])

        # posterior for z, condition on inputs and t
        self.d_nets = nn.ModuleList([
            nn.Linear(latent_dim + num_components, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Gaussian(hidden_dim, latent_dim)
        ])

    def infer_t(self, x, temperature, hard):
        """
        infer t from x
        """
        num_layers = len(self.t_nets)
        output = x
        for i, layer in enumerate(self.t_nets):
            if i == num_layers - 1:  # last layer is gumbel softmax
                output = layer(output, temperature, hard)
            else:
                output = layer(output)
        logits, prob, c = output
        return logits, prob, c

    def infer_d(self, s, c):
        """
        infer z from c
        """
        output = torch.cat((s, c), dim=1)
        for i, layer in enumerate(self.d_nets):
            if i == len(self.d_nets) - 1:
                output = layer(output, is_logv=True)
            else:
                output = layer(output)
        mean, logv, d = output
        return mean, logv, d

    def forward(self, x, s, temperature=1.0, hard=False):

        logits, prob, c = self.infer_t(x, temperature, hard)
        mean, logv, d = self.infer_d(s, c)

        output = {
            'mean': mean,
            'logv': logv,
            'd': d,
            'logits': logits,
            'prob_cat': prob,
            'categorical': c,
            't_score': prob.topk(1)[0],
            't_num': prob.topk(1)[1]
        }
        return output


class Prior(nn.Module):
    """
    uniform prior for c
    gaussian prior for z
    """

    def __init__(self, input_dim, latent_dim, num_components, hidden_dim=512, rec_dim=None):
        super(Prior, self).__init__()

        rec_dim = rec_dim if rec_dim is not None else input_dim
        self.num_components = num_components
        self.latent_dim = latent_dim
        self.codes_t = nn.Linear(num_components, latent_dim)  # template codes

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

    def reconstruct(self, z):
        # reconstruct inputs x
        output = z
        for layer in self.reconstructor:
            output = layer(output)
        return output

    def forward_t(self, batch_size):
        # random sample template
        c = F.one_hot(torch.randint(0, self.num_components, (batch_size,)), self.num_components).float()
        # uniform prior
        if torch.cuda.is_available():
            c = c.cuda()
        return c

    def forward_s(self, batch_size):
        # random sample d
        d = torch.randn([batch_size, self.latent_dim])
        if torch.cuda.is_available():
            d = d.cuda()
        return d

    def forward_syn(self, d, c):
        # template + d: means the syntax prior is dominated by the standard gaussian sampling.
        z = self.codes_t(c) + d
        return z

    def forward(self, d, c):
        z = self.forward_syn(d, c)
        x = self.reconstruct(z)
        output = {'rec': x, "z": z}
        return output


class Template(GaussianLatent):
    """
                            s: semantic
                t: syntax_template     d: delta_syntax
    prior           uniform               N(0,I)
    posterior        q(t|x)              q(d|t,s)

    syn = syn_t + delta_syn

    """

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

    def reconstruct(self, z):
        return self.prior_net.reconstruct(z)

    def posterior(self, syn, sem=None, temperature=1.0, hard=False):
        assert sem is not None, "s can not be the None"
        syn = syn.reshape(syn.size(0), -1)

        ret = self.posterior_net(syn, sem, temperature, hard)
        prior_ret = self.prior_net(ret['d'], ret['categorical'])
        ret.update(prior_ret)
        ret["s"] = sem
        ret["code"] = self.extract_codes()
        return ret

    def prior(self, batch_size):
        c = self.forward_t(batch_size)  # uniform prior for template t
        d = self.forward_d(batch_size)  # gaussian prior for syntactic s

        output = self.prior_net(d, c)
        output["d"] = d
        return output

    def sampling(self, ret, mode="prior", beam_size=1):
        # posterior c + prior d
        bs = ret['batch_size']
        if mode == "prior":
            t = self.forward_t(bs * beam_size)
            d = self.forward_d(bs * beam_size)
        elif mode == "map":
            t = ret["c"]  # maximize for c
            d = ret['mean']  # maximize for mean
        elif mode == "posterior":  # posterior for c, posterior for d
            t = ret["c"]
            if beam_size > 1:
                t = t.unsqueeze(1).expand(bs, beam_size, -1).reshape(bs * beam_size, -1)
            mean, logv = ret["mean"], ret["logv"]
            d = reparameterize(mean, logv, is_logv=True, sample_size=beam_size)
        elif mode == "prior-map":
            t = self.forward_t(bs * beam_size)

            s = ret["s"]  # batch_size, dimension
            if beam_size > 1:
                # match to template
                s = s.unsqueeze(1).expand(bs, beam_size, -1).reshape(bs * beam_size, -1)
            d = self.infer_d(s=s, c=t)[0]
        elif mode == "posterior-map":
            prob = ret['prob']
            cat = Categorical(prob).sample(sample_shape=(beam_size,)).transpose(0, 1).reshape(-1)
            t = F.one_hot(cat, self.num_components).float()  # batch_size*beam_size, num_components

            s = ret["s"]
            if beam_size > 1:
                s = s.unsqueeze(1).expand(bs, beam_size, -1).reshape(bs * beam_size, -1)
            d = self.infer_d(s=s, c=t)[0]
        # elif mode == "topk-map":
        else:
            prob = ret['prob']
            cat = prob.topk(beam_size)[1]  # beam_size, batch_size
            t = F.one_hot(cat, self.num_components).float().reshape(bs * beam_size, -1)
            # batch_size*beam_size, num_components

            s = ret["s"]
            if beam_size > 1:
                s = s.unsqueeze(1).expand(bs, beam_size, -1).reshape(bs * beam_size, -1)
            d = self.infer_d(s=s, c=t)[0]

        z = self.forward_z(d=d, c=t)
        return z

    def infer_d(self, s, c):
        return self.posterior_net.infer_d(s, c)

    def forward_t(self, num):
        return self.prior_net.forward_t(num)

    def forward_d(self, num):
        return self.prior_net.forward_s(num)

    def forward_z(self, d, c):
        return self.prior_net.forward_syn(d, c)

    def extract_codes(self):
        return self.prior_net.codes_t.weight

    def extract_templates(self, x, temperature=1.0):
        x = x.reshape(x.size(0), -1)
        logits, prob, c = self.posterior_net.infer_t(x, temperature, hard=True)
        return c
