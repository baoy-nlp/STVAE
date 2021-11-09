import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from dss_vae.modules.constructor import SyntaxVAEConstructor
from dss_vae.modules.gaussian import GaussianLatent
from dss_vae.modules.layers import GumbelSoftmax, Gaussian
from dss_vae.modules.utils import reparameterize
from .syntax_gm_vae import SyntaxGMVAE


class STVAE(SyntaxGMVAE):
    def __init__(self, args, vocab, embed=None, name='Syntax-Template-VAE'):
        super(STVAE, self).__init__(args, vocab, embed, name)
        self.latent_constructor = STVAEConstructor(args)

        self.hard_gumbel = args.hard_gumbel  # used for template
        self.gumbel_temp = args.init_temp

    def extract_template_codes(self):
        return self.latent_constructor.extract_codes()

    def extract_latent(self, examples, use_tgt=False):
        net_input = self.example_to_input(examples, use_tgt=use_tgt)
        ret = self.input_to_posterior(net_input['src'], is_sampling=False)
        ret['sem'] = ret['sem_mean']
        ret['syn'] = ret['syn_mean']
        return ret


class STVAEConstructor(SyntaxVAEConstructor):
    def __init__(self, args):
        super(STVAEConstructor, self).__init__(args)
        delattr(self, 'syn_net')

        self.num_components = args.num_components
        self.syn_net = TwoStageLatent(
            input_dim=self.output_size,
            latent_dim=self.latent_size,
            num_components=self.num_components,
            hidden_dim=2 * self.latent_size
        )

    def _map_to_syn(self, input_ret, is_sampling, temperature=1.0, hard=0):
        syn_ret = self.syn_net.posterior(
            input_ret['enc_syn'].contiguous(),
            input_ret["sem_mean"].contiguous(),
            temperature=temperature,
            hard=hard > np.random.rand()
        )
        for key, value in syn_ret.items():
            input_ret['syn_{}'.format(key)] = value
        return input_ret  # include syn_z, syn_mean, syn_logv

    def forward(self, input_ret, is_sampling, temperature=1.0, hard=0):
        input_ret = self.map_to_latent(input_ret, is_sampling, temperature, hard)
        input_ret = self.reconstruct(input_ret)
        return input_ret


class TwoStageLatent(GaussianLatent):
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

    def sampling(self, pos_ret, mode="prior", beam_size=1):
        batch_size = pos_ret['batch_size']
        if mode == "prior":
            c = self.forward_t(batch_size * beam_size)
            d = self.forward_d(batch_size * beam_size)
        elif mode == "map":
            c = pos_ret["c"]  # maximize for c
            d = pos_ret['mean']  # maximize for mean
        elif mode == "posterior":  # posterior for c, posterior for d
            c = pos_ret["c"]
            if beam_size > 1:
                c = c.unsqueeze(1).expand(batch_size, beam_size, -1).reshape(batch_size * beam_size, -1)
            mean, logv = pos_ret["mean"], pos_ret["logv"]
            d = reparameterize(mean, logv, is_logv=True, sample_size=beam_size)
        elif mode == "prior-map":
            c = self.forward_t(batch_size * beam_size)
            s = pos_ret["s"]  # batch_size, dimension
            if beam_size > 1:
                # match to template
                s = s.unsqueeze(1).expand(batch_size, beam_size, -1).reshape(batch_size * beam_size, -1)
            d = self.infer_d(s=s, c=c)[0]
        elif mode == "posterior-map":
            prob = pos_ret['prob']
            cat = torch.distributions.Categorical(prob).sample(sample_shape=(beam_size,)).transpose(0, 1).reshape(-1)
            c = F.one_hot(cat, self.num_components).float()  # batch_size*beam_size, num_components
            s = pos_ret["s"]
            if beam_size > 1:
                s = s.unsqueeze(1).expand(batch_size, beam_size, -1).reshape(batch_size * beam_size, -1)
            d = self.infer_d(s=s, c=c)[0]
        # elif mode == "topk-map":
        else:
            prob = pos_ret['prob']
            cat = prob.topk(beam_size)[1]  # beam_size, batch_size
            c = F.one_hot(cat, self.num_components).float().reshape(batch_size * beam_size, -1)
            # batch_size*beam_size, num_components
            s = pos_ret["s"]
            if beam_size > 1:
                s = s.unsqueeze(1).expand(batch_size, beam_size, -1).reshape(batch_size * beam_size, -1)
            d = self.infer_d(s=s, c=c)[0]

        z = self.forward_z(d=d, c=c)
        return z

    def posterior(self, syn, sem=None, temperature=1.0, hard=False):
        assert sem is not None, "s can not be the None"
        syn = syn.reshape(syn.size(0), -1)

        posterior_ret = self.posterior_net(syn, sem, temperature, hard)
        prior_ret = self.prior_net(posterior_ret['d'], posterior_ret['categorical'])

        output = posterior_ret
        for key, value in prior_ret.items():
            output[key] = value
        output["s"] = sem
        return output

    def prior(self, batch_size):
        t = self.forward_t(batch_size)  # uniform prior for template t
        s = self.forward_d(batch_size)  # gaussian prior for syntactic s

        output = self.prior_net(s, t)
        output["d"] = s
        return output

    def extract_codes(self):
        return self.prior_net.extract_codes()

    def extract_templates(self, x, temperature=1.0):
        x = x.reshape(x.size(0), -1)
        logits, prob, c = self.infer_t(x, temperature, hard=True)
        return c

    def reconstruct(self, z):
        return self.prior_net.reconstruct_x(z)

    def forward_t(self, num):
        return self.prior_net.forward_t(num)

    def forward_d(self, num):
        return self.prior_net.forward_d(num)

    def forward_z(self, d, c):
        return self.prior_net.forward_z(d, c)

    def infer_d(self, s, c):
        return self.posterior_net.infer_d(s, c)


class Posterior(nn.Module):
    """
    posterior: infer t, infer s
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

        # posterior for s, condition on inputs and t
        self.s_nets = nn.ModuleList([
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Gaussian(hidden_dim, latent_dim)
        ])

    def forward(self, x, s, codes, temperature=1.0, hard=False):

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

    def infer_t(self, x, temperature, hard):
        n_layers = len(self.t_nets)
        output = x
        for i, layer in enumerate(self.t_nets):
            output = layer(output, temperature, hard) if i == n_layers - 1 else layer(output)
        logits, prob, c = output
        return logits, prob, c

    def infer_d(self, s, c):
        """
        infer z from c
        """
        output = torch.cat((s, c), dim=1)
        for i, layer in enumerate(self.s_nets):
            if i == len(self.s_nets) - 1:
                output = layer(output, is_logv=True)
            else:
                output = layer(output)
        mean, logv, d = output
        return mean, logv, d

    def extract_t(self, s, c):
        output = torch.cat((s, c), dim=1)
        hybrid_output = self.s_nets[0](output)
        return hybrid_output


class Prior(nn.Module):
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

    def reconstruct_x(self, s):
        # reconstruct inputs x
        output = s
        for layer in self.reconstructor:
            output = layer(output)
        return output

    def forward_t(self, batch_size):
        # uniform prior
        t = F.one_hot(torch.randint(0, self.num_components, (batch_size,)), self.num_components).float()
        if torch.cuda.is_available():
            t = t.cuda()
        return t

    def forward_d(self, batch_size):
        # random sample d
        d = torch.randn([batch_size, self.latent_dim])
        if torch.cuda.is_available():
            d = d.cuda()
        return d

    def forward_z(self, d, c):
        # template + d: means the syntax prior is dominated by the standard gaussian sampling.
        z = self.codes_t(c) + d
        return z

    def forward(self, d, c):

        z = self.codes_t(c) + d
        x_rec = self.reconstruct_x(z)

        output = {'rec': x_rec, "z": z}
        return output
