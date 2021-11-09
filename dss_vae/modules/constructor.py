import numpy as np
import torch

from .conditional_gaussian import ConditionalGaussian
from .gaussian import GaussianLatent
from .gaussian_mixture import GaussianMixtureLatent
from .latent import SingleLatentConstructor
from .template import Template


class SyntaxVAEConstructor(SingleLatentConstructor):
    """
    disentangle one latent variable to two latent variable: one for semantic, one for syntax
    """

    def __init__(self, args):
        super(SyntaxVAEConstructor, self).__init__(args)
        delattr(self, 'hidden2mean')
        delattr(self, 'hidden2logv')
        delattr(self, 'latent2decode')

        input_dim = self.d_hidden * self.enc_factor // 2

        self.sem_net = GaussianLatent(
            input_dim=input_dim,
            latent_dim=self.latent_size,
            hidden_dim=2 * self.latent_size
        )
        self.syn_net = GaussianLatent(
            input_dim=input_dim,
            latent_dim=self.latent_size,
            hidden_dim=2 * self.latent_size
        )

        self.output_size = input_dim

    def init_inputs(self, inputs):
        hidden = inputs['hidden']
        bsize = inputs['batch_size']
        if not self.batch_first:
            hidden = hidden.permute(1, 0, 2)

        factor = hidden.size(1)
        inputs = hidden.contiguous().view(bsize, factor, 2, -1)
        return inputs[:, :, 0, :].contiguous().view(bsize, -1), inputs[:, :, 1, :].contiguous().view(bsize, -1)

    def _map_to_sem(self, input_ret, is_sampling=True):
        input_ret['sem_mean'], input_ret['sem_logv'] = self.sem_net.posterior(input_ret['enc_sem'])
        input_ret['sem_z'] = self.sample_latent(input_ret, keys='sem', mode='posterior' if is_sampling else "map")
        return input_ret

    def _map_to_syn(self, input_ret, is_sampling, temperature=1.0, hard=0):
        input_ret['syn_mean'], input_ret['syn_logv'] = self.syn_net.posterior(input_ret['enc_syn'])
        input_ret['syn_z'] = self.sample_latent(input_ret, keys='syn', mode='posterior' if is_sampling else "map")
        return input_ret

    def map_to_latent(self, input_ret, is_sampling=True, temperature=1.0, hard=0):
        """ mapping hidden to two separated latent variables"""
        input_ret['enc_sem'], input_ret['enc_syn'] = self.init_inputs(input_ret)
        input_ret = self._map_to_sem(input_ret, is_sampling)
        input_ret = self._map_to_syn(input_ret, is_sampling, temperature, hard)
        return input_ret

    def reconstruct(self, input_ret):

        def reconstruct(inputs):
            if self.enc_factor > 1:
                inputs = inputs.view(-1, self.enc_factor, self.d_hidden // 2)

            if not self.batch_first:
                inputs = inputs.permute(1, 0, 2)

            return inputs

        input_ret['sem_hid'] = reconstruct(self.sem_net.reconstruct(input_ret['sem_z']))
        input_ret['syn_hid'] = reconstruct(self.syn_net.reconstruct(input_ret['syn_z']))
        input_ret['decode_init'] = torch.cat([input_ret['sem_hid'], input_ret['syn_hid']], dim=-1)
        return input_ret

    def sample_latent(self, input_ret, keys='sem', mode='posterior', sample_size=1):

        sampler = self.syn_net if keys == "syn" else self.sem_net

        z = sampler.sampling(
            pos_ret={
                "mean": input_ret['{}_mean'.format(keys)] if '{}_mean'.format(keys) in input_ret else None,
                "logv": input_ret['{}_logv'.format(keys)] if '{}_logv'.format(keys) in input_ret else None,
                "var": input_ret['{}_var'.format(keys)] if '{}_var'.format(keys) in input_ret else None,
                "batch_size": input_ret["batch_size"],
                "s": input_ret["sem_mean"],
                "z": input_ret['{}_z'.format(keys)] if '{}_z'.format(keys) in input_ret else None,
                # for template
                "c": input_ret['{}_categorical'.format(keys)] if '{}_categorical'.format(keys) in input_ret else None,
                "prob": input_ret['{}_prob_cat'.format(keys)] if '{}_prob_cat'.format(keys) in input_ret else None,
            },
            mode=mode,
            beam_size=sample_size
        )

        input_ret["{}_z".format(keys)] = z
        return z


class SyntaxTVAEConstructor(SyntaxVAEConstructor):
    def __init__(self, args):
        super(SyntaxTVAEConstructor, self).__init__(args)
        delattr(self, 'syn_net')

        self.num_components = args.num_components
        self.syn_net = Template(
            input_dim=self.output_size,
            latent_dim=self.latent_size,
            num_components=self.num_components,
            hidden_dim=2 * self.latent_size
        )

    def _map_to_syn(self, input_ret, is_sampling, temperature=1.0, hard=0):
        syn_ret = self.syn_net.posterior(
            syn=input_ret['enc_syn'].contiguous(),
            sem=input_ret["sem_mean"].contiguous(),
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


class SyntaxCVAEConstructor(SyntaxTVAEConstructor):
    # Syntactic condition on semantics
    # intuitive setting of dependencies hypothesis of semantic and syntax
    # prior for syntax: p(syn|semantic) --- N(\mu,\sigma^{2})
    # posterior for syntax: q(syn|semantic,x) --- N(\mu,\sigma^{2})
    def __init__(self, args):
        super(SyntaxTVAEConstructor, self).__init__(args)
        self.syn_net = ConditionalGaussian(
            input_dim=self.output_size,
            latent_dim=self.latent_size,
            hidden_dim=2 * self.latent_size
        )

    def _map_to_syn(self, input_ret, is_sampling, temperature=1.0, hard=0):
        syn_ret = self.syn_net.posterior(
            input_ret['enc_syn'].contiguous(),
            input_ret["sem_mean"].contiguous(),
        )  # sampling based on the semantic representation
        # q(syn|semantic,x) is gaussian distributions

        for key, value in syn_ret.items():
            input_ret['syn_{}'.format(key)] = value
        return input_ret


class SyntaxGMVAEConstructor(SyntaxVAEConstructor):
    # GMM for Syntactic
    # prior for syntax: c ~ uniform, p(syn|c) --- N(\mu,\sigma^{2})
    # posterior for syntax: c ~ q(c|x), q(syn|c,x) --- N(\mu,\sigma^{2})

    def __init__(self, args):
        super(SyntaxGMVAEConstructor, self).__init__(args)
        delattr(self, 'syn_net')

        input_dim = self.d_hidden * self.enc_factor // 2
        self.num_components = args.num_components
        self.syn_net = GaussianMixtureLatent(
            input_dim=input_dim,
            latent_dim=self.latent_size,
            num_components=args.num_components,
            hidden_dim=self.d_hidden
        )

    def _map_to_syn(self, input_ret, is_sampling, temperature=1.0, hard=0):
        syn_ret = self.syn_net.posterior(input_ret['enc_syn'].contiguous(), temperature, hard)
        for key, value in syn_ret.items():
            input_ret['syn_{}'.format(key)] = value
        return input_ret

    def forward(self, input_ret, is_sampling, temperature=1.0, hard=0):
        input_ret = self.map_to_latent(input_ret, is_sampling, temperature, hard)
        input_ret = self.reconstruct(input_ret)
        return input_ret
