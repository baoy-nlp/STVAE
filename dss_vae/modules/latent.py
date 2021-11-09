import torch
import torch.nn as nn

from .gaussian_mixture import GaussianMixtureLatent
from .utils import reparameterize


class SingleLatentConstructor(nn.Module):
    """
    vanilla VAE 's latent variable
    """

    def __init__(self, args):
        super(SingleLatentConstructor, self).__init__()
        self.d_hidden = args.d_hidden
        self.latent_size = args.latent_size
        self.batch_first = args.batch_first
        self.enc_factor = (2 if args.bidir else 1) * args.enc_num_layers
        self.hidden2mean = nn.Linear(self.d_hidden * self.enc_factor, args.latent_size)
        self.hidden2logv = nn.Linear(self.d_hidden * self.enc_factor, args.latent_size)
        self.latent2decode = nn.Linear(args.latent_size, self.d_hidden * self.enc_factor)

    def map_to_latent(self, input_ret, is_sampling=True):
        hidden = input_ret['hidden']
        if not self.batch_first:
            hidden = hidden.permute(1, 0, 2)

        hidden = hidden.contiguous().view(-1, self.d_hidden * self.enc_factor)

        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        input_ret['mean'] = mean
        input_ret['logv'] = logv

        z = self.sample_latent(input_ret) if is_sampling else mean
        input_ret['z'] = z
        return input_ret

    def reconstruct(self, input_ret):
        z = input_ret['z']
        hidden = self.latent2decode(z)
        if self.enc_factor > 1:
            hidden = hidden.view(-1, self.enc_factor, self.d_hidden)
        if not self.batch_first:
            hidden = hidden.permute(1, 0, 2)
        input_ret['decode_init'] = hidden
        return input_ret

    def sample_latent(self, input_ret, mode="posterior", beam_size=1, **kwargs):
        batch_size = input_ret['batch_size']
        if mode == "prior":
            z = torch.randn([batch_size * beam_size, self.latent_size])
            if torch.cuda.is_available():
                z = z.cuda()
        elif mode == "posterior":
            if beam_size > 1 or "z" not in input_ret:
                mean, logv = input_ret['mean'], input_ret['logv']
                z = reparameterize(mean, logv, is_logv=True, sample_size=beam_size)
        elif mode == "map":
            z = input_ret["mean"]
        else:
            z = input_ret["z"]
            # raise RuntimeError("mode for sampling is not valid")

        input_ret["z"] = z
        return input_ret["z"]

    def forward(self, input_ret, is_sampling):
        input_ret = self.map_to_latent(input_ret, is_sampling=is_sampling)
        input_ret = self.reconstruct(input_ret)
        return input_ret


class SyntaxLatentConstructor(SingleLatentConstructor):
    """
    disentangle one latent variable to two latent variable: one for semantic, one for syntax
    """

    def __init__(self, args):
        super(SyntaxLatentConstructor, self).__init__(args)
        delattr(self, 'hidden2mean')
        delattr(self, 'hidden2logv')
        delattr(self, 'latent2decode')

        var_dim = self.d_hidden * self.enc_factor // 2

        syn_mlp = nn.Sequential(
            nn.Linear(var_dim, self.latent_size * 2, True),
            nn.ReLU()
        )
        self.syn_mean = nn.Sequential(
            syn_mlp,
            nn.Linear(self.latent_size * 2, self.latent_size)
        )
        self.syn_logv = nn.Sequential(
            syn_mlp,
            nn.Linear(self.latent_size * 2, self.latent_size)
        )

        sem_mlp = nn.Sequential(
            nn.Linear(var_dim, self.latent_size * 2, True),
            nn.ReLU()
        )
        self.sem_mean = nn.Sequential(
            sem_mlp,
            nn.Linear(self.latent_size * 2, self.latent_size)
        )
        self.sem_logv = nn.Sequential(
            sem_mlp,
            nn.Linear(self.latent_size * 2, self.latent_size))

        self.syn_to_h = nn.Linear(self.latent_size, var_dim)
        self.sem_to_h = nn.Linear(self.latent_size, var_dim)

        self.output_size = var_dim

    def map_to_latent(self, input_ret, is_sampling=True):
        hidden = input_ret['hidden']
        bs = input_ret['batch_size']
        if not self.batch_first:
            hidden = hidden.permute(1, 0, 2)

        def split_hidden(encode_hidden):
            factor = encode_hidden.size(1)
            hid = encode_hidden.contiguous().view(bs, factor, 2, -1)
            return hid[:, :, 0, :].contiguous().view(bs, -1), hid[:, :, 1, :].contiguous().view(bs, -1)

        sem_hid, syn_hid = split_hidden(hidden)

        input_ret['enc_sem'] = sem_hid
        input_ret['enc_syn'] = syn_hid

        sem_mean = self.sem_mean(sem_hid)
        sem_logv = self.sem_logv(sem_hid)
        syn_mean = self.syn_mean(syn_hid)
        syn_logv = self.syn_logv(syn_hid)

        input_ret['sem_mean'] = sem_mean
        input_ret['sem_logv'] = sem_logv
        input_ret['syn_mean'] = syn_mean
        input_ret['syn_logv'] = syn_logv

        input_ret['sem_z'] = self.sample_latent(input_ret, keys='sem') if is_sampling else input_ret['sem_mean']
        input_ret['syn_z'] = self.sample_latent(input_ret, keys='syn') if is_sampling else input_ret['syn_mean']
        return input_ret

    def reconstruct(self, input_ret):
        batch_size = input_ret['batch_size']

        def reshape(_hidden):
            if self.enc_factor > 1:
                _hidden = _hidden.view(batch_size, self.enc_factor, self.d_hidden // 2)

            if not self.batch_first:
                _hidden = _hidden.permute(1, 0, 2)

            return _hidden

        sem_hid = reshape(self.sem_to_h(input_ret['sem_z']))
        syn_hid = reshape(self.syn_to_h(input_ret['syn_z']))
        input_ret['sem_hid'] = sem_hid
        input_ret['syn_hid'] = syn_hid
        input_ret['decode_init'] = torch.cat([sem_hid, syn_hid], dim=-1)

        return input_ret

    def sample_latent(self, input_ret, keys='sem'):
        batch_size = input_ret['batch_size']
        z = torch.randn([batch_size, self.latent_size])
        if torch.cuda.is_available():
            z = z.cuda()

        if '{}_mean'.format(keys) in input_ret:
            mean = input_ret['{}_mean'.format(keys)]
            logv = input_ret['{}_logv'.format(keys)]
            std = torch.exp(0.5 * logv)
            z = z * std + mean

        input_ret["{}_z".format(keys)] = z
        return z


class GMMLatentConstructor(SingleLatentConstructor):
    def __init__(self, args):
        super(GMMLatentConstructor, self).__init__(args)
        delattr(self, 'hidden2mean')
        delattr(self, 'hidden2logv')
        delattr(self, 'latent2decode')

        self.constructor = GaussianMixtureLatent(
            input_dim=self.d_hidden * self.enc_factor,
            latent_dim=self.latent_size,
            num_components=args.num_components,
            hidden_dim=self.d_hidden
        )

    def map_to_latent(self, input_ret, is_sampling=True, temperature=1.0, hard=0):
        hidden = input_ret['hidden']
        batch_size = input_ret['batch_size']
        if not self.batch_first:
            hidden = hidden.permute(1, 0, 2)

        hidden = hidden.contiguous().view(batch_size, self.d_hidden * self.enc_factor)
        inf_ret = self.constructor.posterior(hidden, temperature, hard)

        for key, value in inf_ret.items():
            input_ret[key] = value

        input_ret['z'] = self.sample_latent(inf_ret) if is_sampling else inf_ret['mean']

        return input_ret

    def reconstruct(self, input_ret):
        z = input_ret['z']
        batch_size = input_ret['batch_size']
        hidden = self.constructor.prior_net.reconstruct_to_input(z)
        if self.enc_factor > 1:
            hidden = hidden.view(batch_size, self.enc_factor, self.d_hidden)
        if not self.batch_first:
            hidden = hidden.permute(1, 0, 2)
        input_ret['decode_init'] = hidden
        return input_ret

    def sample_latent(self, input_ret):
        if 'z' in input_ret:
            return input_ret['z']

        batch_size = input_ret['batch_size']
        z = self.constructor.prior(batch_size)
        input_ret['z'] = z

        return z

    def forward(self, input_ret, is_sampling, temperature=1.0, hard=0):
        input_ret = self.map_to_latent(input_ret, is_sampling, temperature, hard)
        input_ret = self.reconstruct(input_ret)
        return input_ret


class SyntaxGMMLatentConstructor(SyntaxLatentConstructor):
    def __init__(self, args):
        super(SyntaxGMMLatentConstructor, self).__init__(args)
        delattr(self, 'syn_mean')
        delattr(self, 'syn_logv')
        delattr(self, 'syn_to_h')

        input_dim = self.d_hidden * self.enc_factor // 2
        self.num_components = args.num_components
        self.syn_constructor = GaussianMixtureLatent(
            input_dim=input_dim,
            latent_dim=self.latent_size,
            num_components=args.num_components,
            hidden_dim=self.d_hidden
        )

    def map_to_latent(self, input_ret, is_sampling=True, temperature=1.0, hard=0):
        hidden = input_ret['hidden']
        bs = input_ret['batch_size']
        if not self.batch_first:
            hidden = hidden.permute(1, 0, 2)

        def split_hidden(encode_hidden):
            factor = encode_hidden.size(1)
            hid = encode_hidden.contiguous().view(bs, factor, 2, -1)
            return hid[:, :, 0, :].contiguous().view(bs, -1), hid[:, :, 1, :].contiguous().view(bs, -1)

        sem_hid, syn_hid = split_hidden(hidden)

        input_ret['enc_sem'] = sem_hid
        input_ret['enc_syn'] = syn_hid

        sem_mean = self.sem_mean(sem_hid)
        sem_logv = self.sem_logv(sem_hid)
        input_ret['sem_mean'] = sem_mean
        input_ret['sem_logv'] = sem_logv
        sem_z = self.sample_latent(input_ret, keys='sem')
        input_ret['sem_z'] = sem_z

        syn_inf_ret = self.syn_constructor.posterior(syn_hid.contiguous(), temperature, hard)
        for key, value in syn_inf_ret.items():
            input_ret['syn_{}'.format(key)] = value

        return input_ret

    def reconstruct(self, input_ret):
        batch_size = input_ret['batch_size']

        def reshape(_hidden):
            if self.enc_factor > 1:
                _hidden = _hidden.view(batch_size, self.enc_factor, self.d_hidden // 2)

            if not self.batch_first:
                _hidden = _hidden.permute(1, 0, 2)

            return _hidden

        sem_hid = reshape(self.sem_to_h(input_ret['sem_z']))
        syn_hid = reshape(self.syn_constructor.prior_net.reconstruct_to_input(input_ret['syn_z']))
        input_ret['sem_hid'] = sem_hid
        input_ret['syn_hid'] = syn_hid
        input_ret['decode_init'] = torch.cat([sem_hid, syn_hid], dim=-1)

        return input_ret

    def sample_latent(self, input_ret, keys='sem'):
        batch_size = input_ret['batch_size']
        z = torch.randn([batch_size, self.latent_size])
        if torch.cuda.is_available():
            z = z.cuda()
        if keys == "sem" and '{}_mean'.format(keys) in input_ret:
            mean = input_ret['{}_mean'.format(keys)]
            logv = input_ret['{}_logv'.format(keys)]
            std = torch.exp(0.5 * logv)
            z = z * std + mean
        elif keys == 'meta-syn':
            z = self.syn_constructor.meta_prior(batch_size)['z']
        elif keys == "syn":
            z = self.syn_constructor.prior(batch_size)["z"]
        return z

    def forward(self, input_ret, is_sampling, temperature=1.0, hard=0):
        input_ret = self.map_to_latent(input_ret, is_sampling, temperature, hard)
        input_ret = self.reconstruct(input_ret)
        return input_ret
