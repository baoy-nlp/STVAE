import torch
import torch.nn as nn
import torch.nn.functional as F

from .syntax_vae import SyntaxVAE
from .vanilla_vae import GaussianVariable


class GumbelSoftmax(nn.Module):
    def __init__(self, f_dim, c_dim, out=None):
        super(GumbelSoftmax, self).__init__()
        self.net = nn.Linear(f_dim, c_dim, bias=False)
        if out is not None:
            self.out = out
        else:
            self.out = None

        self.f_dim = f_dim
        self.c_dim = c_dim

    def output(self, inputs):
        if self.out is not None:
            return F.linear(inputs, self.out.weight.t())
        else:
            return self.net(inputs)

    def forward(self, inputs, tau=1.0, hard=False):
        logits = self.output(inputs).view(-1, self.c_dim)
        z = F.gumbel_softmax(logits, tau, hard)
        prob = F.softmax(logits, dim=-1)
        return logits, prob, z


class UniformVariable(nn.Module):
    def __init__(self, input_dim, latent_dim, num_components, out=None):
        super().__init__()
        self.num_components = num_components
        hidden_dim = latent_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.out = GumbelSoftmax(latent_dim, num_components, out)

    def forward(self, inputs, tau=1.0, hard=False):
        hidden = self.net(inputs)
        logits, prob, z = self.out(hidden, tau, hard)
        return {
            "logits": logits,
            "prob": prob,
            "z": z
        }

    def posterior(self, inputs, tau=1.0, hard=False):
        """
        :return logits, prob, z
        """
        hidden = self.net(inputs)
        return self.out(hidden, tau, hard)

    def prior(self, inputs=None, n=-1):
        """
        :return: z
        """
        if n < 0:
            n = inputs.size(0)
        z = F.one_hot(
            torch.randint(0, self.num_components, (n,)), self.num_components
        ).float()

        if inputs is not None:
            z = z.to(inputs)

        return z


class SyntaxTemplateVariable(GaussianVariable):
    """
    including:
        template t,     prior ~ U(0,num_components) posterior ~ Categorical p(t|x)
        syntax s,       prior ~ N(0,1)              posterior ~ Gaussian (s|t,z)

        t independent to z
        s dependent on t and z

    potential issue:
        s may be ignored since the t could recover the syntax individually. [add ignore_t]

    Our goal is to learn the meaningful template codes, and each template code will capture some abstract
    syntax information. The syntax should be consistent with the semantics and easy to control since its
    condition on the template and semantics.

    """

    def __init__(self, input_dim, latent_dim, output_dim, num_components, ignore_t=False):
        # used for syntax
        super().__init__(latent_dim + num_components, latent_dim, output_dim)

        # used for template
        self.t_codes = nn.Linear(num_components, latent_dim, bias=False)
        self.t_net = UniformVariable(input_dim, latent_dim, num_components, out=self.t_codes)
        self.ignore_t = ignore_t

    def forward(self, inputs, max_posterior=False, sem=None, tau=1.0, hard=False, **kwargs):
        pos_t, pos_s = self.posterior(inputs, max_posterior, sem, tau, hard)
        z, rec = self._rec(t=pos_t["z"], s=pos_s["z"])
        return {
            "pos_t": pos_t,
            "pos_s": pos_s,
            "z": z,
            "rec": rec  # recover for the inputs
        }

    def two_stage_sampling(self, sem, n=-1, max_posterior=True):
        """ prior for template, syn condition on sem and template """
        t = self.prior(inputs=sem, n=n)
        s = self._posterior(sem, t, max_posterior=max_posterior)
        return t, s

    def posterior(self, inputs, max_posterior=False, sem=None, tau=1.0, hard=False):
        # used for training the VAE
        pos_t = self.t_net(inputs, tau, hard)
        t = pos_t["z"]
        pos_s = self._posterior(sem, t, max_posterior=max_posterior)
        return pos_t, pos_s

    def prior(self, inputs, n=-1, sample_t=True):
        if sample_t:
            return self.t_net.prior(inputs, n)
        else:
            return super().prior(inputs, n)

    def _rec(self, t, s):
        if self.ignore_t:
            z = s
        else:
            t_code = self.t_codes(t)
            z = t_code + s  # final variables

        rec = self.rec(z)
        return z, rec

    def _posterior(self, sem, t, max_posterior=False):
        mean, logv, z = super().posterior(inputs=torch.cat([sem, t], dim=-1), max_posterior=max_posterior)
        return {
            "mean": mean,
            "logv": logv,
            "z": z
        }


class SyntaxTVAE(SyntaxVAE):
    """
    Modeling the template of the syntax
    """

    def __init__(self, args, vocab, syn_vocab):
        super().__init__(args, vocab, syn_vocab)
        self.__delattr__("syn_module")

        self.syn_module = SyntaxTemplateVariable(
            input_dim=self.encoder.output_dim - self.sem_input_dim,
            latent_dim=self.syn_latent_dim,
            output_dim=self.input_dim - self.sem_output_dim,
            num_components=getattr(args, "num_components", 20),
            ignore_t=getattr(args, "ignore_t", False)
        )

    def _compute_latent(self, hidden, **kwargs):
        sem_h, syn_h = self._split_hidden(hidden)
        sem_z = self.sem_module(sem_h, **kwargs)
        syn_z = self.syn_module(inputs=syn_h, sem=sem_z["mean"], **kwargs)
        return sem_z, syn_z
