import torch
import torch.nn as nn

from syntax_vae.blocks import RNNDecoder, RNNEncoder
from syntax_vae.dataset import WrapperField as Vocab


def reparameterize(mean, var, is_logv=False, sample_size=1):
    if sample_size > 1:
        mean = mean.contiguous().unsqueeze(1).expand(-1, sample_size, -1).reshape(-1, mean.size(-1))
        var = var.contiguous().unsqueeze(1).expand(-1, sample_size, -1).reshape(-1, var.size(-1))

    if not is_logv:
        std = torch.sqrt(var + 1e-10)
    else:
        std = torch.exp(0.5 * var)

    noise = torch.randn_like(std)
    z = mean + noise * std
    return z


def word_dropout(inputs, dropoutr, vocab):
    if dropoutr > 0.:
        prob = torch.rand(inputs.size()).to(inputs)
        prob[(inputs.data - vocab.sos_id) * (inputs.data - vocab.pad_id) * (
                inputs.data - vocab.eos_id) == 0] = 1
        replace_inputs = inputs.clone()
        replace_inputs[prob < dropoutr] = vocab.unk_id
        return replace_inputs
    return inputs


class GaussianVariable(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.mean = nn.Linear(input_dim, latent_dim)
        self.logv = nn.Linear(input_dim, latent_dim)
        self.rec = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, output_dim),
            nn.Tanh(),
        )

    def forward(self, inputs, max_posterior=False, **kwargs):
        """
        :param inputs:  batch_size,input_dim
        :param max_posterior:
        :return:
            mean: batch_size, latent_dim
            logv: batch_size, latent_dim
            z: batch_size, latent_dim
            rec: batch_size, output_dim
        """
        mean, logv, z = self.posterior(inputs, max_posterior=max_posterior)

        rec = self.rec(z)

        return {"mean": mean, "logv": logv, "z": z, "rec": rec}

    def posterior(self, inputs, max_posterior=False):
        mean = self.mean(inputs)
        logv = self.logv(inputs)
        z = reparameterize(mean, logv, is_logv=True) if not max_posterior else mean
        return mean, logv, z

    def prior(self, inputs, n=-1):
        if n < 0:
            n = inputs.size(0)
        z = torch.randn([n, self.latent_dim])

        if inputs is not None:
            z = z.to(inputs)

        return z


class VAE(nn.Module):
    def __init__(self, args, vocab: Vocab):
        super().__init__()
        self.args = args
        src_embed = nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=args.embed_dim,
            padding_idx=vocab.pad_id
        )

        self.encoder = RNNEncoder(args, src_embed)

        if args.share_enc_dec_embed:
            tgt_embed = src_embed
        else:
            tgt_embed = nn.Embedding(
                num_embeddings=len(vocab),
                embedding_dim=args.embed_dim,
                padding_idx=vocab.pad_id
            )
        self.decoder = RNNDecoder(args, tgt_embed, sos_id=vocab.sos_id, eos_id=vocab.eos_id)
        self.hidden_factor = 2 if args.rnn_type == "lstm" else 1
        self.input_dim = args.hidden_dim * self.hidden_factor
        # bridge between the encoder and decoder

        self.z_module = GaussianVariable(
            input_dim=self.encoder.output_dim,
            latent_dim=args.latent_dim,
            output_dim=self.input_dim
        )

        self.pad_id = vocab.pad_id
        self.sos_id = vocab.sos_id
        self.eos_id = vocab.eos_id
        self.unk_id = vocab.unk_id

    def preprocess(self, inputs):
        input_len = inputs.ne(self.pad_id).sum(dim=0)
        return inputs, input_len

    def word_drop(self, inputs, unk_rate):
        return word_dropout(inputs[:-1, :], dropoutr=unk_rate, vocab=self)

    def forward(self, inputs, unk_rate=0.0, **kwargs):
        """
        :param inputs:  src_len, batch_size
        :param unk_rate
        :return:
        """
        inputs_token, input_len = self.preprocess(inputs)
        _, encoder_hidden = self.encoder.forward(inputs_token, input_len)

        z_ret = self.z_module.forward(inputs=encoder_hidden, max_posterior=not self.args.kl_factor > 0)

        prev_token = self.word_drop(inputs, unk_rate)

        logits = self.decoder.forward(inputs=prev_token, hidden=z_ret['rec'])

        z_ret['logits'] = logits

        return z_ret

    def saved_params(self):
        params = {
            "name": str(type(self)),
            "args": self.args,
            "state_dict": self.state_dict()
        }
        return params

    @classmethod
    def build(cls, args, vocab, params=None):
        if params is not None:
            args = params['args']
            model = cls(args, vocab["src"])
            model.load_state_dict(params['state_dict'])
        else:
            model = cls(args, vocab["src"])

        return model
