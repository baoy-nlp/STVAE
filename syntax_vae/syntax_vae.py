from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from syntax_vae.blocks import RNNDecoder
from syntax_vae.dataset import WrapperField as Dictionary
from syntax_vae.vanilla_vae import VAE, GaussianVariable


class BowPredictor(nn.Module):
    """ bag-of-word predictor for semantic information """

    def __init__(self, input_dim, hidden_dim, embed_dim, num_tokens, dropout=0.1, pad_id=-1, out=None):
        super().__init__()
        self.pad_id = pad_id
        self.i2h = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            # nn.LayerNorm(embed_dim, eps=1e-6), # for experiments after SVAE-EXP5
            # nn.Tanh(),
            # nn.Dropout(dropout)
            nn.ReLU()  # for experiments after SVAE-EXP9
        )
        self.h2o = nn.Linear(embed_dim, num_tokens)
        if out is not None:
            self.h2o.weight = out.weight

    def forward(self, inputs, targets):
        """
        bag-of-word prediction

        :param inputs: batch_size, hidden
        :param targets: seq_len, batch_size
        :return:
        """
        hidden = self.i2h(inputs)
        logits = self.h2o(hidden)

        if targets.size(0) != logits.size(0):
            targets = targets.transpose(0, 1)  # batch_size, seq_len

        score = -logits.log_softmax(dim=-1)

        loss = score.gather(dim=-1, index=targets)  # batch_size, seq_len

        mask = targets.ne(self.pad_id).float()
        loss = mask * loss

        return loss.sum()


class SequencePredictor(RNNDecoder):

    def __init__(self, args, embeddings, sos_id, eos_id):
        super().__init__(args, embeddings, sos_id, eos_id)

        hidden_factor = 2 if args.rnn_type == 'lstm' else 1

        if not args.hidden_dim * hidden_factor == args.input_dim:
            self.map_to_input = nn.Linear(args.input_dim, args.hidden_dim * hidden_factor)
        else:
            self.map_to_input = None

    def forward(self, inputs, hidden, pad_id=-1):
        prev_token = inputs[:-1, :]
        target = inputs[1:, :].contiguous().view(-1)
        logits = super().forward(prev_token, hidden)
        nll_loss = F.cross_entropy(logits.view(target.size(-1), -1), target, ignore_index=pad_id, reduction='sum')
        return nll_loss

    def init_hidden(self, hidden):
        if self.map_to_input is not None:
            hidden = self.map_to_input(hidden)
        return super().init_hidden(hidden)

    @classmethod
    def build_predictor(cls, input_dim, embed_dim, hidden_dim, sos_id, eos_id, pad_id, num_tokens,
                        share_input_output_embed=True,
                        output_without_embed=False, rnn_type="gru",
                        dropout=0.1, out=None):
        args = Namespace()
        args.input_dim = input_dim
        args.embed_dim = embed_dim
        args.hidden_dim = hidden_dim
        args.dropout = dropout
        args.rnn_type = rnn_type
        args.output_without_embed = output_without_embed
        args.share_input_output_embed = share_input_output_embed
        if out is None:
            embeddings = nn.Embedding(
                num_embeddings=num_tokens,
                embedding_dim=args.embed_dim,
                padding_idx=pad_id
            )
        else:
            embeddings = out

        return cls(args, embeddings=embeddings, sos_id=sos_id, eos_id=eos_id)


class SyntaxVAE(VAE):
    def __init__(self, args, vocab: Dictionary, syn_vocab: Dictionary):
        super().__init__(args, vocab)
        self.syn_vocab = syn_vocab

        self.__delattr__("z_module")

        self.sem_latent_dim = args.sem_latent_dim if getattr(args, "sem_latent_dim", True) else args.latent_dim
        self.syn_latent_dim = args.syn_latent_dim if getattr(args, "syn_latent_dim", True) else args.latent_dim
        self.syn_embed_dim = args.syn_embed_dim if getattr(args, "syn_embed_dim", True) else args.embed_dim
        self.syn_hidden_dim = args.syn_hidden_dim if getattr(args, "syn_hidden_dim", True) else args.hidden_dim

        hidden_sem_factor = self.sem_latent_dim * 1.0 / (self.sem_latent_dim + self.syn_latent_dim)
        self.sem_input_dim = int(self.encoder.output_dim * hidden_sem_factor)
        self.sem_output_dim = int(self.input_dim * hidden_sem_factor)

        self.sem_module = GaussianVariable(
            input_dim=self.sem_input_dim,
            latent_dim=self.sem_latent_dim,
            output_dim=self.sem_output_dim
        )

        self.syn_module = GaussianVariable(
            input_dim=self.encoder.output_dim - self.sem_input_dim,
            latent_dim=self.syn_latent_dim,
            output_dim=self.input_dim - self.sem_output_dim
        )

        # used for multi-task loss: predict semantics
        self.multi_sem = BowPredictor(
            input_dim=self.sem_latent_dim,
            hidden_dim=args.hidden_dim,
            embed_dim=self.decoder.embed_dim,
            num_tokens=self.decoder.num_tokens,
            dropout=args.dropout,
            out=self.decoder.embedding
        )

        self.syn_pad_id = syn_vocab.pad_id
        self.syn_sos_id = syn_vocab.sos_id
        self.syn_eos_id = syn_vocab.eos_id
        self.syn_unk_id = syn_vocab.unk_id
        # used for multi-task loss: predict syntax
        self.multi_syn = SequencePredictor.build_predictor(
            input_dim=self.syn_latent_dim,
            embed_dim=self.syn_embed_dim,
            hidden_dim=self.syn_hidden_dim,
            sos_id=self.syn_sos_id,
            eos_id=self.syn_eos_id,
            pad_id=self.syn_pad_id,
            num_tokens=len(syn_vocab),
            share_input_output_embed=getattr(args, "syn_share_input_output_embed", args.share_input_output_embed),
            output_without_embed=getattr(args, "syn_output_without_embed", args.output_without_embed),
            rnn_type=getattr(args, "syn_rnn_type", args.rnn_type),
            dropout=getattr(args, "syn_dropout", args.dropout)
        )

    def _split_hidden(self, hidden):
        # batch_size, hidden_dim
        return hidden[:, :self.sem_input_dim].contiguous(), hidden[:, self.sem_input_dim:].contiguous()

    def _compute_latent(self, hidden, **kwargs):
        sem_h, syn_h = self._split_hidden(hidden)
        sem_z = self.sem_module(sem_h, **kwargs)
        syn_z = self.syn_module(syn_h, **kwargs)
        return sem_z, syn_z

    def forward(self, inputs, unk_rate=0.0, syn_inputs=None, extract_latent=False, **kwargs):
        # encoding
        inputs_token, input_len = self.preprocess(inputs)
        _, encoder_hidden = self.encoder.forward(inputs_token, input_len)

        # split spaces
        sem_z, syn_z = self._compute_latent(encoder_hidden, **kwargs)

        if extract_latent:
            return {
                'sem-z': sem_z,  # Gaussian Latent Variables
                "syn-z": syn_z,  # Gaussian Latent Variables
            }

        # decoding
        rec_hidden = torch.cat([sem_z["rec"], syn_z['rec']], dim=-1)
        prev_token = self.word_drop(inputs, unk_rate)

        # for compute NLL-Loss
        logits = self.decoder.forward(inputs=prev_token, hidden=rec_hidden)

        # multi-task loss for semantics
        multi_sem = self.multi_sem.forward(inputs=sem_z['z'], targets=inputs[1:, :])

        # multi-task loss for syntax
        multi_syn = self.multi_syn.forward(inputs=syn_inputs, hidden=syn_z['z'], pad_id=self.syn_pad_id)

        return {
            'sem-z': sem_z,  # Gaussian Latent Variables
            "syn-z": syn_z,  # Gaussian Latent Variables
            "logits": logits,
            "multi-sem": multi_sem,
            "multi-syn": multi_syn
        }

    @classmethod
    def build(cls, args, vocab, params=None):
        if params is not None:
            args = params['args']
            model = cls(args, vocab["src"], vocab['syn'])
            model.load_state_dict(params['state_dict'])
        else:
            model = cls(args, vocab["src"], vocab['syn'])

        return model
