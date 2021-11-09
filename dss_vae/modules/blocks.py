import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dss_vae.data.dictionary import Dictionary
from dss_vae.modules.embeddings import Embeddings
from dss_vae.modules.rnn import WrapperRNN


def reflect(x: torch.Tensor, dim=-1):
    return x


class MLPBridger(nn.Module):
    def __init__(self, rnn_type, enc_dim, enc_layer, dec_dim, dec_layer, batch_first=False):
        super(MLPBridger, self).__init__()
        self.rnn_type = rnn_type
        self.dec_dim = dec_dim
        self.dec_layer = dec_layer

        input_dim = enc_dim * enc_layer
        output_dim = dec_dim * dec_layer
        if input_dim == output_dim:
            self.mapper = reflect
        else:
            self.mapper = nn.Linear(in_features=input_dim, out_features=output_dim)

        self.output_dim = output_dim
        self.batch_first = batch_first

    def forward(self, inputs):
        if self.batch_first:
            batch_size, nums, hid_size = inputs.size()
            inputs = inputs.contiguous().view(batch_size, -1)
        else:
            nums, batch_size, hid_size = inputs.size()
            inputs = inputs.permute(1, 0, 2).contiguous().view(batch_size, -1)

        outputs = self.mapper(inputs).contiguous().view(batch_size, self.dec_layer, self.dec_dim)
        if self.batch_first:
            return outputs.permute(1, 0, 2).contiguous()
        return outputs.contiguous()


class RNNEncoder(WrapperRNN):
    def __init__(self, vocab: Dictionary, input_size, hidden_size, embed_droprate, rnn_droprate, num_layers, bidir,
                 rnn_cell, batch_first=True, embedding=None, update_embedding=True):
        super(RNNEncoder, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidir,
            rnn_cell=rnn_cell,
            dropout=rnn_droprate,
            batch_first=batch_first
        )
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = Embeddings(
                num_embeddings=len(vocab),
                embedding_dim=input_size,
                dropout=embed_droprate,
                add_position_embedding=False,
                padding_idx=vocab.pad_id,
            )
        self.embedding.weight.requires_grad = update_embedding

    @classmethod
    def build_module(cls, args, vocab, embed=None):
        return cls(
            vocab=vocab,
            input_size=args.d_model,
            hidden_size=args.d_hidden,
            embed_droprate=args.drop_embed,
            rnn_droprate=args.dropout,
            num_layers=args.enc_num_layers,
            bidir=args.bidir,
            rnn_cell=args.rnn_cell,
            embedding=embed,
        )

    def forward(self, input_var, mask=None):
        input_embedding = self.embedding(input_var)
        return super(RNNEncoder, self).forward(inputs=input_embedding, mask=mask)


class RNNDecoder(RNNEncoder):
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab: Dictionary, max_len, input_size, hidden_size, embed_droprate=0, rnn_droprate=0,
                 num_layers=1, rnn_cell='gru', embedding=None, update_embedding=True,
                 init_use='enc',
                 use_last_output=True):
        super(RNNDecoder, self).__init__(
            vocab=vocab, input_size=input_size, hidden_size=hidden_size, embed_droprate=embed_droprate,
            rnn_droprate=rnn_droprate, num_layers=num_layers, bidir=False, batch_first=True, rnn_cell=rnn_cell,
            embedding=embedding,
            update_embedding=update_embedding
        )
        self.max_len = max_len
        self.use_last_output = use_last_output
        soft_input_size = hidden_size + input_size if use_last_output else hidden_size
        self.projection = nn.Linear(soft_input_size, len(vocab))
        self.sos_id = vocab.sos_id
        self.eos_id = vocab.eos_id
        self.pad_id = vocab.pad_id
        self.init_use = init_use

    @classmethod
    def build_module(cls, args, vocab, embed=None, **kwargs):
        return cls(
            vocab=vocab,
            max_len=kwargs.get('max_len', args.max_len),
            input_size=kwargs.get('input_size', args.d_model),
            hidden_size=kwargs.get('hidden_size', args.d_hidden),
            embed_droprate=kwargs.get('embed_droprate', args.drop_embed),
            rnn_droprate=kwargs.get('rnn_droprate', args.dropout),
            num_layers=kwargs.get('num_layers', args.dec_num_layers),
            rnn_cell=kwargs.get('rnn_cell', args.rnn_cell),
            embedding=embed,
            use_last_output=args.use_last_output
        )

    def forward_step(self, input_var, hidden, encoder_outputs, function):
        batch_size, output_size = input_var.size()
        embed = self.embedding(input_var)
        output, hidden = self.rnn(embed, hidden)

        if self.use_last_output:
            output = torch.cat([output, embed], dim=-1)

        score = self.projection(output.contiguous().view(batch_size * output_size, -1))
        prob = function(score, dim=-1).view(batch_size, output_size, -1)
        return prob, hidden

    def _decode(self, decoder_outputs, sequence_symbols, lengths, step, step_output, step_attn=None):
        decoder_outputs.append(step_output)
        symbols = decoder_outputs[-1].topk(1)[1]
        sequence_symbols.append(symbols)

        eos_batches = symbols.data.eq(self.eos_id)
        if eos_batches.dim() > 0:
            eos_batches = eos_batches.cpu().view(-1).numpy()
            update_idx = ((lengths > step) & eos_batches) != 0
            lengths[update_idx] = len(sequence_symbols)
        return decoder_outputs, sequence_symbols, lengths, symbols

    def scoring(self, scores, tgt_var, norm_by_word=False):
        batch_size = scores.size(1)
        if tgt_var.size(0) == batch_size:
            tgt_var = tgt_var.contiguous().transpose(1, 0)
        flatten_tgt_var = tgt_var[1:].contiguous().view(-1)

        log_scores = F.log_softmax(scores.view(-1, scores.size(2)), dim=-1)
        tgt_log_scores = torch.gather(log_scores, 1, flatten_tgt_var.unsqueeze(1)).squeeze(1)
        tgt_log_scores = tgt_log_scores * (1. - flatten_tgt_var.eq(self.pad_id).float())
        tgt_log_scores = tgt_log_scores.view(-1, batch_size).sum(dim=0)

        if norm_by_word:
            tgt_len = tgt_var.transpose(1, 0).ne(self.pad_id).sum(dim=-1).float()
            tgt_log_scores = tgt_log_scores / tgt_len

        return tgt_log_scores

    def forward(self, inputs, encoder_hidden=None, encoder_outputs=None, score_func=F.log_softmax, forcing_ratio=0.0):
        ret_dict = dict()
        decoder_hidden = self.init_decoder(encoder_hidden)
        inputs, batch_size, max_len = self.validation(inputs, decoder_hidden, encoder_outputs, forcing_ratio)
        use_teacher_forcing = True if random.random() < forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_len] * batch_size)

        if use_teacher_forcing:
            # forward with ground truth
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                               score_func)
            for step in range(decoder_output.size(1)):
                step_output = decoder_output[:, step, :]
                decoder_outputs, sequence_symbols, lengths, symbols = self._decode(
                    decoder_outputs, sequence_symbols, lengths, step, step_output
                )
        else:
            # forward with predicted token
            decoder_input = inputs[:, 0].unsqueeze(1)
            for step in range(max_len):
                decoder_output, decoder_hidden = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs,
                    score_func
                )
                step_output = decoder_output.squeeze(1)
                decoder_outputs, sequence_symbols, lengths, symbols = self._decode(
                    decoder_outputs, sequence_symbols, lengths, step, step_output
                )
                decoder_input = symbols

        ret_dict[RNNDecoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[RNNDecoder.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict, encoder_hidden

    def validation(self, inputs=None, encoder_hidden=None, encoder_outputs=None, forcing_ratio=0.0):
        batch_size = 1
        if inputs is not None or encoder_hidden is not None:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell == 'lstm':
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell == 'gru':
                    batch_size = encoder_hidden.size(1)

        if inputs is None:
            if forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.tensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_len
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length

    def init_decoder(self, encoder_hidden):
        if self.init_use == 'enc':
            return encoder_hidden
        else:
            return None


class BridgeRNNDecoder(RNNDecoder):
    def __init__(self, vocab: Dictionary, max_len, input_size, hidden_size, embed_droprate=0, rnn_droprate=0,
                 num_layers=1, rnn_cell='gru', embedding=None, update_embedding=True, init_use='enc',
                 use_last_output=True):
        super().__init__(vocab, max_len, input_size, hidden_size, embed_droprate, rnn_droprate, num_layers, rnn_cell,
                         embedding, update_embedding, init_use, use_last_output)
        self.bridge = None  # to be set in build module

    @classmethod
    def build_module(cls, args, vocab, embed=None, **kwargs):
        model = super().build_module(args, vocab, embed, **kwargs)
        enc_dim = kwargs.get('enc_dim', args.enc_hidden_dim * (2 if args.bidir else 1))
        dec_dim = kwargs.get('dec_dim', args.dec_hidden_dim)
        model.bridge = MLPBridger(
            rnn_type=args.rnn_cell,
            enc_dim=enc_dim,
            enc_layer=kwargs.get('enc_layer', args.enc_num_layers),
            dec_dim=dec_dim,
            dec_layer=kwargs.get('dec_layer', args.dec_num_layers),
            batch_first=kwargs.get('batch_first', args.batch_first)
        )
        return model

    def init_decoder(self, encoder_hidden):
        if self.init_use == "enc":
            return self.bridge(encoder_hidden)
        else:
            return None


class BridgeSoftmax(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, pad_id=1, droprate=0.1, embed=None):
        super(BridgeSoftmax, self).__init__()
        self.input_size = input_size
        self.bridge = nn.Sequential(
            nn.Dropout(droprate),
            nn.Linear(input_size, hidden_size, True),
            nn.ReLU(),
        )
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)
        if embed is not None and self.output.weight.size() == embed.weight.size():
            # tie the embedding weight and the projection layers
            self.output.weight = embed.weight
        self.pad_id = pad_id

    def forward(self, hidden, logp=True):
        hidden = hidden.contiguous().view(-1, self.input_size)
        score = self.output(self.bridge(hidden))
        if logp:
            return score.log_softmax(dim=-1)
        return score
