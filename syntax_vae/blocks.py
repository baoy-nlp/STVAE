import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    """
        Autoregressive Encoding
    """

    module_cell = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        'rnn': nn.RNN
    }

    def __init__(self, args, embedding: nn.Embedding):
        super().__init__()

        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        self.bidir = args.bidir
        self.num_layers = args.num_layers
        self.embedding = embedding

        rnn_cell = RNNEncoder.module_cell[args.rnn_type]
        self.rnn = rnn_cell(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=self.bidir,
            dropout=args.rnn_dropout,
        )

        self.dropout = nn.Dropout(args.dropout)
        self.output_dim = self.hidden_dim * 2 if self.bidir else self.hidden_dim

    def extract_hidden(self, hiddens):
        if isinstance(self.rnn, nn.LSTM):
            hidden0 = torch.cat((hiddens[0][-2, :, :], hiddens[0][-1, :, :]), dim=1)
            hidden1 = torch.cat((hiddens[1][-2, :, :], hiddens[1][-1, :, :]), dim=1)
            return hidden0 + hidden1
        else:
            return torch.cat((hiddens[-2, :, :], hiddens[-1, :, :]), dim=1)

    def forward(self, src, src_len):
        embed = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embed, src_len)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # [src_len, batch_size, enc hid dim * 2]

        # [batch_size, enc_hid dim*2 ]
        hidden = self.extract_hidden(hidden)
        return outputs, hidden


class RNNDecoder(nn.Module):
    """
        Autoregressive Sequence Generation without Enc-Dec Attention
    """

    module_cell = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        'rnn': nn.RNN
    }

    def __init__(self, args, embeddings: nn.Embedding, sos_id, eos_id):
        super().__init__()
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.num_tokens = embeddings.num_embeddings
        self.embed_dim = embeddings.embedding_dim
        self.hidden_dim = args.hidden_dim

        self.embedding = embeddings

        self.dropout = nn.Dropout(args.dropout)

        rnn_cell = RNNDecoder.module_cell[args.rnn_type]

        self.rnn = rnn_cell(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            bidirectional=False
        )

        self.output_without_embed = getattr(args, "output_without_embed", False)
        if self.output_without_embed:
            self.fc1 = nn.Linear(self.hidden_dim, self.embed_dim)
        else:
            self.fc1 = nn.Linear(self.hidden_dim + self.embed_dim, self.embed_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.embed_dim, self.num_tokens)

        if args.share_input_output_embed:
            self.fc2.weight = self.embedding.weight

    def init_hidden(self, hidden):
        if isinstance(self.rnn, nn.LSTM):
            # batch_size, 2*hidden_size
            hidden = hidden.view(hidden.size(0), 2, 1, -1).permute(1, 2, 0, 3)
            return hidden[0].contiguous(), hidden[1].contiguous()
        else:
            return hidden.unsqueeze(0)

    def forward(self, inputs, hidden, mask=None):
        # hidden = hidden.unsqueeze(0)
        hidden = self.init_hidden(hidden)
        # hidden: num_layers , batch_size, hidden_dim

        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        # inputs: 1 or tgt_len, batch_size

        # embedded: tgt_len, batch_size, embed_dim
        embedded = self.dropout(self.embedding(inputs))

        # output: tgt_len, batch_size, hidden_dim
        # hidden: 1,  batch_size, hidden_dim
        output, hidden = self.rnn(embedded, hidden)

        logits = self.output_layer(output, embedded)

        # logits: 1 or tgt_len, batch_size, num_tokens
        return logits

    def output_layer(self, output, embedded):
        if self.output_without_embed:
            fc1_out = self.fc1(output)
        else:
            fc1_out = self.fc1(torch.cat([output, embedded], dim=-1))
        logits = self.fc2(self.dropout(self.relu(fc1_out)))
        return logits

    def generate(self, hidden, max_sentence=100):
        """

        :param hidden:
        :return:
        """

        def _sample(dist, mode='greedy'):

            if mode == 'greedy':
                _, sample = dist.topk(1, dim=-1)
                sample = sample.reshape(-1)
                return sample
            else:
                raise NotImplementedError

        def _save_sample(save_to, sample, _index, step):
            running_latest = save_to[_index]
            # update token at position t
            running_latest[:, step] = sample.data
            # save back
            save_to[_index] = running_latest

            return save_to

        batch_size = hidden.size(1)

        # hidden = hidden.unsqueeze(0)
        hidden = self.init_hidden(hidden)

        sequence_idx = torch.arange(0, batch_size).long().to(hidden)  # all idx of batch
        sequence_running = torch.arange(0, batch_size).long().to(hidden)
        sequence_mask = torch.ones(batch_size).bool().to(hidden)
        running_seqs = torch.arange(0, batch_size).long().to(hidden)
        prediction = torch.Tensor(batch_size, max_sentence).fill_(self.pad_idx).long().to(hidden)
        t = 0

        inputs = torch.Tensor(batch_size).fill_(self.sos_id).long().to(hidden)
        # batch_size

        while t < self.max_sentence and len(running_seqs) > 0:
            inputs = inputs.unsqueeze(0)  # 1, batch_size

            embedded = self.embedding(inputs)

            output, hidden = self.rnn(embedded, hidden)

            logits = self.output_layer(output, embedded)  # 1, batch_size, vocab

            inputs = _sample(logits)  # batch_size

            prediction = _save_sample(prediction, inputs, sequence_running, t)

            sequence_mask[sequence_running] = (inputs != self.eos_id)
            sequence_running = sequence_idx.masked_select(sequence_mask)

            running_mask = (inputs != self.eos_id).data
            running_seqs = running_seqs.masked_select(running_mask)

            if len(running_seqs) > 0:
                inputs = inputs[running_seqs]
                hidden = hidden[running_seqs, :]
                running_seqs = torch.arange(0, len(running_seqs)).long().to(hidden)

            t += 1

        return prediction
