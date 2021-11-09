"""
    A wrapper class for RNN.
"""
import torch.nn as nn

rnn_cls_dict = {
    "lstm": nn.LSTM,
    "gru": nn.GRU,
    'rnn': nn.RNN
}


class WrapperRNN(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 bidirectional=False,
                 rnn_cell='rnn',
                 dropout=0.1,
                 batch_first=True,
                 keep_shape=True,
                 **kwargs):
        super(WrapperRNN, self).__init__()
        self.rnn_cell = rnn_cell
        rnn_cls = rnn_cls_dict[rnn_cell.lower()]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=batch_first
        )
        self.batch_first = batch_first
        self.bidir = bidirectional
        if bidirectional and keep_shape:
            self.mapper = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, input_size, bias=True)
            )
        else:
            self.mapper = None

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, inputs, hidden=None, mask=None):
        self.flatten_parameters()
        if mask is not None:
            total_len = inputs.size(1)
            lens = mask.sum(dim=-1).long()
            pack_len, idx_sort = lens.sort(dim=-1, descending=True)
            idx_unsort = idx_sort.sort(dim=-1)[1]

            inputs = inputs[idx_sort, :, :]
            pack = nn.utils.rnn.pack_padded_sequence(
                inputs, pack_len.tolist(), batch_first=self.batch_first
            )
            out, hid = self.rnn(pack)
            unpack_out, unpacked_len = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=self.batch_first,
                total_length=total_len
            )
            output = unpack_out[idx_unsort, :, :]

            hid = hid[:, idx_unsort, :]

        else:
            output, hid = self.rnn(inputs)
        if self.bidir:
            output = self.mapper(output)

        if self.batch_first:
            hid = hid.permute(1, 0, 2)

        return output, hid


if __name__ == "__main__":
    test_rnn = WrapperRNN(input_size=3, hidden_size=5, num_layers=2, bidirectional=True)
    import torch

    test_inputs = torch.Tensor(3, 2, 3)
    test_mask = torch.Tensor(
        [
            [1, 1],
            [1, 1],
            [1, 0]
        ]
    )

    test_output = test_rnn(test_inputs, mask=test_mask)

    default_output = test_rnn.rnn(test_inputs)

    print("finish")
