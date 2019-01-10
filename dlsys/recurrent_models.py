import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple


def init_hidden(batch_size, num_hidden_units):
    return torch.zeros(1, batch_size, num_hidden_units)


def assert_input_is_batched(input_tensor: torch.Tensor):
    if len(input_tensor.shape) != 2:
        _err_msg = "expected input shape: [batch, seq_length], but got {shape}"
        raise TypeError(_err_msg.format(shape=input_tensor.shape))


def reinit(tensor: torch.Tensor):
    return tensor.clone().detach().requires_grad_(False)


class HParams(namedtuple("HParams", ["vocab_size", "n_fac", "n_hidden"])):
    pass


class CharSeqStatefulModel(nn.Module):
    def __init__(self, hparams: HParams, batch_size):
        super().__init__()
        _ = hparams
        self.hparams = _
        self.embedding = nn.Embedding(_.vocab_size, _.n_fac)
        self.rnn = nn.RNN(_.n_fac, _.n_hidden)
        self.h = init_hidden(batch_size, _.n_hidden)

        self.l_in = nn.Linear(_.n_fac, _.n_hidden)
        self.l_hidden = nn.Linear(_.n_hidden, _.n_hidden)
        self.l_out = nn.Linear(_.n_hidden, _.vocab_size)

    def forward(self, batched_input_tensors):
        assert_input_is_batched(batched_input_tensors)
        batch_size = batched_input_tensors[0].shape[0]
        if self.h.shape[1] != batch_size:
            self.h = init_hidden(batch_size, self.hparams.n_hidden)
        output, h = self.rnn(self.embedding(batched_input_tensors), self.h)
        self.h = reinit(h)
        return torch.log_softmax(
            self.l_out(output), dim=-1).view(-1, self.hparams.vocab_size)

    def _forward_native(self, batched_input_tensors):
        input_tensors = [
            torch.relu(self.l_in(self.embedding(ch)))
            for ch in batched_input_tensors
        ]
        batch_size = input_tensors[0].shape[0]
        h = torch.zeros(input_tensors[0].size())
        for input_tensor in input_tensors:
            h = torch.tanh(self.l_hidden(h + input_tensor))

        return torch.log_softmax(self.l_out(h))


model = CharSeqStatefulModel(
    HParams(n_hidden=64, n_fac=32, vocab_size=62), batch_size=4)

char_seq = torch.tensor(
    np.random.randint(0, 9, size=(4, 1000)), dtype=torch.long)
model.forward(char_seq)
