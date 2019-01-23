import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np
from tensorboardX import SummaryWriter

from hypothesis import given, example
from hypothesis.strategies import text
import pytest

device = torch.device("cpu")

MAX_LENGTH = 10  # Maximum sentence length

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Voc(object):
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self._reinit_dict()

    def _reinit_dict(self):
        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS",
        }
        self.num_words = len(self.index2word)

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v > min_count:
                keep_words.append(k)

        print("keep_words {} / {} = {:.4f}".format(
            len(keep_words), len(self.word2index),
            len(keep_words) / len(self.word2index)))

        self._reinit_dict()
        for word in keep_words:
            self.add_word(word)


def normalize_string(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]", r" ", s)
    return s


def indexes_from_sentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


@pytest.mark.parametrize("s_in, s_expected", [
    ("abc", "abc"),
    ("abc%", "abc "),
    ("abc!!", "abc ! !"),
    ("ABC", "abc"),
])
def test_normalize_string(s_in, s_expected):
    s_out = normalize_string(s_in)
    assert s_out == s_expected


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=(0 if n_layers == 1 else dropout),
            bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Looking up word embeddings from indexes
        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.
                                                             hidden_size:]
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method,
                             "is not a supported attention method.")
        self.hidden_size = hidden_size
        if self.method == "general":
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(
            torch.cat((hidden.expand(encoder_output.size(0), -1, -1),
                       encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        if self.method == "general":
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == "dot":
            attn_energies = self.dot_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t()

        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self,
                 attn_model,
                 embedding,
                 hidden_size,
                 output_size,
                 n_layers=1,
                 dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexes_from_sentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


raw_sentences = [
    "hello", "what's up?", "who are you?", "where am I?", "where are you from?"
]
for raw_sent in raw_sentences:
    input_sent = normalize_string(raw_sent)

# These information are taken elsewhere
# corpus_name = "cornell movie-dialogs corpus"
# voc = Voc(corpus_name)

raw_sentences = [
    "able was I ere I saw",
    "nothing but blood, tear and sweat",
]
sentences = [normalize_string(raw_sent) for raw_sent in raw_sentences]
indexes_batch = sorted(
    [indexes_from_sentence(voc, sent) for sent in sentences],
    key=len,
    reverse=True)
lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
# Transpose dimensions of batch to match models' expectations
input_batch = [torch.LongTensor(indexes) for indexes in indexes_batch]
input_batch = nn.utils.rnn.pad_sequence(
    input_batch, batch_first=True, padding_value=PAD_token).transpose(0, 1)
# Use appropriate device
input_batch = input_batch.to(device)
lengths = lengths.to(device)

embedded = encoder.embedding(input_batch)
packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths)
print('packed size', packed.data.shape)


def test_pad_sequence():
    a = torch.ones(3, 100)
    b = torch.ones(2, 100)
    c = torch.ones(7, 100)
    seq = nn.utils.rnn.pad_sequence([a, b, c], padding_value=0)
    assert (7, 3, 100) == seq.shape


def test_pad_sequence_padding():
    vec_dim = 32
    a = 3 * torch.ones(3, vec_dim)
    b = 2 * torch.ones(2, vec_dim)
    c = 7 * torch.ones(7, vec_dim)
    raw_seq = sorted([a, b, c], key=len, reverse=True)
    lengths = torch.tensor([len(s) for s in raw_seq])
    padded_seq = nn.utils.rnn.pad_sequence(raw_seq, padding_value=0)
    packed_seq = nn.utils.rnn.pack_padded_sequence(padded_seq, lengths)

    assert sum(s.shape[0] for s in raw_seq) == packed_seq.data.shape[0], \
        "packed sequence must have leading size equal to sum of leading sizes"


def test_sequence_packing_unpacking():
    vec_dim = 32
    a = 3 * torch.ones(3, vec_dim)
    b = 2 * torch.ones(2, vec_dim)
    c = 7 * torch.ones(7, vec_dim)
    raw_seq = sorted([a, b, c], key=len, reverse=True)
    lengths = torch.tensor([len(s) for s in raw_seq])
    total_length = sum(map(len, raw_seq))

    # Build a simple forward seq model
    gru_layer = nn.GRU(
        input_size=32, hidden_size=48, num_layers=9, batch_first=False)

    padded_seq = nn.utils.rnn.pad_sequence(
        raw_seq, batch_first=False, padding_value=0)

    padded_seq_trans = nn.utils.rnn.pad_sequence(
        raw_seq, batch_first=True, padding_value=0)

    assert torch.equal(padded_seq.transpose(0, 1), padded_seq_trans)

    packed_seq = nn.utils.rnn.pack_padded_sequence(padded_seq, lengths)
    # Running inference with some model
    packed_output, packed_hidden = gru_layer(packed_seq)
    unpacked_output, _ = nn.utils.rnn.pad_packed_sequence(
        packed_output, batch_first=False, total_length=total_length)
