import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np

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
