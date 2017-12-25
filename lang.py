import unicodedata
import string
import re
import random
import time
import math

import torch
from torch.autograd import Variable

SOS_TOKEN = 0
EOS_TOKEN = 1
USE_CUDA = False
MAX_LENGTH = 10
GOOD_PREFIXES = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re "
)


def variable_from_pair(input_lang, target_lang, pair): pass
def prepare_data(lang1_name, lang2_name): pass


class Lang:
    def __init__(self, name):
        self.name = name
        self.word_ix = {}
        self.word_count = {}
        self.ix_word = {SOS_TOKEN: 'SOS', EOS_TOKEN: 'EOS'}
        self.n_words = 2

    def add_word(self, word):
        if word not in self.word_ix:
            self.word_ix[word] = self.n_words
            self.word_count[word] = 0
            self.ix_word[self.n_words] = word
            self.n_words += 1
        else:
            self.word_count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)


def variable_from_pair(input_lang, target_lang, pair):
    input_variable = _variable_from_sentence(input_lang, pair[0])
    target_variable = _variable_from_sentence(target_lang, pair[1])
    return (input_variable, target_variable)


def _variable_from_sentence(lang, sentence):
    ixs = _ixs_from_sentence(lang, sentence)
    ixs.append(EOS_TOKEN)
    var = Variable(torch.LongTensor(ixs).view(-1, 1))

    if USE_CUDA:
        var = var.cuda()

    return var


def _ixs_from_sentence(lang, sentence):
    return [lang.word_ix[word] for word in sentence.split(' ')]


def prepare_data(lang1_name, lang2_name):
    input_lang, output_lang, pairs = _read_lang(lang1_name, lang2_name)
    print("Read {} sentence pairs".format(len(pairs)))

    pairs = _filter_pairs(pairs)
    print('Filtered pairs down to {}'.format(len(pairs)))

    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    return input_lang, output_lang, pairs


def _read_lang(lang1, lang2):
    print("Reading Lines")

    pairs = []

    with open('./data/{}-{}.txt'.format(lang1, lang2)) as f:

        for line in f.readlines():
            pair = line.split('\t')
            pair = [_normalize_string(s) for s in pair]
            pairs.append(pair)

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def _normalize_string(s):
    s = _unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def _unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def _filter_pairs(pairs):
    return [pair for pair in pairs if _filter_pair(pair)]


def _filter_pair(pair):
    p1_length = len(pair[0].split(' ')) < MAX_LENGTH
    p2_length = len(pair[1].split(' ')) < MAX_LENGTH
    prefix_filter = pair[1].startswith(GOOD_PREFIXES)
    return p1_length and p2_length


if __name__ == '__main__':
    input_lang, output_lang, pairs = prepare_data('eng', 'fra')
