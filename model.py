import time
import math
import random

import torch
import torch.nn.functional as F

from torch import nn
from torch import optim
from torch.autograd import Variable

from lang import USE_CUDA
from lang import MAX_LENGTH
from lang import prepare_data
from lang import variable_from_pair
from lang import SOS_TOKEN


HIDDEN_SIZE = 500
N_LAYERS = 2
LEARNING_RATE = 0.0001
NUM_EPOCHS = 200000


class Encoder(nn.Module):

    def __init__(self, input_vocab_size, hidden_size, n_layers=1):
        super(Encoder, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)

        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA:
            return hidden.cuda()
        else:
            return hidden


class Attn(nn.Module):

    def __init__(self, hidden_size, max_length=MAX_LENGTH):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        attn_energies = Variable(torch.zeros(seq_len))
        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        for i in range(seq_len):
            attn_energies[i] = self._score(hidden, encoder_outputs[i])

        return F.softmax(attn_energies, dim=0).view(1, 1, -1)

    def _score(self, hidden, encoder_output):
        energy = self.attn(hidden)
        energy = hidden.dot(encoder_output)
        return energy


class AttnDecoder(nn.Module):

    def __init__(self, hidden_size, output_vocab_size, n_layers=1):
        super(AttnDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_vocab_size = output_vocab_size
        self.n_layers = n_layers

        self.attn = Attn(hidden_size)
        self.embedding = nn.Embedding(output_vocab_size, hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(2 * hidden_size, output_vocab_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):

        embedded = self.embedding(word_input).view(1, 1, -1)

        rnn_input = torch.cat([embedded, last_context.unsqueeze(0)], dim=2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        output = F.log_softmax(self.out(torch.cat([rnn_output, context], dim=1)), dim=1)

        return output, context, hidden, attn_weights


def train(
    input_variable,
    output_variable,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    max_length=MAX_LENGTH
):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    input_length = input_variable.size()[0]
    output_length = output_variable.size()[0]

    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[SOS_TOKEN]]))
    decoder_hidden = encoder_hidden
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))

    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    for di in range(output_length):
        decoder_output, decoder_context, decoder_hidden, _ = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_variable[di])
        decoder_input = target_variable[di]

    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / output_length


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm, %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


if __name__ == '__main__':
    clip = 25
    input_lang, output_lang, pairs = prepare_data('eng', 'fra')

    encoder = Encoder(input_lang.n_words, HIDDEN_SIZE, N_LAYERS)
    decoder = AttnDecoder(HIDDEN_SIZE, output_lang.n_words, N_LAYERS)

    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()

    plot_every = 200
    print_every = 1000

    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0


    for epoch in range(NUM_EPOCHS):

        training_pair = variable_from_pair(input_lang, output_lang, random.choice(pairs))
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss

        if epoch == 0: continue

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / NUM_EPOCHS), epoch, epoch / NUM_EPOCHS * 100, print_loss_avg)
            print(print_summary)

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
