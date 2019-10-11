import torch
import torch.nn as nn

from models.base_rnn import BaseRNN
from models.generator import Generator
from models.attention import MLPAttention, TopicAttention


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_size, embedding=None, use_attention=True):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if embedding:
            self.embedding.weight = embedding

        self.gru = nn.GRU(self.embedding_size*2+self.hidden_size, self.hidden_size)
        if use_attention:
            self.attention = MLPAttention(self.hidden_size)
            self.topic_attention = TopicAttention(
                                                  embedding_size=self.embedding_size,
                                                  hidden_size=self.hidden_size,
                                                  )

        self.generator = Generator(vocab_size=self.vocab_size,
                                   embedding_size=self.embedding_size,
                                   hidden_size=self.hidden_size)

    def forward_step(self, input_step, output, last_hidden, encoder_outputs, topics):

        embedded = self.embedding(input_step)  # batch*1*300

        context = self.attention(output, encoder_outputs)  # batch*1*hidden
        topic_vectors = self.topic_attention(output, last_hidden, topics)  # batch*1*300

        input_rnn = (embedded, context, topic_vectors)

        output, hidden = self.gru(torch.cat(input_rnn), last_hidden)

        prob_output = self.generator(output, input_step, context, topics)

        return prob_output, output, hidden

    def forward(self, tgt, encoder_output, hidden, topics, output=None):
        tgt_answer, tgt_length = tgt
        batch_size, max_len = tgt_answer.size()

        decoder_states = []
        for i in range(max_len):
            input_step = tgt_answer[:, i]
            prob_output, output, hidden = self.forward_step(input_step, output, hidden, encoder_output, topics)

            decoder_states.append(prob_output)
        return decoder_states





