import torch
import torch.nn as nn

from models.base_rnn import BaseRNN


class Encoder(BaseRNN):
    def __init__(self, vocab_size, embedding_size, hidden_size, max_length=30, embedding=None):
        super(Encoder, self).__init__(vocab_size, embedding_size, hidden_size, max_length)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        if embedding:
            self.embedding.weight = nn.Parameter(embedding)

    def forward(self, input_sequence, input_length, hidden=None):
        embedded = self.embedding(input_sequence)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_length, batch_first=True)

        output, hidden = self.gru(packed, hidden)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output, hidden


if __name__ == '__main__':
    encoder = Encoder(vocab_size=1000, embedding_size=200, hidden_size=512)
    inputs = torch.LongTensor([
        [2, 4, 5, 12, 23],
        [3, 2, 34, 12, 23],
        [34, 12, 23, 33, 56]
    ])
    length = torch.LongTensor([5, 4, 3])
    output, hidden = encoder(inputs, length)
