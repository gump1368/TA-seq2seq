import torch.nn as nn


class BaseRNN(nn.Module):
    """
    A base class for RNN.
    Here we use bidirectional gru with layers 1 and batch first.
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, max_length, dropout=0.2, n_layers=1, rnn_cell='gru',
                 bidirectional=True, batch_first=True):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        if rnn_cell.lower() in ['lstm', 'gru']:
            self.gru = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                              bidirectional=self.bidirectional, batch_first=self.batch_first)
        else:
            raise ValueError('Unsupported rnn_cell: {}'.format(rnn_cell))

    def forward(self, *args, **kwargs):
        raise NotImplementedError
