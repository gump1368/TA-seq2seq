import torch
import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder


class TopicSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, max_length=30, embedding=None):
        super(TopicSeq2Seq, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.embedding = self._init_embedding(embedding)

        self.encoder = Encoder(vocab_size, embedding_size, hidden_size)
        self.decoder = Decoder(vocab_size=self.vocab_size,
                               embedding_size=self.embedding_size,
                               hidden_size=self.hidden_size,
                               )

    def _init_embedding(self, embedding):
        embedded = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        if embedding:
            embedded.weight = nn.Parameter(embedding)
        return embedded

    def forward(self, inputs):
        src, src_length = inputs['src']
        tgt, tgt_length = inputs['tgt']
        topic, topic_length = inputs['cue']

        encoder_outputs, hidden = self.encoder(src, src_length)
        decoder_state = self.decoder(tgt, encoder_outputs, hidden, topic)

        return decoder_state


