import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPAttention(nn.Module):
    """
    apply concat attention mechanism in decoder.

    """
    def __init__(self, hidden_size=None):
        super(MLPAttention, self).__init__()
        self.hidden_size = hidden_size

        self.attention = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.v = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, output, encoder_output):
        """

        :param output: last step output: batch_size*1*hidden_size
        :param encoder_output:
        :return:
        """
        # TODO: mask
        batch_size, input_size, hidden_size = encoder_output.size()
        energy = self.attention(torch.cat((output.expand(-1, input_size, -1), encoder_output), dim=2)).tanh()

        attention_weight = F.softmax(torch.sum(self.v*energy, dim=2), dim=1)

        return attention_weight


class TopicAttention(nn.Module):
    """
    To get the weight of linear combination of topics
    """
    def __init__(self, hidden_size, embedding_size):
        super(TopicAttention, self).__init__()
        self.hidden_size = hidden_size
        self.topic_attention = nn.Linear(self.hidden_size*2+embedding_size, self.hidden_size)
        self.v = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, output, encoder_output, topics):
        """

        :param output:
        :param encoder_output:
        :param topics:
        :return:
        """
        topic_size = topics.size(1)
        topic_embedded = self.topic_embedding(topics)
        last_encoder_hidden = encoder_output[:-1]

        combined = torch.cat((output, last_encoder_hidden), dim=2)
        energy = self.topic_attention(torch.cat((combined.expand(-1, topic_size, -1), topic_embedded), dim=-1))
        topic_attention_weight = F.softmax(torch.sum(self.v*energy, dim=2), dim=1)

        return topic_attention_weight
