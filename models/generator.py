import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Generate outputs
    """
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Generator, self).__init__()

        self.vocab_size = vocab_size
        self.fc_V = nn.Linear(hidden_size+embedding_size, vocab_size)
        self.fc_K = nn.Linear(hidden_size*2+embedding_size, vocab_size)

        self.one_hot_embedding = self.init_weight()
        self.init_weight()

    def init_weight(self):
        embedding = nn.Embedding(self.vocab_size, self.vocab_size)
        weight = torch.eye(self.vocab_size)
        weight[0] = torch.zeros(self.vocab_size)
        embedding.weight = nn.Parameter(weight)
        return embedding

    def forward(self, output, input_step, context, topic):
        topic_indexs, topic_length = topic
        topic_one_hot = self.one_hot_embedding(topic_indexs)  # batch*length*vocab_size
        k = torch.sum(topic_one_hot, dim=1)  # batch*1*vocab_size
        v = k.eq(0)

        energy = F.tanh(v*self.fc_V(torch.cat((output, input_step), dim=2)) +
                        k*self.fc_K(torch.cat((output, input_step, context), dim=2)))  # batch*1*vocab_size

        prob = F.softmax(energy, dim=-1).squeeze()
        return prob



