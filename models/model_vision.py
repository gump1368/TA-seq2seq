import torch
from tensorboardX import SummaryWriter

from models.encoder import Encoder

encoder = Encoder(vocab_size=20000, hidden_size=512, max_length=30)
input_sequence = torch.LongTensor([
    [1, 3, 5, 4, 6],
    [2, 5, 9, 29, 30],
    [4, 23, 34, 0, 0]
])

input_length = torch.LongTensor([5, 5, 3])


with SummaryWriter(comment='encoder') as w:
    w.add_graph(encoder, (input_sequence, input_length), verbose=True)
