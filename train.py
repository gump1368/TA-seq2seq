import torch
from torchtext.data import Iterator
from torch.utils.data import DataLoader

from source.utils.args import config
from source.utils.prepare_data import create_features

args = config()

data = torch.load(args.data_dir+'/demo_30000.data.pt')

train_data = create_features(data['train'])
valid_data = create_features(data['valid'])

train_iter = DataLoader(train_data, args.batch_size)
valid_data = DataLoader(valid_data, args.batch_size)


