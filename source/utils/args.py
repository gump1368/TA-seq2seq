import argparse

SOURCE = '/home/gump/Software/pycharm-2018.1.6/projects/TA-Seq2Seq/data/'


def config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=SOURCE+'resource/')
    parser.add_argument("--data_prefix", type=str, default="demo")
    parser.add_argument("--save_dir", type=str, default=SOURCE+'models/')
    parser.add_argument("--embed_file", type=str, default=None)

    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=800)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--attn", type=str, default='mlp',
                        choices=['none', 'mlp', 'dot', 'general'])

    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr_decay", type=float, default=None)

    args = parser.parse_args()

    return args

