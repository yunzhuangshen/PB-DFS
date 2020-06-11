import  argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    'problem',
    choices=['mis', 'ca', 'ds', 'vc'],
)

parser.add_argument('--learning_rate', default=0.01)
parser.add_argument('--epochs', default=200)
parser.add_argument('--hidden1', default=64)
parser.add_argument('--dropout', default=0)
parser.add_argument('--weight_decay', default=5e-4)
parser.add_argument('--early_stopping', default=10)
parser.add_argument('--max_degree', default=1)
parser.add_argument('--num_layers', default=20)
args = parser.parse_args()
