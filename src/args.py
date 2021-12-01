import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--data', type=str, default="imdb",
                        help="dataset to use ")
    parser.add_argument('--augment', type=str, default='none',
                        help="data augmentation technqiue: rs-RandomSwap")
    parser.add_argument('--swap_prob', type=float, default = 0.2,
                        help='random swap parameter where number of times = 0.2*len of sentence')