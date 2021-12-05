import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--data', type=str, default="imdb",
                        help="dataset to use: imdb, agnews, sogou, amz, yelp, yahoo")
    parser.add_argument('--augment', type=str, default='none',
                        help="data augmentation technqiue: rs-RandomSwap")
    parser.add_argument('--p', type=float, default = 0.2,
                        help='hyper parameter for eda')

    parser.add_argument('--size', type=str, default='s',
                        help="s=500, m=2000, l=5000")

