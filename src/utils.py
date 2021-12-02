import random
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torchtext.datasets
def random_swap(words, p):
    n = int(len(words)*p)
    if n<1:
        return words
    else:
        words = words.split()
        for i in range(n):
            words = swap(words)

        return " ".join(words)

def random_deletion(words, p):
    words = words.split()
    if len(words) == 1:
        return words[0]

    new = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new.append(word)

    # if you end up deleting all words, just return a random word
    if len(new) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    sentence = ' '.join(new)

    return sentence

def swap(words):
    idx = random.sample(range(len(words)), 2)
    print(words, idx)
    words[idx[0]], words[idx[1]] = words[idx[1]], words[idx[0]]

    return words


def split_data_set(size, train_iter, test_iter):
    ratio = len(train_iter)/(len(train_iter)+ len(test_iter))
    test_size = int((1-ratio) * size/ratio)
    train_list, test_list= list(train_iter), list(test_iter)
    num_class = len(set([label for (label, text) in train_iter]))
    train_idx = [i for i in range(len(train_list))]
    test_idx = [i for i in range(len(test_list))]

    random.shuffle(train_list)
    random.shuffle(test_list)

    sample_train_idx = np.random.choice(train_idx,size , replace=False)
    sample_test_idx = np.random.choice(test_idx, test_size, replace=False)

    sample_train = [train_list[i] for i in sample_train_idx ]
    sample_test  = [test_list[i] for i in sample_test_idx ]

    return pd.DataFrame(sample_train,columns=['label', 'text']), pd.DataFrame(sample_test,columns=['label', 'text'])


def get_dataset(dataset, size):
    sizes = {'s':500, 'm':2000, 'l':5000}
    size = sizes[size]

    if dataset == 'imdb':
        train_iter, test_iter = torchtext.datasets.IMDB(root='../data', split=('train', 'test'))
    elif dataset =='agnews':
        train_iter, test_iter = torchtext.datasets.AG_NEWS(root='../data', split=('train', 'test'))
    elif dataset =='sogou':
        train_iter, test_iter = torchtext.datasets.SogouNews(root='../data', split=('train', 'test'))
    elif dataset=='amz':
        train_iter, test_iter = torchtext.datasets.AmazonReviewPolarity(root='../data', split=('train', 'test'))
    elif dataset == 'yelp':
        train_iter, test_iter = torchtext.datasets.YelpReviewPolarity(root='../data', split=('train', 'test'))
    elif dataset=='yahoo':
        train_iter, test_iter = torchtext.datasets.YahooAnswers(root='../data', split=('train', 'test'))

    return split_data_set(size, train_iter, test_iter)
