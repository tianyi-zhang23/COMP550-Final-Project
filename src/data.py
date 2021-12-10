import pandas as pd
import random
import numpy as np


def split_data_set(size, train_iter, test_iter):
    train_size = int(0.8*size)
    test_size = int(0.2*size)
    train_list, test_list= list(train_iter), list(test_iter)
    num_class = len(set([label for (label, text) in train_iter]))
    train_idx = [i for i in range(len(train_list))]
    test_idx = [i for i in range(len(test_list))]

    random.shuffle(train_list)
    random.shuffle(test_list)

    sample_train_idx = np.random.choice(train_idx,train_size , replace=False)
    sample_test_idx = np.random.choice(test_idx, test_size, replace=False)

    sample_train = [train_list[i] for i in sample_train_idx ]
    sample_test  = [test_list[i] for i in sample_test_idx ]

    return pd.DataFrame(sample_train,columns=['label', 'text']), pd.DataFrame(sample_test,columns=['label', 'text'])

def get_dataset(dataset, size):

    if dataset == 'imdb':
        if size == 's':
            train_df = pd.read_pickle("../data/IMDB/imdb_small_train.pkl")
            test_df = pd.read_pickle('../data/IMDB/imdb_small_test.pkl')
        elif size == 'm':
            train_df = pd.read_pickle("../data/IMDB/imdb_med_train.pkl")
            test_df = pd.read_pickle('../data/IMDB/imdb_med_test.pkl')
        elif size == 'l':
            train_df = pd.read_pickle("../data/IMDB/imdb_large_train.pkl")
            test_df = pd.read_pickle('../data/IMDB/imdb_large_test.pkl')

    elif dataset =='agnews':
        if size == 's':
            train_df = pd.read_pickle("../data/AG_NEWS/agnews_small_train.pkl")
            test_df = pd.read_pickle('../data/AG_NEWS/agnews_small_test.pkl')
        elif size == 'm':
            train_df = pd.read_pickle("../data/AG_NEWS/agnews_med_train.pkl")
            test_df = pd.read_pickle('../data/AG_NEWS/agnews_med_test.pkl')
        elif size =='l':
            train_df = pd.read_pickle("../data/AG_NEWS/agnews_large_train.pkl")
            test_df = pd.read_pickle('../data/AG_NEWS/agnews_large_test.pkl')

    elif dataset == 'sogou':

        if size == 's':
            train_df = pd.read_pickle("../data/SogouNews/sogou_small_train.pkl")
            test_df = pd.read_pickle('../data/SogouNews/sogou_small_test.pkl')
        elif size == 'm':
            train_df = pd.read_pickle("../data/SogouNews/sogou_med_train.pkl")
            test_df = pd.read_pickle('../data/SogouNews/sogou_med_test.pkl')
        elif size == 'l':
            train_df = pd.read_pickle("../data/SogouNews/sogou_large_train.pkl")
            test_df = pd.read_pickle('../data/SogouNews/sogou_large_test.pkl')

    elif dataset == 'amz':

        if size == 's':
            train_df = pd.read_pickle("../data/AmazonReviewPolarity/amz_small_train.pkl")
            test_df = pd.read_pickle('../data/AmazonReviewPolarity/amz_small_test.pkl')
        elif size == 'm':
            train_df = pd.read_pickle("../data/AmazonReviewPolarity/amz_med_train.pkl")
            test_df = pd.read_pickle('../data/AmazonReviewPolarity/amz_med_test.pkl')
        elif size == 'l':
            train_df = pd.read_pickle("../data/AmazonReviewPolarity/amz_large_train.pkl")
            test_df = pd.read_pickle('../data/AmazonReviewPolarity/amz_large_test.pkl')
    elif dataset == 'yelp':
        if size == 's':
            train_df = pd.read_pickle("../data/YelpReviewPolarity/yelp_small_train.pkl")
            test_df = pd.read_pickle('../data/YelpReviewPolarity/yelp_small_test.pkl')
        elif size == 'm':
            train_df = pd.read_pickle("../data/YelpReviewPolarity/yelp_med_train.pkl")
            test_df = pd.read_pickle('../data/YelpReviewPolarity/yelp_med_test.pkl')
        elif size == 'l':
            train_df = pd.read_pickle("../data/YelpReviewPolarity/yelp_large_train.pkl")
            test_df = pd.read_pickle('../data/YelpReviewPolarity/yelp_large_test.pkl')

    elif dataset=='yahoo':
        if size == 's':
            train_df = pd.read_pickle("../data/YahooAnswers/yahoo_small_train.pkl")
            test_df = pd.read_pickle('../data/YahooAnswers/yahoo_small_test.pkl')
        elif size == 'm':
            train_df = pd.read_pickle("../data/YahooAnswers/yahoo_med_train.pkl")
            test_df = pd.read_pickle('../data/YahooAnswers/yahoo_med_test.pkl')
        elif size == 'l':
            train_df = pd.read_pickle("../data/YahooAnswers/yahoo_large_train.pkl")
            test_df = pd.read_pickle('../data/YahooAnswers/yahoo_large_test.pkl')

    return train_df, test_df