import random
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torchtext.datasets
from typing import Any

import torch
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from torch.utils.data import Dataset
#import nlpaug.augmenter.word as naw
from enum import Enum

# Preprocessing Functions

def do_tokenize(sentence):
  # tokenize words, skip ALL punctuations
  return [word for word in word_tokenize(sentence) if word.isalpha()]

def get_tokenized_data(corpus):
  x = []
  for sentence in corpus:
    x.append(do_tokenize(sentence))
  return x

# Augmentation Functions

stop_words = stopwords.words('english')


class Language(Enum):
    German = 0
    French = 1


backtranslation_model_names = {
    Language.German: ('facebook/wmt19-en-de', 'facebook/wmt19-de-en'),
    Language.French: ('Helsinki-NLP/opus-mt-en-fr', 'Helsinki-NLP/opus-mt-fr-en')
}

_backtranslation_aug: 'dict[Language, Any]' = {}


def backtranslation(text: str, lang: Language) -> str:
    if lang not in _backtranslation_aug:
        _backtranslation_aug[lang] = naw.BackTranslationAug(
            from_model_name=backtranslation_model_names[lang][0],
            to_model_name=backtranslation_model_names[lang][1]
        )
    return _backtranslation_aug[lang].augment(text)


_contextual_word_embs_aug = None


def contextual_word_embeddings(text: str) -> str:
    global _contextual_word_embs_aug
    if _contextual_word_embs_aug is None:
        _contextual_word_embs_aug = naw.ContextualWordEmbsAug()
    return _contextual_word_embs_aug.augment(text)


def random_swap(words, p):
    words = words.split()
    n = int(len(words) * p)
    if n < 1:
        return words
    else:
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

def get_synonyms(word):

  synonyms = set()

  for syn in wordnet.synsets(word):
    for l in syn.lemmas():
      synonym = l.name().replace("_", " ").replace("-", " ").lower()
      synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
      synonyms.add(synonym)
  
  if word in synonyms:
    synonyms.remove(word)
  
  return list(synonyms)


def synonym_replacement(sentence, p):
  words = sentence.split()
  # n = # words to replace
  n = int(len(words)*p)

  new_words = words.copy()
  random_word_list = list(set([word for word in words if word not in stop_words]))
  random.shuffle(random_word_list)
  num_replaced = 0

  for random_word in random_word_list:
    synonyms = get_synonyms(random_word)

    if len(synonyms) >= 1:
      synonym = random.choice(list(synonyms))
      new_words = [synonym if word == random_word else word for word in new_words]
      num_replaced += 1
    
    if num_replaced >= n:
      break

  return ' '.join(new_words)


def random_insertion(words, p):
    words = words.split()
    n = int(len(words)*p)
    new_words = words.copy()

    for _ in range(n):
        add_word(new_words)

    sentence = ' '.join(new_words)
    return sentence


def add_word(new_words):
    synonyms = []
    counter = 0

    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return

    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


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
        train_iter, test_iter = torchtext.datasets.IMDB(root='../data', split=('train', 'test'))

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
