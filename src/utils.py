import random
import torch
from torch.utils.data import Dataset
import nlpaug.augmenter.word as naw
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

def backtranslation(text: str, lang: Language) -> str:
    models = {
        Language.German: ('facebook/wmt19-en-de', 'facebook/wmt19-de-en')
    }
    aug = naw.BackTranslationAug(
        from_model_name=models[lang][0],
        to_model_name=models[lang][1])
    return aug.augment(text)


def random_swap(words, p):
    n = int(len(words) * p)
    if n < 1:
        return words
    else:
        words = words.split()
        for i in range(n):
            words = swap(words)

        return " ".join(words)


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

def synonym_replacement(sentence, n):
  # n = # words to replace

  words = sentence.split()
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
'''
class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return self.dataframe.iloc[index]

'''
