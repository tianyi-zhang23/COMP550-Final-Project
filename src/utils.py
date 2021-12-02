import random
import torch
from torch.utils.data import Dataset
import nlpaug.augmenter.word as naw
from enum import Enum


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


'''
class PandasDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return self.dataframe.iloc[index]

'''
