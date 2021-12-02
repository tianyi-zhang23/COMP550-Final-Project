import random
from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import Dataset
import nlpaug.augmenter.word as naw
from enum import Enum


class Language(Enum):
    German = 0


backtranslation_model_names = {
    Language.German: ('facebook/wmt19-en-de', 'facebook/wmt19-de-en')
}

backtranslation_aug: dict[Language, Any] = {}


def backtranslation(text: str, lang: Language) -> str:
    if lang not in backtranslation_aug:
        backtranslation_aug[lang] = naw.BackTranslationAug(
            from_model_name=backtranslation_model_names[lang][0],
            to_model_name=backtranslation_model_names[lang][1]
        )
    return backtranslation_aug[lang].augment(text)


contextual_word_embs_aug = None


def contextual_word_embeddings(text: str) -> str:
    global contextual_word_embs_aug
    if contextual_word_embs_aug is None:
        contextual_word_embs_aug = naw.ContextualWordEmbsAug()
    return contextual_word_embs_aug.augment(text)


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
