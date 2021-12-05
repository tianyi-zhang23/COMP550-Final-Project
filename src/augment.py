import numpy as np
import pandas as pd
import torchtext
from tqdm import tqdm
from utils import Language, backtranslation, contextual_word_embeddings


def load_imdb_online():
    train_iter, test_iter = torchtext.datasets.IMDB(root='../data', split=('train', 'test'))
    train_df = pd.DataFrame(train_iter, columns=['label', 'text'])
    test_df = pd.DataFrame(test_iter, columns=['label', 'text'])
    return train_df, test_df


def save_imdb_original():
    train_df, test_df = load_imdb_online()
    train_df.to_pickle("./imdb_train_df_original.pkl")
    test_df.to_pickle("./imdb_test_df_original.pkl")


def save_imdb_backtrans(lang):
    for i, chunk in enumerate(np.array_split(pd.read_pickle("./imdb_train_df_original.pkl"), 250)):
        chunk['text'] = chunk.progress_apply(lambda row: backtranslation(row['text'], lang), axis=1)
        chunk.to_pickle(f"./imdb_train_df_augd_backtrans_{lang}/imdb_train_df_augd_backtrans_{lang}_{i}.pkl")


def save_imdb_contextual_word_embeddings():
    for i, chunk in enumerate(np.array_split(pd.read_pickle("./imdb_train_df_original.pkl"), 250)):
        chunk['text'] = chunk.progress_apply(contextual_word_embeddings, axis=1)
        chunk.to_pickle(f"./imdb_train_df_augd_contextual_word_embeddings"
                           f"/imdb_train_df_augd_contextual_word_embeddings_{i}.pkl")


if __name__ == '__main__':
    tqdm.pandas()
    # save_imdb_original()
    save_imdb_backtrans(Language.German)
