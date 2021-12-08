import numpy as np
import pandas as pd
#import torchtext # uncomment to run load_imdb_online
from tqdm import tqdm
from utils import Language, backtranslation, contextual_word_embeddings


def load_imdb_online():
    train_iter, test_iter = torchtext.datasets.IMDB(root='../data', split=('train', 'test'))
    train_df = pd.DataFrame(train_iter, columns=['label', 'text'])
    test_df = pd.DataFrame(test_iter, columns=['label', 'text'])
    return train_df, test_df


def save_imdb_original():
    train_df, test_df = load_imdb_online()
    train_df.to_pickle("./imdb_train_df_original.pkl", protocol=4)
    test_df.to_pickle("./imdb_test_df_original.pkl", protocol=4)


def save_imdb_backtrans(lang, begin_chunk_idx = 0, device='cuda'):
    for i, chunk in list(enumerate(np.array_split(pd.read_pickle("./imdb_train_df_original.pkl"), 250)))[begin_chunk_idx:]:
        chunk['text'] = chunk.progress_apply(lambda row: backtranslation(row['text'], lang, device=device), axis=1)
        chunk.to_pickle(f"./imdb_train_df_augd_backtrans_{lang}/imdb_train_df_augd_backtrans_{lang}_{i}.pkl")


# Just one chunk since this is relatively fast
def save_imdb_contextual_word_embeddings():
    df = pd.read_pickle("./imdb_train_df_original.pkl")
    df['text'] = df.progress_apply(lambda row: contextual_word_embeddings(row['text']), axis=1)
    df.to_pickle(f"./imdb_train_df_augd_contextual_word_embeddings.pkl")



if __name__ == '__main__':
    tqdm.pandas()
    # save_imdb_original()
    #save_imdb_backtrans(Language.German, begin_chunk_idx=99, device='cpu')
    save_imdb_contextual_word_embeddings()
