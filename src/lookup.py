import pandas as pd
from pandas import read_pickle

# Script for creating a dataframe for the backtranslated data of any subset of the IMDB dataset, given that
# we have already done backtranslation on the entire dataset.
if __name__ == '__main__':
    df = read_pickle("./imdb_train_df_augd_backtrans_Language.German/imdb_train_df_augd_backtrans_Language.German_0.pkl")
    for i in range(1,250):
        df = df.append(read_pickle(f"./imdb_train_df_augd_backtrans_Language.German/imdb_train_df_augd_backtrans_Language.German_{i}.pkl"))
    original = read_pickle("./imdb_train_df_original.pkl")
    original["backtranslated"] = df["text"]
    original = original.set_index("text")

    imdb_large = read_pickle("../data/IMDB/imdb_large_train.pkl")
    imdb_large['text'] = imdb_large.apply(lambda row: original.loc[row['text']]["backtranslated"], axis=1)
    imdb_large.to_pickle("../data/IMDB/imdb_large_train_backtrans_german.pkl")