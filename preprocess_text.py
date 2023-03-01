import numpy as np
import pandas as pd
import string
import re
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split

def custom_standardization(input_data):
        import preprocessor as p
        processed_data = p.clean(input_data)
        lowercase_value = processed_data.lower()
        return lowercase_value

def suicidal_intent_data_load():
    train = pd.read_csv("Dataset/Twitter/train.csv", encoding = "ISO-8859-1",usecols=[0,5],header=None)
    test_df = pd.read_csv("Dataset/Twitter/test.csv", encoding = "ISO-8859-1",usecols=["Sentiment","SentimentText"])

    train.columns = ["target","text"]
    test_df.columns = ["target","text"]

    train['target'] = np.where(train['target']==4, 0, 1)
    test_df['target'] = np.where(test_df['target']==1, 0, 1)

    pandarallel.initialize()
    train["text"] = train["text"].parallel_apply(custom_standardization)
    test_df["text"] = test_df["text"].parallel_apply(custom_standardization)

    train_df, val_df = train_test_split(train, test_size=0.2)
    del train
    return train_df,val_df,test_df
