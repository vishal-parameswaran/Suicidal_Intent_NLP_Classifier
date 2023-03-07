import numpy as np
import pandas as pd
import string
import re
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
import preprocessor as p
import nltk

stemmer = nltk.SnowballStemmer("english")
stop_words = nltk.corpus.stopwords.words('english')

def custom_standardization(input_data,remove_stopwords=True):
        processed_data = p.clean(input_data)
        lowercase_value = processed_data.lower()
        punctuation_re = re.compile("[^\w\s]")
        modified = re.sub(punctuation_re,"",lowercase_value)
        modified = modified.split(' ')
        if remove_stopwords:
                modified = [word for word in modified if word not in stop_words]
        stemmed = ' '.join(stemmer.stem(word) for word in modified)
        return stemmed

def suicidal_intent_data_load(test_dataset = True,remove_stopwords = True,standardization=True):
    train = pd.read_csv("Dataset/Twitter/train.csv", encoding = "ISO-8859-1",usecols=[0,5],header=None)
    test_df = pd.read_csv("Dataset/Twitter/test.csv", encoding = "ISO-8859-1",usecols=["Sentiment","SentimentText"])

    train.columns = ["target","text"]
    test_df.columns = ["target","text"]

    train['target'] = np.where(train['target']==4, 0, 1)
    test_df['target'] = np.where(test_df['target']==1, 0, 1)

    pandarallel.initialize()
    if standardization:
        print("Hi")
        train["text"] = train["text"].parallel_apply(custom_standardization,remove_stopwords=remove_stopwords)
        if  test_dataset:
            test_df["text"] = test_df["text"].parallel_apply(custom_standardization)
        
    train_df, val_df = train_test_split(train, test_size=0.2)
    del train
    if test_dataset:
        return train_df,val_df,test_df
    else:
        return train_df,val_df
