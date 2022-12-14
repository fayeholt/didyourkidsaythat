import pandas as pd
import re
import string
# source: https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/

# removes punctuation, converts to lowercase, and tokenizes
def process(df):
    # remove punctuation
    punct = []
    for text in df['tweet']:
        punct.append("".join([i for i in text if i not in string.punctuation]))
    df['tweet'] = punct
    # convert to lowercase
    df['tweet'] = df['tweet'].apply(lambda x: x.lower())
    # tokenize
    tokens = []
    for text in df['tweet']:
        tokens.append(text.split())
    df['tweet'] = tokens

    return df



