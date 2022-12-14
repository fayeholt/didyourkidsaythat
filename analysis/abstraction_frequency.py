import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# using Python's wordfreq package - returns frequency from 0 to 1
from wordfreq import word_frequency
from preprocess import process


# load the DKST dataset
df = pd.read_csv(r'/Users/madelineholt/didyourkidsaythat/data/WokeKids_dataset.csv')
# preprocess text
df = process(df)
print(df.head())

# load the abstraction lexicon from University of Stuttgart NLP - returns an abstraction rating from 0 - 10
AC = pd.read_csv(r'/Users/madelineholt/didyourkidsaythat/data/AC_ratings_google3m_koeper_SiW.csv', on_bad_lines='skip', delimiter='\t')
x = AC['RATING']
AC['RATING'] = (x-min(x))/(max(x)-min(x))
AC_dict = dict(zip(AC.WORD, AC.RATING))
# print(AC_dict)

# test word frequency package
# print(word_frequency('everyone', 'en'))

def AC_encoding(data, AC_dictionary):
    abstracts = []
    for tweet in data['tweet']:
        temp = []
        for w in tweet:
            try:
                temp.append(AC_dictionary[w])
            except:
                temp.append(-1)
        abstracts.append(sum(temp)/len(temp))
        # abstracts.append(temp)
    data['AC'] = abstracts
    return data

def freq_encoding(data):
    freqs = []
    for tweet in data['tweet']:
        temp = []
        for w in tweet:
            try:
                # multiply by 10 to compare to abstraction encoding
                temp.append(word_frequency(w, 'en'))
            except:
                temp.append(-1)
        freqs.append(sum(temp)/len(temp))
        # freqs.append(temp)
    data['freq'] = freqs
    return data

df2 = AC_encoding(df, AC_dict)
df2 = freq_encoding(df2)

# df2['upvote_percent'] = df2['upvote_percent'].apply(1 if x > .92 else 0)

# df2.loc[df2["upvote_percent"] >= .92, "upvote_percent"] = 1
# df2.loc[df2["upvote_percent"] < .92, "upvote_percent"] = 0

df2.to_csv('/Users/madelineholt/didyourkidsaythat/data/encoded_dataset_3.csv', index=False)

