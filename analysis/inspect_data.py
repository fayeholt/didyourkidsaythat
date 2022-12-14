import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r'/Users/madelineholt/didyourkidsaythat/data/WokeKids_dataset.csv')

# inspect dataset further here
print(df.describe())
print(df['age'].describe())

# plot histograms for percent count and age

fig = plt.figure(figsize=(10, 7))
plt.boxplot(df['upvote_percent'])
plt.show()

df_copy = df.dropna(how='any')
fig = plt.figure(figsize=(10, 7))
plt.boxplot(df_copy['age'])
plt.show()