#%% imports
import pandas as pd

#%% read data
df1 = pd.read_csv('stock_sentiment.csv')
df2 = pd.read_csv('tweet_sentiment.csv')

print(df1.head())
print(df2.head())

print(df1.shape)
print(df2.shape)

print(df1.describe())
print(df2.describe())
