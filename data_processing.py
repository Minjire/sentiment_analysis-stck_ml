# %% imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import nltk

# %% read data
df1 = pd.read_csv('stock_sentiment.csv')
df2 = pd.read_csv('tweet_sentiment.csv')

print(df1.head())
# print horizontal asterisks
print(''.ljust(70, '*'))
print(df2.head())
print(''.rjust(70, '*'))

print("\nDataframes Shapes:")
print(df1.shape)
print(df2.shape)

print("\nDataframes information:")
print(df1.describe())
print(''.ljust(70, '*'))
print(df2.describe())

# %% combine dataframes
# assign same column names
df2.columns = df1.columns

print(df1.Sentiment.unique())
print(df2.Sentiment.unique())

# check data types
print('\n', df1.dtypes)
print(''.ljust(25, '*'))
print(df2.dtypes)
# change 0(negative) to -1 in df1 to align with df2
df1.Sentiment.replace(0, -1, inplace=True)
print(df1.Sentiment.unique())

# append dataframes
stock_df = df1.append(df2, ignore_index=True)

print(f"Dataframe Shape: {stock_df.shape}")
print(f"Sentiment unique values: {stock_df.Sentiment.unique()}")
print(stock_df.tail(10))

# %%
# check for null values
print(stock_df.isnull().sum())
# display null rows
print(stock_df[stock_df.isna().any(axis=1)])
# drop null rows and reset index
stock_df.dropna(inplace=True)
stock_df.reset_index(drop=True, inplace=True)
print(stock_df.tail(10))

# %% check number of unique values in sentiment column
print(stock_df.Sentiment.value_counts())
sns.countplot(stock_df['Sentiment'])

# %% Data Cleaning--Remove Punctuations
print(string.punctuation)


# function to remove punctuations
def remove_punc(text):
    text_punc_removed = [char for char in text if char not in string.punctuation]
    text_punc_removed_join = ''.join(text_punc_removed)

    return text_punc_removed_join


# remove punctuations from our dataset
stock_df['Text Without Punctuation'] = stock_df['Text'].apply(remove_punc)
print(stock_df.head(10))
