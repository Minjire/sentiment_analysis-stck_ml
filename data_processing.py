# %% imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import stopwords
import gensim
from tabulate import tabulate
from wordcloud import WordCloud
import plotly.express as px

# pdtabulate = lambda df: tabulate(df, headers='keys')

pdtabulate = lambda df: tabulate(df, headers='keys', tablefmt='psql')

# %% read data
df1 = pd.read_csv('data/raw/stock_sentiment.csv')
df2 = pd.read_csv('data/raw/tweet_sentiment.csv')

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
plt.title('Sentiment Frequencies')
plt.savefig("figures/sentiment frequencies.png", bbox_inches='tight')
plt.show()

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

# %% Data Cleaning--Remove Punctuations
# download stopwords
nltk.download('stopwords')

stop_words = stopwords.words('english')
# add stop words depending on the dataset
stop_words.extend(
    ['from', 'subject', 're', 'edu', 'use', 'will', 'aap', 'co', 'day', 'user', 'stock', 'today', 'week', 'year',
     'https'])
print(stop_words)


# remove stopwords and short words (< 2 characters)
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in stop_words and len(token) >= 3:
            result.append(token)

    return result


# apply pre-processing to the text column
stock_df['Text Without Punc & Stopwords'] = stock_df['Text Without Punctuation'].apply(preprocess)
# join the words into a string
stock_df['Text Without Punc & Stopwords Joined'] = stock_df['Text Without Punc & Stopwords'].apply(
    lambda x: " ".join(x))
print(stock_df.head(10))

# %% Visualize in a WordCloud
# plot the word cloud for text with positive sentiment
plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=1000, width=1600, height=800).generate(
    " ".join(stock_df[stock_df['Sentiment'] == 1]['Text Without Punc & Stopwords Joined']))
plt.imshow(wc, interpolation='bilinear')
plt.xticks([])
plt.yticks([])
plt.savefig("figures/positive sentiment word cloud.png", bbox_inches='tight')
plt.show()

# word cloud for negative sentiment
plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=1000, width=1600, height=800).generate(
    " ".join(stock_df[stock_df['Sentiment'] == -1]['Text Without Punc & Stopwords Joined']))
plt.imshow(wc, interpolation='bilinear')
plt.xticks([])
plt.yticks([])
plt.savefig("figures/negative sentiment word cloud.png", bbox_inches='tight')
plt.show()

# %% investigate on maximum length of data in the document
# This will be later used when word embeddings are generated
nltk.download('punkt')
# drop row with excessively long text compared to others
# store indices of excessively long text compared to others
long_texts = []

maxlen = -1
for doc in stock_df['Text Without Punc & Stopwords Joined']:
    tokens = nltk.word_tokenize(doc)
    if (maxlen < len(tokens) and len(tokens) < 50):
        maxlen = len(tokens)
    if len(tokens) > 50:
        long_texts.append(doc)
print("The maximum number of words in any document is:", maxlen)

stock_df = stock_df[~stock_df['Text Without Punc & Stopwords Joined'].isin(long_texts)]
tweets_length = [len(nltk.word_tokenize(x)) for x in stock_df['Text Without Punc & Stopwords Joined']]
# Plot the distribution for the number of words in a text
fig = px.histogram(x=tweets_length, nbins=50, labels={"x": "Text Length"}, title="Histogram of Length of Texts")
fig.write_html("figures/texts lengths distribution.html")
fig.show()

# %% save dataframe for model training
stock_df.to_csv('data/processed/cleaned_text.csv', index=False)
