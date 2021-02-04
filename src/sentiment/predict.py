# %% imports
import pandas as pd
# from src.sentiment.data_processing import preprocess, remove_punc
from src.sentiment.remove_punctuation import remove_punc
from src.sentiment.preprocess import preprocess

# %% read data for specific day(today)
safaricom_df = pd.read_csv('data/raw/scraper/safaricom.csv')
pd.options.display.max_columns = None

print(safaricom_df.head())
# %% process data (remove punctuations and stop words)
# remove punctuations from our dataset
safaricom_df['Text Without Punctuation'] = safaricom_df['News'].apply(remove_punc)
print(safaricom_df.head(10))

# apply pre-processing to the text column
safaricom_df['Text Without Punc & Stopwords'] = safaricom_df['Text Without Punctuation'].apply(preprocess)
# join the words into a string
safaricom_df['Text Without Punc & Stopwords Joined'] = safaricom_df['Text Without Punc & Stopwords'].apply(
    lambda x: " ".join(x))
print(safaricom_df.head(10))


# %% reset pandas column display
pd.options.display.max_columns = 0
