# %% imports
import pandas as pd
from src.sentiment.remove_punctuation import remove_punc
from src.sentiment.preprocess import preprocess
from tensorflow import keras
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

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

# %% load tokenizer and model
with open('src/sentiment/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = keras.models.load_model('final_model/final_model.h5')

# %% Testing data
saf_test_data = safaricom_df['Text Without Punc & Stopwords']
# tokenize data
saf_test_sequences = tokenizer.texts_to_sequences(saf_test_data)

print(saf_test_sequences[:10])

print("The encoding for document\n", saf_test_data[4:5], "\n is: ", saf_test_sequences[4])

# pad data
padded_saf_test = pad_sequences(saf_test_sequences, maxlen=29, truncating='post')

for i, doc in enumerate(padded_saf_test[:6]):
    print(f"The padded encoding for document: {i + 1} is: {doc}")

# %% predict data
raw_predictions = model.predict(padded_saf_test)
print(raw_predictions[:10])

y_values = [0, 1, -1]
predictions = []
for i in raw_predictions:
    predictions.append(y_values[np.argmax(i)])

print(predictions[:10])
# %% add to dataframe
safaricom_df['raw_predictions'] = raw_predictions.tolist()
safaricom_df['predictions'] = predictions
print(safaricom_df.head(10))

# %% save data and predictions
safaricom_df.to_csv('data/predicted/safaricom_predictions.csv', index=False)

# %% reset pandas column display
pd.options.display.max_columns = 0
