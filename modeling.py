# %% imports
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# %% import dataset
clean_stock_df = pd.read_csv('cleaned_text.csv')
pd.options.display.max_columns = None
print(clean_stock_df.head(10))

# modify column with list values for proper representation
clean_stock_df['Text Without Punc & Stopwords'] = clean_stock_df['Text Without Punc & Stopwords'].apply(eval)

# %% obtain total words present in dataset
list_of_words = []
for i in clean_stock_df['Text Without Punc & Stopwords']:
    for j in i:
        list_of_words.append(j)
print(list_of_words)

# obtain the total number of unique words
total_words = len(set(list_of_words))
print(total_words)

# %% split the data into test and train
X = clean_stock_df['Text Without Punc & Stopwords']
y = clean_stock_df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %%Create a tokenizer to tokenize the words and create sequences of tokenized words
tokenizer = Tokenizer(num_words=total_words)
tokenizer.fit_on_texts(X_train)

# Training data
train_sequences = tokenizer.texts_to_sequences(X_train)
# Testing data
test_sequences = tokenizer.texts_to_sequences(X_test)

print(train_sequences[:10])
print(test_sequences[:10])

print(print("The encoding for document\n", X_train[1:2], "\n is: ", train_sequences[1]))

# %% perform padding to ensure uniform text length
padded_train = pad_sequences(train_sequences, maxlen=29, padding='post', truncating='post')
padded_test = pad_sequences(test_sequences, maxlen=29, truncating='post')

for i, doc in enumerate(padded_train[:3]):
    print(f"The padded encoding for document: {i + 1} is: {doc}")

# %%Convert the data to categorical 2D representation
y_train_cat = to_categorical(y_train, 3)
y_test_cat = to_categorical(y_test, 3)

print(y_train_cat.shape, y_test_cat.shape)
print(y_train_cat[20:28])
print(y_train[20:28])
