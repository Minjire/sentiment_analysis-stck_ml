"""
Train model with all data using best metrics.
"""
# %% imports
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import pickle

# %% import dataset
clean_stock_df = pd.read_csv('data/processed/sentiment/cleaned_text.csv')
pd.options.display.max_columns = None
print(clean_stock_df.head(10))

# %%
print(clean_stock_df.isna().sum())
# drop null rows and reset index
clean_stock_df.dropna(inplace=True)
clean_stock_df.reset_index(drop=True, inplace=True)

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

print(X.shape, y.shape)

# %%Create a tokenizer to tokenize the words and create sequences of tokenized words
tokenizer = Tokenizer(num_words=total_words)
tokenizer.fit_on_texts(X)

# saving tokenizer for use in new test data
with open('src/sentiment/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Training data
train_sequences = tokenizer.texts_to_sequences(X)

print(train_sequences[:10])

print("The encoding for document\n", X[1:2], "\n is: ", train_sequences[1])

# %% perform padding to ensure uniform text length
padded_train = pad_sequences(train_sequences, maxlen=29, padding='post', truncating='post')

for i, doc in enumerate(padded_train[:3]):
    print(f"The padded encoding for document: {i + 1} is: {doc}")

# %%Convert the data to categorical 2D representation
y_cat = to_categorical(y, 3)

print(y_cat.shape)
print(y_cat[20:28])

# %% Build Deep Neural Network
# Sequential model
model = Sequential()

# embedding layer
model.add(Embedding(total_words, output_dim=512))
# Bi-Directional RNN and LSTM
model.add(LSTM(256))

# Dense layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# %%
# train model
# 8 epochs found to be optimal
EPOCHS = 8
model.fit(padded_train, y_cat, batch_size=32, validation_split=0.2, epochs=EPOCHS)

# %% save model
model.save("final_model/final_model.h5")

# %% reset pandas column display
pd.options.display.max_columns = 0
