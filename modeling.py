# %% imports
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# %% import dataset
clean_stock_df = pd.read_csv('data/processed/cleaned_text.csv')
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
model.fit(padded_train, y_train_cat, batch_size=32, validation_split=0.2, epochs=EPOCHS)

# %% save model
model.save("models/" + str(EPOCHS) + "_epochs_model.h5")

# %% Assess Trained Model
pred = model.predict(padded_test)
print(pred[:10])

# %% get predictions
y_values = [0, 1, -1]
prediction = []
for i in pred:
    prediction.append(y_values[np.argmax(i)])

# list containing original values
original = []
for i in y_test_cat:
    original.append(y_values[np.argmax(i)])

accuracy = accuracy_score(original, prediction)
print(accuracy)

# %% Plot the confusion matrix, classification report
report = classification_report(original, prediction)
print(f"Classification Report:\n{report}")
cm = confusion_matrix(original, prediction)
print(f"Confusion Matrix:\n{cm}")
sns.heatmap(cm, annot=True, fmt="d", xticklabels=y_values, yticklabels=y_values)
plt.savefig("figures/" + str(EPOCHS) + "_epochs Confusion matrix.png", bbox_inches='tight')
plt.show()

# %% print results to text document
original_stdout = sys.stdout
with open("reports/Model_Reports.txt", 'a') as f:
    sys.stdout = f
    print("***** " + str(EPOCHS) + " EPOCHS ******\n")
    print("Test Accuracy:")
    print(accuracy)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)
    sys.stdout = original_stdout

# %% reset pandas column display
pd.options.display.max_columns = 0
