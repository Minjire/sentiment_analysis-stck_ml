import nltk
import gensim
from nltk.corpus import stopwords

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
