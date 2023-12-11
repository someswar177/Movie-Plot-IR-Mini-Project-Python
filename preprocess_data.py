import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

def preprocess(data):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    stemmer = SnowballStemmer(language='english')
    data = data.lower()
    data = re.sub(r'[^\w\s]', '', data)
    tokens = nltk.word_tokenize(data)
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Load the dataset
dataframe = pd.read_csv("IMDB dataset.csv")

# Preprocess the text data
dataframe['processed_text'] = dataframe.apply(lambda row: preprocess(' '.join(map(str, row))), axis=1)

# Save the processed text to a CSV file
dataframe['processed_text'].to_csv('processed_text.csv', index=False, header=True)

# Use TfidfVectorizer for TF-IDF calculation
tfidf_vectorizer = TfidfVectorizer(lowercase=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(dataframe['processed_text'])

# Save the TF-IDF matrix to a NumPy compressed archive
save_npz('tfidf_matrix.npz', tfidf_matrix)

# Save the vocabulary to a file with UTF-8 encoding
with open('vocabulary.txt', 'w', encoding='utf-8') as file:
    file.write('\n'.join(tfidf_vectorizer.get_feature_names_out()))
