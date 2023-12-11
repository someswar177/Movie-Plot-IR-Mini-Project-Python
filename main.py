from flask import Flask, render_template, request,jsonify
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz
import os

app = Flask(__name__)

nltk.download('stopwords')
nltk.download('punkt')

retrieved_doc = 15

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

# Check if processed_text.csv exists
if not os.path.exists('processed_text.csv'):
    print("Processing and saving text...")
    dataframe['processed_text'] = dataframe.apply(lambda row: preprocess(' '.join(map(str, row))), axis=1)
    dataframe['processed_text'].to_csv('processed_text.csv', index=False, header=True)
else:
    print("processed_text.csv already exists. Loading...")
    # Load the preprocessed text
    processed_text_df = pd.read_csv('processed_text.csv')
    if 'processed_text' in processed_text_df.columns:
        dataframe['processed_text'] = processed_text_df['processed_text']
    else:
        print("Error: 'processed_text' column not found in processed_text.csv")

# Initialize TF-IDF vectorizer outside the function
tfidf_vectorizer = TfidfVectorizer(lowercase=True)

# Check if tfidf_matrix.npz exists
if not os.path.exists('tfidf_matrix.npz'):
    print("Calculating TF-IDF matrix...")
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataframe['processed_text'])
    save_npz('tfidf_matrix.npz', tfidf_matrix)
    print("tfidf_matrix.npz is saved")
    with open('vocabulary.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(tfidf_vectorizer.get_feature_names_out()))
        print("vocabulary.txt is created")
else:
    print("tfidf_matrix.npz already exists. Loading...")
    # Load pre-fitted TF-IDF matrix and vocabulary
    tfidf_matrix = load_npz('tfidf_matrix.npz')
    with open('vocabulary.txt', 'r', encoding='utf-8') as file:
        vocabulary = file.read().splitlines()

    # Use the existing vocabulary
    tfidf_vectorizer.vocabulary_ = {word: index for index, word in enumerate(vocabulary)}

# Fit the vectorizer if it's not already fitted
if not tfidf_vectorizer.get_params()['stop_words']:
    tfidf_vectorizer.fit_transform(dataframe['processed_text'])

# Preprocess the text data (only if not already preprocessed)
if 'processed_text' not in dataframe.columns:
    dataframe['processed_text'] = dataframe.apply(lambda row: preprocess(' '.join(map(str, row))), axis=1)

def evaluate(feedback):
    retrieved = 30
    recall = []
    precision = []
    relevant = feedback.count(1)
    relevant_count = 0

    for ct in range(retrieved_doc):
        if feedback[ct] == 1:
            relevant_count += 1
        recall.append(relevant_count / relevant)
        precision.append(relevant_count / (ct + 1))

    print("\n")
    print("RECALL", recall)
    print("\n")
    print("PRECISION", precision)
    print("\n")
    return recall, precision

dataframe['relevance_score'] = 0.0

@app.route('/', methods=['GET', 'POST'])
def index():
    recall = None
    precision = None

    if request.method == 'POST':
        query = request.form['query']

        # Preprocess the query
        query_tfidf = tfidf_vectorizer.transform([preprocess(query)])

        # Calculate cosine similarity using NumPy
        cosine_similarities = np.dot(tfidf_matrix, query_tfidf.T).toarray().flatten()

        # Get the indices of the top retrieved documents
        retrieved_indices = cosine_similarities.argsort()[-retrieved_doc:][::-1]

        # Display the retrieved documents with all columns
        result_df = dataframe.iloc[retrieved_indices]

        return render_template('index.html', query=query, results=result_df.to_dict(orient='records'))

    if request.method == 'GET' and 'feedback' in request.args:
        # Retrieve the feedback values from the query parameters
        feedback = list(map(int, request.args.get('feedback').split(',')))

        # Calculate precision and recall
        recall, precision = evaluate(feedback)

        # Pass the results (recall and precision) as JSON response along with the feedback values
        return jsonify({'recall': recall, 'precision': precision, 'feedback': feedback})

    return render_template('index.html', recall=recall, precision=precision)

if __name__ == '__main__':
    app.run(debug=False)