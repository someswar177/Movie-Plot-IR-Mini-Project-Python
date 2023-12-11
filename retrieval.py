import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import math
import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy import spatial
import matplotlib.pyplot as plt
import pickle
import nltk

nltk.download('stopwords')
nltk.download('punkt')

def snow_ball_stemmer(data_list):
    stem = SnowballStemmer(language='english')
    return list(map(lambda word: stem.stem(word), data_list))

def text_cleaner(data_str):
    data_str = data_str.strip(' ').lower()
    data_str = data_str
    new_str = ''
    for char in data_str:
        if char not in '''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~''':
            new_str += char
    return new_str.strip(' ')

# Takes a list of words as parameter and returns a new list after removing the stopwords
def remove_stop_words(data_list):
    new_list = []
    for word in data_list:
        if word not in nltk.corpus.stopwords.words('english'):
            new_list.append(word)
    return new_list

def preprocess(data):
    return remove_stop_words(nltk.word_tokenize(text_cleaner(data)))

inverted_index = {}

# Takes a term and its corresponding documentID and adds it to the inverted index
def add_term_to_inverted_index(term, documentID):
    try:
        for document in inverted_index[term]['posting_list'].copy():
            if document['docID'] > documentID:
                inverted_index[term]['posting_list'].append({'docID': documentID, 'count': 1})
                inverted_index[term]['count'] = len(inverted_index[term]['posting_list'])
                inverted_index[term]['posting_list'] = sorted(inverted_index[term]['posting_list'], key=lambda x: x['docID'])
                break
            elif document['docID'] == documentID:
                document['count'] += 1
                break
        else:
            inverted_index[term]['posting_list'].append({'docID': documentID, 'count': 1})
            inverted_index[term]['count'] = len(inverted_index[term]['posting_list'])
            inverted_index[term]['posting_list'] = sorted(inverted_index[term]['posting_list'], key=lambda x: x['docID'])
    except KeyError:
        inverted_index[term] = {'count': 1, 'posting_list': [{'docID': documentID, 'count': 1}]}

dataframe_original = pd.read_csv(r"IMDB dataset.csv")

dataframe = dataframe_original.replace(['Unknown', 'unknown'], '')

total_doc = 5000
retrieved_doc = 30

for temp in dataframe.head(total_doc).iterrows():
    value = ''
    docID = temp[0]
    for temp_value in temp[1].to_dict().values():
        value += str(temp_value) + ' '
    value = value.replace('nan', '')
    processed_list = preprocess(value)
    for term in processed_list:
        add_term_to_inverted_index(term, docID)

def idf_func(c):
    return math.log2(total_doc / c)

def calc_idf():
    for term in inverted_index.keys():
        c = inverted_index[term]['count']
        idf_value = idf_func(c)
        inverted_index[term]['idf'] = idf_value

calc_idf()

# CALCULATING TF-IDF SCORE
tfidf_dict = {}
tfidf_list = []

for term in inverted_index.keys():
    tfidf_list.append(term)
    temp = inverted_index[term]['posting_list']
    for t in temp:
        tfidf_dict[t['docID'], term] = inverted_index[term]['idf'] * (1 + math.log2(t['count']))

D = np.zeros((total_doc, len(inverted_index)))

for i in tfidf_dict:
    k = tfidf_list.index(i[1])
    D[i[0]][k] = tfidf_dict[i]

def cosine_similarity1(query_dict):
    Q = np.zeros((len(inverted_index)))
    res = []

    for i in inverted_index.keys():
        k = tfidf_list.index(i)
        if i in query_dict.keys():
            Q[k] = query_dict[i]

    for row in range(D.shape[0]):
        result = 1 - spatial.distance.cosine(D[row], Q)
        res.append([row, result])

    return res

query = input("Enter your query: ")
q_list = preprocess(query)
query_dict = {}

for q in q_list:
    if q in inverted_index.keys():
        query_dict[q] = (inverted_index[q]['idf']) * (1 + math.log2(q_list.count(q)))

res = cosine_similarity1(query_dict)
sorted_res = sorted(res, key=lambda x: x[1], reverse=True)

# Displaying the retrieved documents
pd.set_option('display.max_colwidth', None)

# Ensure that `retrieved_doc` does not exceed the length of `sorted_res`
if retrieved_doc > len(sorted_res):
    retrieved_doc = len(sorted_res)

# Access the first `retrieved_doc` rows
dataframe.iloc[[docid[0] for docid in sorted_res[:retrieved_doc]]].head(retrieved_doc)


# Relavence feedback
# Recall and Precision
# Recall = #(relevant retrieved) / #(relevant) -> tp / tp + fn
# Precision = #(relevant retrieved) / #(retrieved) -> tp / tp + fp

print('Enter 1 if the document is relevant\n0 if the document is irrelevant')
feedback = []

for f in range(retrieved_doc):
    print('Is the document', f, 'relevant')
    x = int(input())
    feedback.append(x)

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

    print("RECALL", recall)
    print("PRECISION", precision)

evaluate(feedback)
