import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
import nltk

nltk.download('stopwords')

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

def remove_stop_words(data_list):
    new_list = []
    for word in data_list:
        if word not in nltk.corpus.stopwords.words('english'):
            new_list.append(word)
    return new_list

def preprocess(data):
    return remove_stop_words(nltk.word_tokenize(text_cleaner(data)))

inverted_index = {}

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