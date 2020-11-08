import os
import numpy as np
import pandas as pd
import re
import string
import json
import nltk
import pickle
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 


directory = '/mnt/d/SearchEngine/Engine/Corpus/'

def clean(df):
	total_rows = df.shape[0]
	
	tokens=[]
	lemmatizer = WordNetLemmatizer()
	for i in range(total_rows):
		tokens_without_sw_row=[]
		document_test = re.sub(r'[^\x00-\x7F]+', ' ', df['text'][i])
		document_test = re.sub(r'@\w+', '', document_test)
		document_test = document_test.lower()
		document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)
		document_test = re.sub(r'[0-9]', '', document_test)
		document_test = re.sub(r'\s{2,}', ' ', document_test)
		text_tokens = word_tokenize(document_test)
		tokens_without_sw_row=[word for word in text_tokens if not word in stopwords.words()]
		lemmatized_tokens_row=[lemmatizer.lemmatize(w) for w in tokens_without_sw_row]
		tokens.append(lemmatized_tokens_row)
	# final=[lemmatizer.lemmatize(w) for w in tokens_without_sw]
	return tokens

def extract(directory):

	dictionary=[]
	for filename in os.listdir(directory):
		if filename.endswith(".csv"):
			print(filename)
			df = pd.read_csv(directory+filename,names=['text'])
			dictionary.append(clean(df))
	return dictionary

docs = extract(directory)


with open('token.pkl','wb') as f:
	pickle.dump(docs, f)

print(np.shape(docs))
print("Number of rows in doc 5", len(docs[5]))
print("Number of rows in doc 10",len(docs[10]))
print(docs[0][0])
print(docs[4][1])
print(docs[1][5])
print(docs[5][3])







		