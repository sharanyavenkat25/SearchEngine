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


directory = '/mnt/d/SearchEngine/Corpus/'

def clean(df):
	total_rows = df.shape[0]
	
	tokens_without_sw=[]
	lemmatizer = WordNetLemmatizer()
	for i in range(total_rows):
		document_test = re.sub(r'[^\x00-\x7F]+', ' ', df['text'][i])
		document_test = re.sub(r'@\w+', '', document_test)
		document_test = document_test.lower()
		document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)
		document_test = re.sub(r'[0-9]', '', document_test)
		document_test = re.sub(r'\s{2,}', ' ', document_test)
		text_tokens = word_tokenize(document_test)
		tokens_without_sw.extend([word for word in text_tokens if not word in stopwords.words()])
	final=[lemmatizer.lemmatize(w) for w in tokens_without_sw]
	return final

def extract(directory):

	dictionary=[]
	for filename in os.listdir(directory):
		if filename.endswith(".csv"):
			print(filename)
			df = pd.read_csv(directory+filename,names=['text'])
			dictionary.append(clean(df))
	return dictionary

docs = extract(directory)

with open('docs_token_list.pkl','wb') as f:
	pickle.dump(docs, f)

print(np.shape(docs))






		