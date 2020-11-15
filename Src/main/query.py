import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

### download the tokens from pickle file created
with open('inverted_index_final.pkl', 'rb') as f:
    docs = pickle.load(f)


def query(q):

	lemmatized_tokens_row = []
	tokens = []
	lemmatizer = WordNetLemmatizer()
	text_tokens = word_tokenize(q)
	tokens_without_sw_row=[word for word in text_tokens if not word in stopwords.words()]
	lemmatized_tokens_row=[lemmatizer.lemmatize(w) for w in tokens_without_sw_row]
	tokens.append(lemmatized_tokens_row)
	#print(docs["resurgence"])
	final = []
	for word in tokens[0]:
		res = []
		if word in docs:
			#print(docs[word])
			for docu in docs[word]:
					rows = list(docs[word][docu].keys())
					for r in rows:
						rand = (docu, r)
					
						res.append(rand)
		final.append(res)
	#print(final[1])
	leng = len(final)
	if(leng == 1 or leng == 0):
		return final
	elif(leng >= 1):
		final_docs = []
		for i in final[0]:
			if i in final[1]:
				final_docs.append(i)
		if(leng == 2):
			return final_docs
	
		for k in range(2, len(final)):
			temp = final_docs
			final_docs = []
			for j in final[k]:
				if j in temp:
					final_docs.append(j)
		return final_docs


def free_text(q):
	lemmatized_tokens_row = []
	tokens = []
	lemmatizer = WordNetLemmatizer()
	text_tokens = word_tokenize(q)
	tokens_without_sw_row=[word for word in text_tokens if not word in stopwords.words()]
	lemmatized_tokens_row=[lemmatizer.lemmatize(w) for w in tokens_without_sw_row]
	tokens.append(lemmatized_tokens_row)
	#print(docs["resurgence"])
	final = []
	res = []
	for word in tokens[0]:
		# res = []
		if word in docs:
			#print(docs[word])
			for docu in docs[word]:
					rows = list(docs[word][docu].keys())
					for r in rows:
						rand = (docu, r)
						res.append(rand)
	return res
