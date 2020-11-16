import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')



with open('mapper.pkl', 'rb') as f:
	doc_name_mapping = pickle.load(f)

with open('inverted_index_final.pkl', 'rb') as f:
	index = pickle.load(f)


with open('token.pkl', 'rb') as f:
	token = pickle.load(f)




l = len(token)
doc_vecs={}
for i in range(l):
	l1=len(token[i])
	for j in range(l1):
		processed_tokens=' '.join(token[i][j])
		token[i][j]=processed_tokens
	doc_name=doc_name_mapping[i]
	doc_vecs[doc_name]=token[i]

#BERT
for i in doc_vecs.keys():
	print(i)
	rows=np.array(doc_vecs[i])
	rows=rows.flatten()
	doc_vecs[i]=list(rows)
	corpus=doc_vecs[i]
	corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
	doc_vecs[i]=corpus_embeddings
	
	




# new tfidf giving better ranks - lemmatised and tokenised

# words=index.keys()
# for i in doc_vecs.keys():
# 	print(i)
# 	vectorizer= TfidfVectorizer(vocabulary=words,use_idf=True)
# 	corpus=doc_vecs[i]
# 	x= vectorizer.fit_transform(corpus)
# 	doc_vecs[i]=x

	
	
				


# old tf-idf giving not so good ranks	

# documents=doc_name_mapping.values() #names of the document
# words = index.keys() # all the unique words of dictionary 
# doc_vecs={}
# lemmatizer = WordNetLemmatizer()
# for csv in documents:
# 	print("Vectorising doc : ",csv)
# 	vectorizer= TfidfVectorizer(vocabulary=words,use_idf=True)
# 	df = pd.read_csv(f'/mnt/d/SearchEngine/data/Corpus/{csv}',names=['rownum','Snippet'])
# 	corpus=list(df['Snippet'])
# 	x= vectorizer.fit_transform(corpus)
# 	doc_vecs[csv]=x
	
	
# print(doc_vecs['BBCNEWS.201701.csv'])


with open('tfidf_vectorspace_bert.pkl','wb') as f:
	pickle.dump(doc_vecs, f)

# print(doc_vecs['BBCNEWS.201701.csv'][1].toarray()[0])
# print(len(doc_vecs['BBCNEWS.201701.csv'][1].toarray()[0]))

