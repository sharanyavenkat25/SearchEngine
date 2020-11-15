# How will cosine similarity work for queries like PES * ??

# Check for sentences having query appearing twice
# one word queries is an issue
# been* check 

import math
import os
import sys
import time
import pickle
import csv
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
from nltk.stem import WordNetLemmatizer 
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numpy as np
from numpy.linalg import norm
from numpy import dot
from query import query, free_text
from permuteindex import wildcard_queries

print("Loading backend files...")

with open('tfidf_vectorspace_new.pkl','rb') as f:
	extracted=pickle.load(f)

with open('mapper.pkl', 'rb') as f:
	doc_name_mapping = pickle.load(f)

with open('inverted_index_final.pkl', 'rb') as f:
	index = pickle.load(f)

print("Done loading!")
def retrieve_docs(tup, doc_name_mapping):
	
	doc_name = doc_name_mapping[tup[0]]
	df = pd.read_csv("/mnt/d/SearchEngine/data/Corpus/" + doc_name, header=None)
	# df = pd.read_csv("/Users/rohitpentapati/Documents/Niha/sem7/AIR/SearchEngine/data/Corpus/" + doc_name, header=None)
	content=df.iloc[[tup[1]]].values.tolist()[0][1]
	return (doc_name,content)

def print_results(ranked_docs):
	for i in ranked_docs:
		data=retrieve_docs(i,doc_name_mapping)
		rank=ranked_docs[i]
		print("Document : ",data[0],"Row Number : ",i[1])
		print("Text : ",data[1])
		print("Cosine Similarity : ",'%.3f'%rank)
		print("------------------------------------------")
		

def print_table(t, no_of_docs):
	print("============================")
	print("Time (s)| No. of Documents |")
	print("----------------------------")
	print('%.2f'%t, '   |', no_of_docs, '               |')
	print("============================")
	
def print_wildcard(final):
	for t in final:
		print("Document : ",t[0],"Row Number : ",t[1])
		print("Text : ",t[2])
		print("------------------------------------------")



def computeTF_IDF_for_query(query,words):

	vectorizer = TfidfVectorizer(vocabulary=words,use_idf=True)
	corpus = [query]
	query_vec = vectorizer.fit_transform(corpus)
	return query_vec
	
q=input('Enter your Query : ')
print("Your query is : ",q)
itr=int(input('Enter how many top searches you would like to see (Enter 0 if you want to retrive all Docs): '))



if(itr>0):  
	print("\nSearching Corpus....Retrieving top ",itr," documents \n")
elif(itr==0):
	print("\nSearching Corpus....Retrieving all documents \n")
else:
	print("Invalid value of number of docs required")
	exit()

if(len(q.split(' '))==1 and '*' in q):
	print(" Query entered is a Wildcard Query!\n")
	parts=q.split('*')
	start1 = time.time()

	docs_rows=wildcard_queries(parts)
	end1 = time.time()
	res=[]
	for i in docs_rows:
		meta=retrieve_docs(i,doc_name_mapping)
		res.append((meta[0],i[1],meta[1]))

	if(itr>0):
		final=res[:itr]
		print_wildcard(final)
	elif(itr==0):
		final=res
		print_wildcard(final)
	else:
		print("Invalid value of number of docs required")

	t1= end1-start1
	print()
	if(itr>0):
		if(itr<=len(docs_rows)):
			print_table(t1, itr)
		else:
			print_table(t1,len(docs_rows))

	else:
		print_table(t1, len(docs_rows))
	print()


else :

	#query pre processing
	

	tokens_q=q.split()
	lemmatizer = WordNetLemmatizer()
	tokens_q_lemmatized=[lemmatizer.lemmatize(w) for w in tokens_q]

	query_string=' '.join(tokens_q_lemmatized)
	query_vec=computeTF_IDF_for_query(query_string,index.keys())

	#getting docs
	start = time.time()
	docs_rows = query(q)

	if docs_rows == []:
		# print("in free text")
		docs_rows = free_text(q)


	#generating ranks
	final_ranks=[]
	sim={}
	for i in docs_rows:
		docid=i[0]
		doc_name=doc_name_mapping[docid]
		rowid=i[1]
		doc_vector=extracted[doc_name][rowid]
		d_vec=doc_vector.toarray()[0]
		q_vec=query_vec.toarray()[0]
		similarities=float(dot(d_vec,q_vec) /(norm(d_vec) * norm(q_vec)))
		final_ranks.append(similarities)

	end = time.time()


	print("\nRetrieved Documents  : \n")
	ranked_docs={}
	ranked_docs = {docs_rows[i]: final_ranks[i] for i in range(len(final_ranks))}
	if(itr>0): 
		ranked_docs={k:v for k, v in sorted(ranked_docs.items(), key=lambda item: -item[1])[:itr]}
	else:
		ranked_docs={k:v for k, v in sorted(ranked_docs.items(), key=lambda item: -item[1])}


	print_results(ranked_docs)

	t = end-start

	print()
	if(itr>0):
		if(itr<=len(docs_rows)):
			print_table(t, itr)
		else:
			print_table(t,len(docs_rows))

	else:
		print_table(t, len(docs_rows))
	print()



