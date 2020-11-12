# How will cosine similarity work for queries like PES * ??
# Set the float precision - too many decimals
# Check for sentences having query appearing twice

import math
import pickle
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from query import query

def retrieve_docs(docs_rows, doc_name_mapping):
	docs = []
	for tup in docs_rows:
		doc_name = doc_name_mapping[tup[0]]
		df = pd.read_csv("/Users/rohitpentapati/Documents/Niha/sem7/AIR/SearchEngine/data/Corpus/" + doc_name, header=None)
		docs.append(df.iloc[[tup[1]]].values.tolist()[0][1])
		
	return docs


def computeTF_IDF(docs):

	vectorizer = TfidfVectorizer()
	vectors = vectorizer.fit_transform(docs)
	feature_names = vectorizer.get_feature_names()
	dense = vectors.todense()
	denselist = dense.tolist()
	df = pd.DataFrame(denselist, columns=feature_names)
	return df

def extract_query_columms(df, q):
	data = {}
	for word in q:
		df1 = df[word]
		df1 = df1.tolist()
		data[word] = df1
	# print(data)
	return(pd.DataFrame(data))

def dot_product(query_list, document):
	val = 0
	l = len(query_list)
	for i in range(l):
		val+= query_list[i]*document[i]
	return val

def magnitude(q):
	val = 0
	l = len(q)
	for i in range(l):
		val+=q[i]*q[i]
	val = math.sqrt(val)
	return val

def cosine_similarity(query_list, document):
	return((dot_product(query_list, document))/(magnitude(query_list)*magnitude(document)))


# sentence1 = "The tiger (Panthera tigris) is the largest extant cat species and a member of the genus Panthera."
# sentence2 = "The tiger is among the most recognisable and popular of the world's charismatic megafauna tiger."
# senetence3 = "Tiger is scattered throughout sub-Saharan Africa, South Asia, and Southeast Asia and are found in different habitats, including savannahs, forests, deserts, and marshes. They are herbivorous, and they stay near water when it is accessible."

# query = "tiger"

with open('mapper.pkl', 'rb') as f:
    doc_name_mapping = pickle.load(f)

docs_rows = query('climate change')


docs = retrieve_docs(docs_rows, doc_name_mapping)
# print(docs)



# docs_tfidf = computeTF_IDF([sentence1, sentence2, senetence3])
docs_tfidf = computeTF_IDF(docs)
print(docs_tfidf)
# [query] = list of words in the query
# query_tfidf = computeTF_IDF(['tiger cat'])
query_tfidf = computeTF_IDF(['climate change'])
query_tfidf = query_tfidf.sort_index(axis = 1)

extracted_tfidf = extract_query_columms(docs_tfidf, ['climate','change'])
extracted_tfidf = extracted_tfidf.sort_index(axis = 1)

print(extracted_tfidf)

# print(docs_tfidf)
# print(docs_tfidf['tiger'])

# print(query_tfidf)

query_list = query_tfidf.values.tolist()
print(query_list)
print("\n")

ranks = []
l = len(docs)
for i in range(l):
	doc = extracted_tfidf.iloc[[i]]
	doc = doc.values.tolist()
	print(doc)
	rank = cosine_similarity(query_list[0],doc[0])
	ranks.append(rank)
	# print(ranks)

print("Final Ranks", ranks)


