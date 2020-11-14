# How will cosine similarity work for queries like PES * ??
# Set the float precision - too many decimals
# Check for sentences having query appearing twice

#------------------
# dictionary - doc name first each dovc name has a dict where keys are rows and values are dict of words(key) and weights(value)
#------------------

import math
import pickle
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial


from query import query

with open('tfidf.pickle','rb') as f:
	extracted=pickle.load(f)

with open('mapper.pkl', 'rb') as f:
    doc_name_mapping = pickle.load(f)

with open('inverted_index_final.pkl', 'rb') as f:
    index = pickle.load(f)
# def retrieve_docs(docs_rows, doc_name_mapping):
# 	docs = []
# 	for tup in docs_rows:
# 		doc_name = doc_name_mapping[tup[0]]
# 		df = pd.read_csv("/mnt/d/SearchEngine/data/Corpus/" + doc_name, header=None)
# 		docs.append(df.iloc[[tup[1]]].values.tolist()[0][1])
		
# 	return docs


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
	print(len(query_list),len(document))
	return((dot_product(query_list, document))/(magnitude(query_list)*magnitude(document)))


# sentence1 = "The tiger (Panthera tigris) is the largest extant cat species and a member of the genus Panthera."
# sentence2 = "The tiger is among the most recognisable and popular of the world's charismatic megafauna tiger."
# senetence3 = "Tiger is scattered throughout sub-Saharan Africa, South Asia, and Southeast Asia and are found in different habitats, including savannahs, forests, deserts, and marshes. They are herbivorous, and they stay near water when it is accessible."

# query = "tiger"


q='global warming'
tokens_q=q.split()
tokens_q=[x for x in tokens_q if x in index.keys()]

docs_rows = query(q)

print("--------------Documents retirved from query.py-------------------")
print(docs_rows)
#[(docid,rowid)]
query_string=''.join(tokens_q)
query_tfidf = computeTF_IDF([query_string])
query_tfidf = query_tfidf.sort_index(axis = 1)
query_list = query_tfidf.values.tolist()

final_ranks=[]
tfidf_weights=[]
for i in docs_rows:
	docid=i[0]
	doc_name=doc_name_mapping[docid]
	rowid=str(i[1])
	for j in extracted[doc_name][rowid]:
		if j in tokens_q:
			print("Token found! :",j)
			tfidf_weights.append(extracted[doc_name][rowid][j])
	
	print(tfidf_weights)
	rank = 1 - spatial.distance.cosine(query_list[0], tfidf_weights)
	# rank = cosine_similarity(query_list[0],tfidf_weights)
	print(rank)
	final_ranks.append(rank)
	tfidf_weights=[]

print("Final Ranks : ")
print(final_ranks)


# # docs = retrieve_docs(docs_rows, doc_name_mapping)
# # print(docs)



# # docs_tfidf = computeTF_IDF([sentence1, sentence2, senetence3])
# docs_tfidf = computeTF_IDF(docs)
# print(docs_tfidf)
# # [query] = list of words in the query
# # query_tfidf = computeTF_IDF(['tiger cat'])

# extracted_tfidf = extract_query_columms(docs_tfidf, ['global','warming'])
# extracted_tfidf = extracted_tfidf.sort_index(axis = 1)

# print(extracted_tfidf)

# # print(docs_tfidf)
# # print(docs_tfidf['tiger'])

# # print(query_tfidf)

# # query_list = query_tfidf.values.tolist()
# # print(query_list)
# # print("\n")

# ranks = []
# l = len(docs)
# for i in range(l):
# 	doc = extracted_tfidf.iloc[[i]]
# 	doc = doc.values.tolist()
# 	print(doc)
# 	rank = cosine_similarity(query_list[0],doc[0])
# 	ranks.append(rank)
# 	# print(ranks)

# print("Final Ranks", ranks)


