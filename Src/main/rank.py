# How will cosine similarity work for queries like PES * ??
# Set the float precision - too many decimals
# Check for sentences having query appearing twice
# Nan issue - prince charles
# top three - does not print based on order

# generted tf_idf table again


#------------------
# dictionary - doc name first each dovc name has a dict where keys are rows and values are dict of words(key) and weights(value)
#------------------


import math
import time
import pickle
import csv
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
from nltk.stem import WordNetLemmatizer 


from query import query, free_text
print("Loading backend...")
with open('tf_idf_final.pkl','rb') as f:
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
		print("Similarity measure : ",'%.3f'%rank)
		print("------------------------------------------")
		

def print_table(t, no_of_docs):
	print("============================")
	print("Time (s)| No. of Documents |")
	print("----------------------------")
	print('%.2f'%t, '   |', no_of_docs, '               |')
	print("============================")
	

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




q=input('Enter your Query : ')
print("Your query is : ",q)

start = time.time()

tokens_q=q.split()

lemmatizer = WordNetLemmatizer()
tokens_q_lemmatized=[lemmatizer.lemmatize(w) for w in tokens_q]
print(tokens_q_lemmatized)
tokens_q=[x for x in tokens_q_lemmatized if x in index.keys()]

docs_rows = query(q)
print(docs_rows)

if docs_rows == []:
	docs_rows = free_text(q)
print(docs_rows)

query_string=''.join(tokens_q)
query_tfidf = computeTF_IDF([query_string])
query_tfidf = query_tfidf.sort_index(axis = 1)
query_list = query_tfidf.values.tolist()

final_ranks=[]
tfidf_weights={}
for i in docs_rows:
	docid=i[0]
	doc_name=doc_name_mapping[docid]
	rowid=i[1]
	for j in extracted[doc_name][rowid]:
		if j in tokens_q:
			tfidf_weights[j]=extracted[doc_name][rowid][j]

	#sort the dictionary
	# print(tfidf_weights)
	tfidf_weights={k:v for k, v in sorted(tfidf_weights.items(), key=lambda item: item[0])}

	# print(tfidf_weights)
	rank = 1 - spatial.distance.cosine(query_list[0], list(tfidf_weights.values()))
	# rank = cosine_similarity(query_list[0],list(tfidf_weights.values()))
	# print(rank)
	final_ranks.append(rank)
	tfidf_weights={}


print("Final Ranks : ")
ranked_docs={}
ranked_docs = {docs_rows[i]: final_ranks[i] for i in range(len(final_ranks))} 
ranked_docs={k:v for k, v in sorted(ranked_docs.items(), key=lambda item: -item[1])}

print_results(ranked_docs)

end = time.time()

t = end-start
# print("Total time taken: ", '%.2f'%t, " seconds")
print()
print_table(t, len(docs_rows))
print()

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


