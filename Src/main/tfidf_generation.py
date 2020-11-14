# TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
# IDF(t) = log_e(Total number of documents / Number of documents with term t in it).

# inverted table -> {word:{doc_id:{row_id1:[21,20],row_id2:[1]}}}
# [[document[row]]]
# tf_idf table -> {doc_name:{row_number:{word1:weight1,word2:weight2}}}


import pickle


with open('token.pkl', 'rb') as f:
	list_tokens = pickle.load(f)

with open('mapper.pkl', 'rb') as f:
	doc_name_mapping = pickle.load(f)

with open('inverted_index_final.pkl', 'rb') as f:
	index = pickle.load(f)

total_corpus = 0
for doc in list_tokens:
	total_corpus+=len(doc)

idf = {}

for word in index.keys():
	count = 0
	for doc in index[word].keys():
		count+=len(index[word][doc].keys())
	idf[word] = total_corpus/count

# print(idf)

tf ={}
l = len(list_tokens)

for i in range(l):
	print(i)
	dn = doc_name_mapping[i]
	tf[dn] = {}
	for row in range(len(list_tokens[i])):
		tf[dn][row] = {}
		total_words_row = len(list_tokens[i][row])
		for word in list_tokens[i][row]:
			tf_value = list_tokens[i][row].count(word)/total_words_row
			idf_value = idf[word]
			tf[dn][row][word] = tf_value/idf_value

# print(tf)

with open('tf_idf_final.pkl', 'wb') as f:
	pickle.dump(tf,f)
	

