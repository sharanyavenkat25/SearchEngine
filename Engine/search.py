import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
### download the tokens from pickle file created
with open('token.pkl', 'rb') as f:
    docs = pickle.load(f)





print(docs[34][1334])
print(len(docs))
#print(docs[0][0])
#[[[word1,word2,word3], [line2]]]

posting = {}
for document in range(0,len(docs)):

	for row in range(0, len(docs[document])):
		
		
		#ctr=0
		for word in docs[document][row]:
			
			res = {}
			res[row] = []
			occurrences = lambda s, lst: (i for i,e in enumerate(docs[document][row]) if e == word)
			ii = list(occurrences(word, docs[document][row]))
			#print(word)
			#ii = np.where(docs[document][row] == word)[0]
			#if word == "beena":
				#print(index)
			
			
			#w = docs[document][row][word]
			
			if word in posting:
				res[row] = ii
				#print(temp)
				'''if row in posting[word][document]:
						try:
								posting[word][document][row].append[index]
						except KeyError:
							pass
				#res[row].append(index)'''
				temp ={}
				temp[document] = res
				posting[word].update(temp)
			else: 
				res[row] = ii
			#ctr+=1
				temp = {}
				temp[document] = res
				#print(posting)
				posting[word] = temp
			#print(word, posting)

		#break
		
#{"beena": {0: {0: [17]}, 24: {72: [32]}}
print(posting)
'''posting = {} #for dictionary 
for doc in range(0, len(docs)):
	posting[doc+1] = docs[doc]

final_post = {}
for docu in posting:
	for term in posting[docu]:
		
			if term in final_post and (docu not in final_post[term]) :
					final_post[term].append(docu)
			else:
					final_post[term] = [docu]

#print(final_post)'''
with open ('inverted_index.pkl', "wb") as r:
	pickle.dump(posting,r)
def query(q, final_post):
	print("query:", q)
	query = q.split()
	res = []
	for word in query:
		#matching = [s for s in final_post if word in s] #for regex matching -- gives all combos of regex
		#for match in matching:
		if word in final_post:
			res.append(final_post[word])
	#print(res)
	if(len(res)>0):
		check = set.intersection(*[set(list) for list in res])
		print(check)

#query("different", final_post)



'''