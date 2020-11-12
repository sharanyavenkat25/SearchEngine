import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
### download the tokens from pickle file created
with open('token.pkl', 'rb') as f:
    docs = pickle.load(f)




print(docs[22][945])
posting = {}
for document in range(0,len(docs)):

	for row in range(0, len(docs[document])):
		
		
		#ctr=0
		for word in docs[document][row]:
			
			res = {}
			res[row] = []
			occurrences = lambda s, lst: (i for i,e in enumerate(docs[document][row]) if e == word)
			ii = list(occurrences(word, docs[document][row]))
			
			
			res[row] = ii
			if word in posting:
				if document in posting[word]:
					posting[word][document].update(res)
				else:
					temp ={}
					temp[document] = res
					posting[word].update(temp)
			else: 
				
				temp = {}
				temp[document] = res
				#print(posting)
				posting[word] = temp
			#print(word, posting)

		#break
		



with open ('inverted_index_final.pkl', "wb") as r:
	pickle.dump(posting,r)
	#def query(q, final_post):
	

