import os
import pickle

with open('inverted_index_final.pkl', 'rb') as f:
		inverted = pickle.load(f)

permuterm = {}
with open('PermutermIndex.txt') as f:
	for line in f:
		temp = line.split(',')
		permuterm[temp[0]] = temp[1]

def bitwise_and(list1,list2):
	res=[x for x in list1 if x in list2]
	# res = list(set([tuple(sorted(ele)) for ele in list1]) & set([tuple(sorted(ele)) for ele in list2]))
	# print(res)
	return res


def prefix_match(term, prefix):
	term_list = []
	for tk in term.keys():
		if tk.startswith(prefix):
			term_list.append(term[tk])
	return term_list
		


def processQuery(query):    
	term_list = prefix_match(permuterm,query) 
	docID = []
	for term in term_list:
		term = term.strip("\n")
		print("Wildcard Query matched with : ",term)
		for docu in inverted[term]:
			rows = list(inverted[term][docu].keys())
			for r in rows:
				rand = (docu, r)
				docID.append(rand) 
	
	return docID

# parts = query.split("*")
def wildcard_queries(parts):
	
	#X*Y*Z
	if len(parts) == 3: 
		case = 4
	#X*
	elif parts[1] == '':
		case = 1
	#*Y
	elif parts[0] == '':
		case = 2
	#X*Y
	elif parts[0] != '' and parts[1] != '':
		case = 3

	#*Y*Z
	if case == 4:
		if parts[0] == '':
			case = 1

	
	
	#X*
	if case == 1:
		query = parts[0]
	#*Y  
	elif case == 2:
		query = parts[1] + "$"
	#X*Y
	elif case == 3:
		query = parts[1] + "$" + parts[0]

	#X*Y*Z
	elif case == 4:
		queryA = parts[2] + "$" + parts[0]
		queryB = parts[1]


		
	if case != 4:
		return (processQuery(query))

	elif case == 4:
		print("Query type : X*Y*Z")
		print(f"Query split as : {queryA} and {queryB}")
		print(f"Retrieving docs for {queryA} and {queryB}...taking intersection")
		term_list1 = prefix_match(permuterm,queryA)
		term_list2 = prefix_match(permuterm,queryB)

		docID1 = []
		for term in term_list1:
			term = term.strip("\n")
			for docu in inverted[term]:
				rows = list(inverted[term][docu].keys())
				for r in rows:
					rand = (docu, r)
					docID1.append(rand) 
		temp1 = docID1
		
		docID2 = []
		for term in term_list2:

			term = term.strip("\n")
			for docu in inverted[term]:
				rows = list(inverted[term][docu].keys())
				for r in rows:
					rand = (docu, r)
					docID2.append(rand) 

		temp2 = docID2

		print("temp 1 : ",sorted(temp1[:10]))
		print("temp 2 : ",sorted(temp2[:10]))
		temp = bitwise_and(temp1,temp2)
		return temp

	 





	

