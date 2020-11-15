import pickle

with open('inverted_index_final.pkl', 'rb') as f:
	index = pickle.load(f)

def rotate(str, n):
    return str[n:] + str[:n]

file = open("PermutermIndex.txt","w")
keys = sorted(index.keys())
for key in keys:
	print(key)
	dkey = key + "$"
	for i in range(len(dkey),0,-1):
		out = rotate(dkey,i)
		file.write(out)
		file.write(",")
		file.write(key)
		file.write("\n")
	
file.close()