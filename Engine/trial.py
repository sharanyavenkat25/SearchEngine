import pickle

with open('inverted_index.pkl', 'rb') as d:
    docs1 = pickle.load(d)

print(len(docs1))
print(docs1["beena"])