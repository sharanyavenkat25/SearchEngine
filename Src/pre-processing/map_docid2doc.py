from os import listdir
from os.path import isfile, join
import pickle 

mypath = "/mnt/d/SearchEngine/data/Corpus/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
mapper={}
for i in range(len(onlyfiles)):
	mapper[i]=onlyfiles[i]

with open('mapper.pkl', 'wb') as f:
	pickle.dump(mapper,f)

