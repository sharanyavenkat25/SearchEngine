import os
import pandas as pd
import re
import string


import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

directory = '/mnt/d/SearchEngine/TelevisionNews/'
op_dir='/mnt/d/SearchEngine/Corpus/'


def create_corpus():
	for filename in os.listdir(directory):
		if filename.endswith(".csv"):
			print(filename, type(filename))
			df = pd.read_csv(directory+filename)
			df_text= df['Snippet']
			df_text.columns=['Snippet']

			# clean(df_text)

			outdir = '/mnt/d/SearchEngine/Corpus/'
			if not os.path.exists(outdir):
				os.mkdir(outdir)
			fullname = os.path.join(op_dir,filename)
			df_text.to_csv(fullname)

create_corpus()