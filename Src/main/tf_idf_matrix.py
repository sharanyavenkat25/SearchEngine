

def computeTF_IDF(docs):

	vectorizer = TfidfVectorizer()
	vectors = vectorizer.fit_transform(docs)
	feature_names = vectorizer.get_feature_names()
	dense = vectors.todense()
	denselist = dense.tolist()
	df = pd.DataFrame(denselist, columns=feature_names)
	return df