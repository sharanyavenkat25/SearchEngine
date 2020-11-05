import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

### download the tokens from pickle file created
with open('docs_token_list.pkl', 'rb') as f:
    docs = pickle.load(f)


# Instantiate a TfidfVectorizer object
vectorizer = TfidfVectorizer()
# It fits the data and transform it as a vector
X = vectorizer.fit_transform(docs)
# Convert the X as transposed matrix
X = X.T.toarray()
# Create a DataFrame and set the vocabulary as the index
df = pd.DataFrame(X, index=vectorizer.get_feature_names())
print(df.head())
print(df.shape)


def get_similar_articles(q, df):
  print("query:", q)
  q = [q]
  q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0],)
  sim = {}
  for i in range(10):
    sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
  
  sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
  
  for k, v in sim_sorted:
    if v != 0.0:
      print("Similar Articles:", v)
      print(docs[k])
      print()


# q1 = 'barcelona'
# q2 = 'gareth bale'
# q3 = 'shin tae yong'

# get_similar_articles(q1, df)
# print('-'*100)
# get_similar_articles(q2, df)
# print('-'*100)
# get_similar_articles(q3, df)



