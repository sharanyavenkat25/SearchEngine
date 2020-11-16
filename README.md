# SearchEngine
Build a search engine for Environmental News NLP archive

## Implementation
Implemented a vector space model for providing ranked results for the document search engine built on News NLP Archive in python.

- Our search engine implementation uses hash tables to store the inverted index. The inverted index stores docID, rowID and positions of tokens in each sentence.
- The engine supports free text, phrase as well as wildcard queries
- Two vector models are used to generate embeddings from text - Tfidf and BERT
- Ranking is done using Cosine similarity of embeddings from queries and matching text from docs

## Running files
All dataset info can be found in SearchEngine/data
```
Original dataset can be found in SearchEngine/data/TelevisionNews

Refined Snippets from original dataset can be found in SearchEngine/data/Corpus
```

All source code can be found in SearchEngine/Src

SearchEngine/Src contains multiple pickle files which store our inverted index, permuterm index and tdidf weight matrix as well as embeddings for each document.

To run Tfidf model - 
```
cd SearchEngine/Src/main
python3 rank_tfidf.py
```
To run BERT model - 
```
cd SearchEngine/Src/main
python3 rank_bert.py
```

