# SearchEngine
Build a search engine for Environmental News NLP archive

## Implementation
Implemented a vector space model for providing ranked results for the document search engine built on News NLP Archive in python.

- Our search engine implementation uses hash tables to store the inverted index. The inverted index stores docID, rowID and positions of tokens in each sentence.
- The engine supports free text, phrase as well as wildcard queries
- Two vector models are used to generate embeddings from text - Tfidf and BERT
- Ranking is done using Cosine similarity of embeddings from queries and matching text from docs

## Running files
Original dataset can be found in SearchEngine/data/TelevisionNews

Refined Snippets from original dataset can be found in SearchEngine/data/Corpus
```
cd SearchEngine/Src/main
python3 rank.py
```

