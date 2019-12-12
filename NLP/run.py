import pickle
import project_nlp as project_nlp

fname = './Data/sample_documents.pickle'
documents = pickle.load(open(fname,"rb"))

## Step- 1. Construct the index...
index = project_nlp.InvertedIndex()
index.index_documents(documents)

## Test cases
Q = 'New York Times Trump travel'
DoE = {'New York Times':0, 'New York':1,'New York City':2}
doc_id = 3

## 2. Split the query...
query_splits = index.split_query(Q, DoE)

## 3. Compute the max-score...
result = index.max_score_query(query_splits, doc_id)
print('Test1 The maximum score:')
print(result)


documents = {1: 'According to Times of India, President Donald Trump was on his way to New York City after his address at UNGA.',
             2: 'The New York Times mentioned an interesting story about Trump.',
             3: 'I think it would be great if I can travel to New York this summer to see Trump.'}
index = project_nlp.InvertedIndex()
index.index_documents(documents)

## Test cases
Q = 'The New New York City Times of India'
DoE = {'Times of India':0, 'The New York Times':1,'New York City':2}
doc_id = 1

## 2. Split the query...
query_splits = index.split_query(Q, DoE)

## 3. Compute the max-score...
result = index.max_score_query(query_splits, doc_id)
print('Test2 The maximum score:')
print(result)


documents = {1: 'According to Los Angeles Times, The Boston Globe will be experiencing another recession in 2020. However, The Boston Globe decales it a hoax.',
             2: 'The Washington Post declines the shares of George Washington.',
             3: 'According to Los Angeles Times, the UNSW COMP6714 students should be able to finish project part-1 now.'}
index = project_nlp.InvertedIndex()
index.index_documents(documents)

## Test cases
Q = 'Los The Angeles Boston Times Globe Washington Post'
DoE = {'Los Angeles Times':0, 'The Boston Globe':1,'The Washington Post':2, 'Star Tribune':3}
doc_id = 1

## 2. Split the query...
query_splits = index.split_query(Q, DoE)

## 3. Compute the max-score...
result = index.max_score_query(query_splits, doc_id)
print('Test3 The maximum score:')
print(result)