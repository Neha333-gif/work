import time 
import json 
import os 
from typing import List, Dict, Any 
import numpy as np 
from collections import Counter, defaultdict 
import math 

# --------------chroma and embedding ------------------

import chromadb 
from sentence_transformers import SentenceTransformer

# ---------------- BM25 from rank_b25------------------

from rank_bm25 import BM25Okapi

# config 
# import folder 

folder_path = r"C:\Users\peepl\Downloads\latency_tickets\tickets_latency"

text_fields = ["title", "description"]

repeat = 5

# number of top results we require in output 

top_k = 3


# embedding model 

embedding_model = "all-MiniLM-L6-v2"

k1 = 1.5 
b = 0.75


# ----- load tickets -----

def load_tickets(folder_path):
    tickets = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".json"):
            try:
                with open(os.path.join(folder_path, filename), encoding = "utf-8") as f:
                    data = json.load(f)
                    if "ticket_id" not in data:
                        data["ticket_id"] = filename.replace(".json", "")
                    tickets.append(data)
            except :
                pass    

    print(f"loaded {len(tickets)} tickets from folder path {folder_path}")   
    return tickets 


# ----- prepare documents --------------

def get_docs(tickets):
    docs = []
    for t in tickets:
        text = " ".join(str(t.get(f, "")) for f in text_fields)
        docs.append(text)
    return docs 


# bm25 setup and search 

def build_bm25_index(docs): 
    tokenized_docs = [doc.lower().split() for doc in docs]
    N = len(tokenized_docs)
    if N == 0:
        return {"avgdl" : 0, "N" : 0, "idf" : {}, "doc_length" : [], "tokenized_docs" : [] }

    doc_length = [len(doc) for doc in tokenized_docs]
    avgdl = sum(doc_length)/N  

    df = defaultdict(int)

    for tokens in tokenized_docs: 
        # if a term present in a line 
        for term in set(tokens):
            # take a set to eliminate duplicates cuz we only need how many docs the term  present in
            df[term] += 1

    idf = {}
    for term, freq in df.items():
        idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    
    return {
        "tokenized_docs" : tokenized_docs,
        "doc_lengths": doc_length,
        "avgdl": avgdl,
        "idf": idf,
        "N": N
    }



""" 
docs = [ "dog cat bird", " cat dog cat"]
tokenized_docs = [["dog", "cat", "bird], ["cat", "dog", "cat"]]
doc_length = [3, 3]
avgdl = (2*3)/2 = 3
df = {cat : 2, dog : 2, bird : 1} # document length
idf : formula # higher for bird cuz it appear only in 1 doc

"""


def bm25_score(query_tokens, doc_idx: int, index: Dict):
    score = 0.0 
    doc = index["tokenized_docs"][doc_idx]
    dl = index["doc_lengths"][doc_idx]
    avgdl = index["avgdl"]
    counter = Counter(doc)


    for term in query_tokens:
        if term not in index["idf"]: 
            # ignore the tokens which arent present in the document but present in query
            continue 
        tf = counter.get(term, 0) # get counter value else assign 0
        if tf == 0: # if 0 then ignore
            continue 
        sat_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
        score += sat_tf * index["idf"][term]
    return score 

def bm25_search(query: str, index: Dict, top_k: int):
    query_tokens = query.lower().split()
    if not query_tokens:
        return np.array([]), np.array([])
    
    scores = np.array([bm25_score(query_tokens, i, index) for i in range(index["N"])])
    ranked_idx = np.argsort(scores)[::-1][:top_k] 
    return ranked_idx, scores[ranked_idx]

"""

if theres a query ["cat dog cat frog"]

-> ["cat", "dog", "frog"]
tf= {"cat" : 2, "dog" : 1, "frog" : 1}
sat_tf will be high for cat because it repeats twice then add the scores for document
for frog there least sat_tf because its not there in doc above so least value 

"""


# -------  tf-idf -----------

def build_tf_idf_vectors(docs):
    all_tokens = [text.lower().split() for text in docs]
    vocab = sorted(set(w for tokens in all_tokens for w in tokens))
    V = len(vocab)
    N = len(all_tokens)

    if N == 0 or V == 0:
        return np.zeros((N, V)), np.array([]), {}
    
    term2idx = {term : i for i, term in enumerate(vocab)} 
    # here all tokens are set which means duplicates eliminated
    # i is index given to each term in tokens

    doc_freq = np.zeros(V)

    for tokens in all_tokens: # for all doc in document
        for t in set(tokens): # set the doc
            if t in term2idx: # if terms present in term2idx
                doc_freq[term2idx[t]] += 1 # increase counter for terms

    idf = np.log(N / (doc_freq + 1)) + 1

    vectors = np.zeros((N, V)) # rows of documents and columns of words in them
    for i, tokens in enumerate(all_tokens): # i is index for each row
        counter = Counter(tokens) # for all terms in tokens set counter for term frequency
        for term, tf in counter.items():
            if term in term2idx: # if term present in set -> term2idx (has a order)
                vectors[i, term2idx[term]] = tf * idf[term2idx[term]] # the term is created in ith row or row of its doc and assigned value

    # norm for the rows 
    norms = np.linalg.norm(vectors, axis = 1, keepdims = True) # normalize it since we have to get length for eucledian distance
    norms[norms == 0] = 1 # if norms == 0 for a axis = 1 for a row then it must be 1 else it will crash
    vectors = vectors / norms # divide so that total of the axis will be one after we divide it (unit length)

    return vectors, idf, term2idx 

""" 
ex -> 
docs = ["cat dog cat", "cat dog bird] -> len -> 2
vocab = ["bird", "cat", "dog"] -> len -> 3
term2idx = { "bird" : 0, "cat" : 1, "dog" : 2}
doc_freq = [0, 0, 0]
         = [1, 2, 2]
idf = bird gets highest value 
      dog = cat 
      [0.595, 0.595, 0.595] -> cat dog cat -> doc 0
      [0.595, 0.595, 1] -> cat dog bird -> doc 1
after normalization:

  doc 0 : (2*0.595 +  1*0.595)^(0.5) = 1.307
  doc 1 : (1*0.595 + 1*0.595 + 1*1)^(0.5) = 1.33

  bird gets highest weight cuz its rare 


"""

def tfidf_cosine_search(query: str, vectors, idf, term2idx, top_k : int):
    query_tokens = query.lower().split()
    q_vec = np.zeros(len(idf)) # creates q_vec only for terms present in term2idx
    q_counter = Counter(query_tokens)
    for term, tf in q_counter.items():
        if term in term2idx:
            # if query token not present already in term2idx then ignore which means not present in document
            q_vec[term2idx[term]] = tf * idf[term2idx[term]]

    q_norm = np.linalg.norm(q_vec) # add values to divide it later
    if q_norm > 0:
        q_vec = q_vec / q_norm # final value when adding axis = 1 is 1 for all rows 


    scores = vectors @ q_vec # here dot product between the vectors and q_vec
    # dot products are basically to check how inclined is query and our document
    # highest score shows that those tickets are ranked high for preference 
    ranked_idx = np.argsort(scores)[::-1][:top_k]
    return ranked_idx, scores[ranked_idx]



def eucledian_search(query: str, vectors, idf, term2idx, top_k : int):
    query_tokens = query.lower().split()
    q_vec = np.zeros(len(idf))
    q_counter = Counter(query_tokens)
    for term, tf in q_counter.items():
        if term in term2idx:
            q_vec[term2idx[term]] = tf * idf[term2idx[term]]


    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec /= q_norm 

    # eucledian distance caluculation 

    dists = np.linalg.norm(vectors - q_vec, axis = 1) # distance we substarct from vectors
    ranked_idx = np.argsort(dists)[:top_k]
    return ranked_idx, -dists[ranked_idx] # - represents rank display values from highest 


# ------- chroma DB --------------

def build_chroma(docs, collection_name = "tickets"):
    client = chromadb.Client() 
    try:
        client.delete_collection(collection_name) # delete if there is already a collection_name tickets 
    except:
        pass 

    collection = client.create_collection(name = collection_name) # create a collection 

    model = SentenceTransformer(embedding_model) # convert text -> vectors 
    embeddings = model.encode(docs, show_progress_bar = True, batch_size = 32)
    # each document converts into numerical vector 

    collection.add(
        documents = docs, # actual document without splitting
        embeddings = embeddings.tolist(), # embeddings
        ids = [f"doc{i}" for i in range(len(docs))] # create ids like doc{index}
    )

    return collection, model 

def chroma_search(collection, model, query:str, top_k : int): # find semantically simmilar documents to query
    query_emb = model.encode([query])[0] # query becomes vector 
    # [0] because else return result will eb in 2d form -> [[]]
    results = collection.query(
        query_embeddings = [query_emb.tolist()],
        n_results = top_k
    )
    # find nearest vectors 
    return results['ids'][0], results['distances'][0]

# ---- main -----

if __name__ == "__main__": # experimental runner
    tickets = load_tickets(folder_path)
    docs = get_docs(tickets)
    if not tickets:
        print(" No tickets loaded. Exciting.") 
        exit(1) # exit if no tickets

    # time differences 

    t0 = time.time() # record beginning time
    bm25_index = build_bm25_index(docs) # wait for entire process
    print(f"BM25 index built in {time.time() - t0:.2f}'s") 
    # time diffrence between the end of bm_25 process and initial seconds 
    # prints latency

    t0 = time.time()
    tfidf_vectors, tfifd_idf, ifidf_term2idx = build_tf_idf_vectors(docs)
    print(f"tfidf vectors built in {time.time() - t0:.2f}s")
        
    t0 = time.time()
    chroma_coll, embed_model = build_chroma(docs)
    print(f"chroma built in {time.time() - t0:.2f}s\n")

    test_queries = [
            "printer not working",
            "password reset failes",
            "laptop slow after every update",
            "server down",
            "need new monitor"
        ] 
        # some queries which are simmilar to it help desk just per checking latency
        

    print(f" latency : ({repeat} repeats , top- {top_k})\n")
    for query in test_queries:
        times = {"bm25":[], "cosine":[], "euclid": [], "chroma": []} # creates dictionary for storing values
        for _ in range(repeat): # for the loop end at repeat(5 as defined previosly)
                
            t0 = time.time()
            bm25_search(query, bm25_index, top_k)
            times["bm25"].append(time.time() - t0) 
            # record difference and append it to respective times[term]
                
            t0 = time.time()
            tfidf_cosine_search(query, tfidf_vectors, tfifd_idf, ifidf_term2idx, top_k)
            times["cosine"].append(time.time() - t0)
                  
                
            t0 = time.time()
            eucledian_search(query, tfidf_vectors, tfifd_idf, ifidf_term2idx, top_k)
            times["euclid"].append(time.time() - t0)

            t0 = time.time()
            chroma_search(chroma_coll, embed_model, query, top_k)
            times["chroma"].append(time.time() - t0)

        avg = {k: np.mean(v)* 1000 for k, v in times.items()} 
        # take mean of array of timimgs -> 5 times repeated 

        print(f"{query:<35} {avg['bm25']:>6.2f}ms  {avg['cosine']:>6.2f}ms  "
              f"{avg['euclid']:>6.2f}ms  {avg['chroma']:>6.2f}ms") # print statement
