import json
import os
import math
from collections import Counter, defaultdict
import numpy as np
from typing import List, Dict, Any

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

TICKETS_FOLDER = r"C:\Users\peepl\Downloads\ticket_raise\tickets_extracted"           # ← folder where your .json files are stored

TEXT_FIELDS = ["title", "description"]        # fields to use for search

# BM25 hyperparameters
BM25_K1 = 1.5
BM25_B  = 0.75

TOP_K = 5                                     # how many best matches to return

# ────────────────────────────────────────────────
# 1. Load all tickets from folder
# ────────────────────────────────────────────────

def load_tickets(folder_path: str) -> List[Dict[str, Any]]:
    tickets = []
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return tickets

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                    # Optional: add filename as ticket_id if missing
                    if "ticket_id" not in data:
                        data["ticket_id"] = filename.replace(".json", "")
                    tickets.append(data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    print(f"Loaded {len(tickets)} tickets from folder '{folder_path}'")
    return tickets


# ────────────────────────────────────────────────
# 2. Simple tokenizer
# ────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return text.lower().split()


# ────────────────────────────────────────────────
# 3. BM25 Implementation
# ────────────────────────────────────────────────

def build_bm25_index(tickets: List[Dict]):
    tokenized_docs = []
    for ticket in tickets:
        text = " ".join(str(ticket.get(field, "")) for field in TEXT_FIELDS)
        tokens = tokenize(text)
        tokenized_docs.append(tokens)

    N = len(tokenized_docs)
    doc_lengths = [len(t) for t in tokenized_docs]
    avgdl = sum(doc_lengths) / N if N > 0 else 1.0

    df = defaultdict(int)
    for tokens in tokenized_docs:
        for term in set(tokens):
            df[term] += 1

    idf = {}
    for term, freq in df.items():
        idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    return {
        "tokenized_docs": tokenized_docs,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "idf": idf,
        "N": N
    }


def bm25_score(query_tokens: List[str], doc_idx: int, index: Dict) -> float:
    score = 0.0
    doc = index["tokenized_docs"][doc_idx]
    dl = index["doc_lengths"][doc_idx]
    avgdl = index["avgdl"]
    counter = Counter(doc)

    for term in query_tokens:
        if term not in index["idf"]:
            continue
        tf = counter.get(term, 0)
        if tf == 0:
            continue
        sat_tf = tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl))
        score += sat_tf * index["idf"][term]

    return score


# ────────────────────────────────────────────────
# 4. TF-IDF + Cosine / Euclidean
# ────────────────────────────────────────────────

def build_tfidf_vectors(tickets: List[Dict]):
    all_text = [" ".join(str(t.get(f, "")) for f in TEXT_FIELDS) for t in tickets]
    all_tokens = [tokenize(txt) for txt in all_text]
    vocab = sorted(set(w for tokens in all_tokens for w in tokens))
    V = len(vocab)
    N = len(tickets)

    term2idx = {term: i for i, term in enumerate(vocab)}

    doc_freq = np.zeros(V)
    for tokens in all_tokens:
        for t in set(tokens):
            if t in term2idx:
                doc_freq[term2idx[t]] += 1

    idf = np.log(N / (doc_freq + 1)) + 1

    vectors = np.zeros((N, V))
    for i, tokens in enumerate(all_tokens):
        counter = Counter(tokens)
        for term, tf in counter.items():
            if term in term2idx:
                vectors[i, term2idx[term]] = tf * idf[term2idx[term]]

    # L2 normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors /= norms

    return vectors, idf, term2idx


def cosine_similarity(q_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return vectors @ q_vec


def euclidean_distance(q_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vectors - q_vec, axis=1)


# ────────────────────────────────────────────────
# 5. Search function – supports all 3 methods
# ────────────────────────────────────────────────

def search_tickets(query: str, tickets:List[Dict], method: str , top_k: int = TOP_K):
    query_tokens = tokenize(query)
    if not query_tokens:
        
        return {"error": "Empty query"}

    results = []

    method = method.lower()

    if method == "bm25":
        index = build_bm25_index(tickets)
        scores = [bm25_score(query_tokens, i, index) for i in range(len(tickets))]
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        score_values = scores

    elif method in ["cosine", "euclidean"]:
        vectors, idf, term2idx = build_tfidf_vectors(tickets)

        q_vec = np.zeros(len(idf))
        q_counter = Counter(query_tokens)
        for term, tf in q_counter.items():
            if term in term2idx:
                q_vec[term2idx[term]] = tf * idf[term2idx[term]]

        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        if method == "cosine":
            scores = cosine_similarity(q_vec, vectors)
            ranked_idx = np.argsort(scores)[::-1][:top_k]
            score_values = scores
        else:  # euclidean
            dists = euclidean_distance(q_vec, vectors)
            ranked_idx = np.argsort(dists)[:top_k]
            score_values = -dists  # invert (higher = better)

    else:
        return {"error": "Invalid method. Use: bm25 / cosine / euclidean"}

    for pos, idx in enumerate(ranked_idx):
        ticket = tickets[idx]
        results.append({
            "rank": pos + 1,
            "score": round(float(score_values[idx]), 4),
            "ticket_id": ticket.get("ticket_id", ""),
            "title": ticket.get("title", ""),
            "description": ticket.get("description", ""),
            "status": ticket.get("status", ""),
            "priority": ticket.get("priority", ""),
            "category": ticket.get("category", ""),
            "department": ticket.get("department", ""),
            "assigned_to": ticket.get("assigned_to", ""),
            "created_date": ticket.get("created_date", ""),
            "updated_date": ticket.get("updated_date", ""),
            "reporter": ticket.get("reporter", ""),
            "tags": ticket.get("tags", [])
        })

    return {
        "query": query,
        "method": method,
        "top_results": results,
        "total_tickets": len(tickets)
    }


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────

if __name__ == "__main__":
    tickets = load_tickets(TICKETS_FOLDER)

    if not tickets:
        print("No tickets found. Exiting.")
        exit(1)

    print("\nHelpdesk Ticket Search")
    print("Available methods: bm25 / cosine / euclidean")
    print("Type 'exit' to quit\n")

    while True:
        query = input("search query :").strip()
        if query.lower() in ["exit", "quit"]:
            print("\n goodbye!")
            break 
        if not query:
            print("\n plz enter a search term \n ")
            continue 

        method_input = input("method (1/2/3 or name)").strip().lower()

        # map inputs to method 
        if method_input in ["1", "bm25"]:
            method = "bm25"
        elif method_input in ["2", "cosine", "cos"]:
            method = "cosine"
        elif method_input in ["3", "euclidean", "euc", "euclid"]:
            method = "euclidean"
        else:
            print("\n invalid method")
            method = "bm25"

        result = search_tickets(query, tickets, method = method, top_k = TOP_K)

        if "error" in result:
            print(f"\n {result["error"]}")
        else:
            print(json.dumps(result, indent = 2))

        

--------------------------
--------------------------

output -> 


output -> 

Loaded 100 tickets from folder 'C:\Users\peepl\Downloads\ticket_raise\tickets_extracted'

Helpdesk Ticket Search
Available methods: bm25 / cosine / euclidean
Type 'exit' to quit

{
  "query": "email",
  "method": "bm25",
  "top_results": [
    {
      "rank": 1,
      "score": 2.0584,
      "ticket_id": "TKT-00018",
      "title": "Email synchronization problem - messages not appearing",
      "description": "Mobile device needs email configuration assistance. User cannot set up corporate email account on new device. Configuration settings need to be provided. This is preventing access to email on mobile device.",
      "status": "In Progress",
      "priority": "Low",
      "category": "Printer",
      "department": "Sales",
      "assigned_to": "Sarah Johnson",
      "created_date": "2024-12-30",
      "updated_date": "2024-06-22",
      "reporter": "user9@company.com",
      "tags": [
        "hardware",
        "network"
      ]
    },
    {
      "rank": 2,
      "score": 1.9198,
      "ticket_id": "TKT-00002",
      "title": "Network connectivity issues - intermittent disconnections",
      "description": "Mobile device needs email configuration assistance. User cannot set up corporate email account on new device. Configuration settings need to be provided. This is preventing access to email on mobile device.",
      "status": "In Progress",
      "priority": "High",
      "category": "Email",
      "department": "IT Support",
      "assigned_to": "Sarah Johnson",
      "created_date": "2024-07-23",
      "updated_date": "2024-01-24",
      "reporter": "user22@company.com",
      "tags": [
        "access",
        "software"
      ]
    },
    {
      "rank": 3,
      "score": 1.8898,
      "ticket_id": "TKT-00011",
      "title": "Password reset required - account locked after failed attempts",
      "description": "Mobile device needs email configuration assistance. User cannot set up corporate email account on new device. Configuration settings need to be provided. This is preventing access to email on mobile device.",
      "status": "In Progress",
      "priority": "Critical",
      "category": "Email",
      "department": "Sales",
      "assigned_to": "Lisa Anderson",
      "created_date": "2024-11-24",
      "updated_date": "2024-07-10",
      "reporter": "user5@company.com",
      "tags": [
        "update",
        "bug",
        "printer",
        "hardware"
      ]
    },
    {
      "rank": 4,
      "score": 1.5412,
      "ticket_id": "TKT-00048",
      "title": "Issue with email - Permission denied for database access",
      "description": "User reported an issue where they are experiencing problems with installing their email account. The problem started earlier today and is causing data loss concerns. User has verified credentials are correct but the issue persists. This is blocking critical work tasks.",
      "status": "Closed",
      "priority": "Medium",
      "category": "Application",
      "department": "Finance",
      "assigned_to": "John Smith",
      "created_date": "2024-11-22",
      "updated_date": "2024-08-06",
      "reporter": "user44@company.com",
      "tags": [
        "email",
        "network"
      ]
    },
    {
      "rank": 5,
      "score": 1.5218,
      "ticket_id": "TKT-00051",
      "title": "Issue with email - Error when opening VPN connection",
      "description": "User reported an issue where they are experiencing problems with using their email account. The problem started yesterday afternoon and is impacting critical business operations. User has verified credentials are correct but the issue persists. Urgent resolution needed as this affects daily operations.",
      "status": "Open",
      "priority": "Low",
      "category": "Printer",
      "department": "Operations",
      "assigned_to": "Mike Davis",
      "created_date": "2024-07-06",
      "updated_date": "2024-06-18",
      "reporter": "user37@company.com",
      "tags": [
        "configuration",
        "printer"
      ]
    }
  ],
  "total_tickets": 100
}
{
  "query": "server",
  "method": "cosine",
  "top_results": [
    {
      "rank": 1,
      "score": 0.0,
      "ticket_id": "TKT-00100",
      "title": "Issue with application - Connection problem with network drive",
      "description": "User reported an issue where they are experiencing problems with installing their email account. The problem started yesterday afternoon and is causing data loss concerns. User has attempted to reconnect multiple times but the issue persists. Please investigate and provide solution as soon as possible.",
      "status": "Resolved",
      "priority": "High",
      "category": "Software",
      "department": "Finance",
      "assigned_to": "John Smith",
      "created_date": "2024-04-15",
      "updated_date": "2024-03-25",
      "reporter": "user13@company.com",
      "tags": [
        "configuration",
        "update",
        "urgent",
        "access"
      ]
    },
    {
      "rank": 2,
      "score": 0.0,
      "ticket_id": "TKT-00099",
      "title": "Issue with account - Configuration issue for database access",
      "description": "User reported an issue where they are experiencing problems with using their email account. The problem started this morning and is impacting critical business operations. User has cleared cache and cookies but the issue persists. Urgent resolution needed as this affects daily operations.",
      "status": "Pending",
      "priority": "Medium",
      "category": "Account",
      "department": "Operations",
      "assigned_to": "David Wilson",
      "created_date": "2024-06-15",
      "updated_date": "2024-04-18",
      "reporter": "user6@company.com",
      "tags": [
        "configuration",
        "update",
        "network"
      ]
    },
    {
      "rank": 3,
      "score": 0.0,
      "ticket_id": "TKT-00098",
      "title": "Issue with printer - Update required for remote desktop",
      "description": "User reported an issue where they are experiencing problems with updating cloud services. The problem started a few days ago and is impacting critical business operations. User has tried restarting the application but the issue persists. Please investigate and provide solution as soon as possible.",
      "status": "In Progress",
      "priority": "Critical",
      "category": "Access",
      "department": "Sales",
      "assigned_to": "John Smith",
      "created_date": "2024-09-25",
      "updated_date": "2024-04-23",
      "reporter": "user28@company.com",
      "tags": [
        "bug",
        "email",
        "vpn"
      ]
    },
    {
      "rank": 4,
      "score": 0.0,
      "ticket_id": "TKT-00097",
      "title": "Issue with vpn - Timeout error in shared folder",
      "description": "User reported an issue where they are experiencing problems with configuring VPN services. The problem started earlier today and is causing significant delays. User has tried restarting the application but the issue persists. Urgent resolution needed as this affects daily operations.",
      "status": "Resolved",
      "priority": "High",
      "category": "Hardware",
      "department": "Marketing",
      "assigned_to": "Emily Chen",
      "created_date": "2024-04-03",
      "updated_date": "2024-04-09",
      "reporter": "user10@company.com",
      "tags": [
        "update",
        "security",
        "printer"
      ]
    },
    {
      "rank": 5,
      "score": 0.0,
      "ticket_id": "TKT-00096",
      "title": "Issue with email - Performance degradation in network drive",
      "description": "User reported an issue where they are experiencing problems with accessing mobile devices. The problem started yesterday afternoon and is preventing them from completing their work. User has tried from a different device but the issue persists. Escalation may be required if not resolved quickly.",
      "status": "Resolved",
      "priority": "Critical",
      "category": "Software",
      "department": "HR",
      "assigned_to": "David Wilson",
      "created_date": "2024-01-14",
      "updated_date": "2024-06-04",
      "reporter": "user9@company.com",
      "tags": [
        "printer",
        "software",
        "hardware",
        "urgent"
      ]
    }
  ],
  "total_tickets": 100
}
{
  "query": "software",
  "method": "euclidean",
  "top_results": [
    {
      "rank": 1,
      "score": -1.2048,
      "ticket_id": "TKT-00065",
      "title": "Issue with software - Performance degradation in software application",
      "description": "User reported an issue where they are experiencing problems with installing the printer. The problem started last week and is causing significant delays. User has tried from a different device but the issue persists. Multiple users have reported similar issues.",
      "status": "In Progress",
      "priority": "Low",
      "category": "Hardware",
      "department": "Operations",
      "assigned_to": "Sarah Johnson",
      "created_date": "2024-05-21",
      "updated_date": "2024-03-12",
      "reporter": "user22@company.com",
      "tags": [
        "update",
        "security",
        "bug"
      ]
    },
    {
      "rank": 2,
      "score": -1.2635,
      "ticket_id": "TKT-00007",
      "title": "Software license expiration - activation required",
      "description": "Software update installation fails with error code 0x80070005. The update process starts but terminates midway. This is preventing the installation of critical security patches. User has tried running as administrator but issue persists. Urgent resolution needed.",
      "status": "In Progress",
      "priority": "Low",
      "category": "Printer",
      "department": "Marketing",
      "assigned_to": "Lisa Anderson",
      "created_date": "2024-01-05",
      "updated_date": "2024-08-01",
      "reporter": "user39@company.com",
      "tags": [
        "hardware",
        "network"
      ]
    },
    {
      "rank": 3,
      "score": -1.3023,
      "ticket_id": "TKT-00047",
      "title": "Issue with software - Installation failure of database access",
      "description": "User reported an issue where they are experiencing problems with using cloud services. The problem started this morning and is causing significant delays. User has tried restarting the application but the issue persists. This is blocking critical work tasks.",
      "status": "Pending",
      "priority": "Critical",
      "category": "Account",
      "department": "IT Support",
      "assigned_to": "David Wilson",
      "created_date": "2024-01-22",
      "updated_date": "2024-05-27",
      "reporter": "user36@company.com",
      "tags": [
        "network",
        "hardware"
      ]
    },
    {
      "rank": 4,
      "score": -1.3039,
      "ticket_id": "TKT-00044",
      "title": "Issue with access - Installation failure of email account",
      "description": "User reported an issue where they are experiencing problems with updating a software application. The problem started yesterday afternoon and is causing significant delays. User has verified credentials are correct but the issue persists. This is blocking critical work tasks.",
      "status": "Open",
      "priority": "Critical",
      "category": "Network",
      "department": "Marketing",
      "assigned_to": "Mike Davis",
      "created_date": "2024-09-06",
      "updated_date": "2024-03-23",
      "reporter": "user15@company.com",
      "tags": [
        "update",
        "network"
      ]
    },
    {
      "rank": 5,
      "score": -1.3063,
      "ticket_id": "TKT-00046",
      "title": "Issue with software - Update required for cloud service",
      "description": "User reported an issue where they are experiencing problems with accessing their email account. The problem started earlier today and is causing data loss concerns. User has tried restarting the application but the issue persists. This is blocking critical work tasks.",
      "status": "Resolved",
      "priority": "Critical",
      "category": "Printer",
      "department": "IT Support",
      "assigned_to": "Mike Davis",
      "created_date": "2024-07-31",
      "updated_date": "2024-11-27",
      "reporter": "user32@company.com",
      "tags": [
        "security",
        "bug",
        "email",
        "printer"
      ]
    }
  ],
  "total_tickets": 100
}

 goodbye!

        

