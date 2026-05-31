# scripts/05_chunking.py
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")


df.sort_values(by='abstract_len', ascending=False)
df["abstract_len"] = df["abstract"].astype(str).apply(lambda x: len(x.split()))
head(30)


df.sort_values(by='abstract_len', ascending=True)
df["abstract_len"] = df["abstract"].astype(str).apply(lambda x: len(x.split()))
head(30)


def chunk_text(text, chunk_size, overlap):
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)]

chunks_list = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    abstract = str(row['abstract'])
    for i, chunk in enumerate(chunk_text(abstract, 100, 20)):
        embedding = model.encode(chunk).tolist()
        chunks_list.append({
            "id": f"{row['arxiv_id']}_chunk_{i}",
            "values": embedding,
            "metadata": {
                "arxiv_id": str(row["arxiv_id"]),
                "title": str(row["title"]),
                "chunk_text": chunk,
                "chunk_num": i,
                "year": int(row["year"]),
                "category": str(row["category"])
            }
        })

for i in tqdm(range(0, len(chunks_list), 100)):
    index.upsert(vectors=chunks_list[i:i + 100])

def search(query, top_k=TOP_K):
    query_embedding = model.encode(query).tolist()
    result = index.query(query_embedding, top_k=top_k, include_metadata=True)
    
    papers = []
    for match in result["matches"]:
        metadata = match["metadata"]
        paper_id = metadata["arxiv_id"]
        title = metadata["title"]
        chunk_text = metadata["chunk_text"]
        score = match["score"]
        papers.append({
            "id": paper_id,
            "title": title,
            "chunk_text": chunk_text,
            "score": score
        })
    
    return papers

