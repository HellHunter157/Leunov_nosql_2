# scripts/04_search.py
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")  # для отримання повного abstract

def search(query, top_k=5):

    query_embedding = model.encode(query).tolist()

  
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    papers = []

    for match in results["matches"]:
        metadata = match["metadata"]

        papers.append({
            "id": match["id"],
            "arxiv_id": metadata.get("arxiv_id"),
            "title": metadata.get("title"),
            "abstract": metadata.get("abstract"),
            "authors": metadata.get("authors"),
            "year": metadata.get("year"),
            "category": metadata.get("category"),
            "score": match["score"]
        })

    return papers














