import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"

INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200

pc = Pinecone(api_key=os.environ["pcsk_grkL1_AHK2xTGx8KPf37Rwp92sYUeuuKZuU4QBPYodya6tapbiBoTaUjSskoiwcfZNvkY"])

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


index = pc.Index(INDEX_NAME)


df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

print(f"Кількість статей: {len(df)}")
print(f"Кількість ембеддингів: {len(embeddings)}")


if len(df) != len(embeddings):
    raise ValueError(
        f"Кількість статей ({len(df)}) "
        f"не збігається з кількістю ембеддингів ({len(embeddings)})"
    )


for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Завантаження"):
    end = min(start + BATCH_SIZE, len(df))

    vectors = []

    for i in range(start, end):
        row = df.iloc[i]

        vectors.append({
            "id": f"paper_{i}",
            "values": embeddings[i].tolist(),
            "metadata": {
                "arxiv_id": str(row.get("id", "")),
                "title": str(row.get("title", "")),
                "abstract": str(row.get("abstract", ""))[:500],
                "authors": str(row.get("authors", ""))[:200],
                "year": int(row.get("year", 0)),
                "category": str(row.get("categories", ""))
            }
        })

    index.upsert(vectors=vectors)

print("\nВектори успішно завантажено в Pinecone!")