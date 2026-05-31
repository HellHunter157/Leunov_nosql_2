
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_FILE = "embeddings/embeddings.npy"

os.makedirs("embeddings", exist_ok=True)

df = pd.read_parquet(INPUT_FILE)

texts = (
    df["title"].fillna("") +
    " [SEP] " +
    df["abstract"].fillna("")
).tolist()

print(f"Загальна кількість текстів: {len(texts)}")

model = SentenceTransformer("allenai/specter2_base")


embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)


print(f"\nКількість ембеддингів: {len(embeddings)}")
print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0])}")

np.save(OUTPUT_FILE, embeddings)

print(f"\nЕмбеддинги успішно збережено у файл: {OUTPUT_FILE}")