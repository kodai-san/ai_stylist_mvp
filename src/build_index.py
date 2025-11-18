# src/build_index.py
from pathlib import Path
import numpy as np
import faiss

ROOT = Path(__file__).resolve().parent.parent
VEC_PATH = ROOT / "data" / "processed" / "product_vectors.npy"
META_PATH = ROOT / "data" / "processed" / "product_meta.json"
INDEX_PATH = ROOT / "data" / "processed" / "product_index.faiss"

vecs = np.load(str(VEC_PATH)).astype("float32")
# cosine via inner product on normalized vectors
row_norms = np.linalg.norm(vecs, axis=1, keepdims=True)
vecs = vecs / np.clip(row_norms, 1e-9, None)

index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
faiss.write_index(index, str(INDEX_PATH))
print("Index size:", index.ntotal, "→", INDEX_PATH)
