import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss

class FAISSIndexManager:
    """
    Builds and queries a FAISS index from precomputed vectors and metadata.
    """
    def __init__(self, dim: int):
        # Use inner product on L2-normalized vectors for cosine similarity
        self._index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vector: np.ndarray, meta: Dict[str, Any]):
        vec = vector / np.linalg.norm(vector)
        self._index.add(vec.reshape(1, -1))
        self.metadata.append(meta)

    def load_jsonl(self, jsonl_path: Path, vector_key: str = "vector"):
        with jsonl_path.open('r') as f:
            for line in f:
                rec = json.loads(line)
                vec = np.array(rec[vector_key], dtype="float32")
                self.add(vec, rec)

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int
    ) -> List[Tuple[Dict[str, Any], float]]:
        q = query_vec / np.linalg.norm(query_vec)
        D, I = self._index.search(q.reshape(1, -1), top_k)
        results: List[Tuple[Dict[str, Any], float]] = []
        for idx, score in zip(I[0], D[0]):
            if 0 <= idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))
        return results