"""
Embedder — FAISS Index Builder
───────────────────────────────
Uses sentence-transformers/all-MiniLM-L6-v2 (runs fully locally, ~80 MB).
Builds a cosine-similarity FAISS index over chunks.
"""

from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

from .chunker import Chunk

# Single global model (loaded once per process)
_MODEL: SentenceTransformer | None = None
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _MODEL


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Return L2-normalised embeddings of shape (N, D)."""
    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine via inner product
    )
    return vecs.astype(np.float32)


# ──────────────────────────────────────────────
# FAISS Index wrapper
# ──────────────────────────────────────────────

class FAISSIndex:
    """Thin FAISS wrapper for cosine-similarity search."""

    def __init__(self, chunks: List[Chunk]):
        texts = [c.text for c in chunks]
        self.chunks = chunks
        self.vecs = embed_texts(texts)
        dim = self.vecs.shape[1]

        # IndexFlatIP works as cosine similarity when vecs are L2-normalised
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.vecs)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Returns list of (chunk_index, score) sorted descending.
        """
        q_vec = embed_texts([query])  # shape (1, D)
        scores, indices = self.index.search(q_vec, min(top_k, len(self.chunks)))
        results = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]
        return results