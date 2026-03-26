"""
Hybrid Retriever
────────────────
Combines:
  1. TinyBM25   — keyword retrieval (zero extra deps, built-in)
  2. FAISSIndex — dense semantic retrieval
  3. RRF        — Reciprocal Rank Fusion to merge both result lists
  4. MMR        — Maximal Marginal Relevance for diversity

Usage:
    retriever = HybridRetriever(chunks)
    top_chunks = retriever.retrieve(query, top_k=6)
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from .chunker import Chunk
from .embedder import FAISSIndex, embed_texts


# ──────────────────────────────────────────────
# TinyBM25
# ──────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    return re.findall(r"\b[a-z0-9_]+\b", text.lower())


class TinyBM25:
    """
    Pure-Python BM25 implementation (Okapi BM25).
    k1=1.5, b=0.75
    """
    k1 = 1.5
    b  = 0.75

    def __init__(self, corpus: List[str]):
        self.n = len(corpus)
        self.tokenised = [_tokenise(doc) for doc in corpus]
        self.dl = [len(toks) for toks in self.tokenised]
        self.avgdl = sum(self.dl) / max(self.n, 1)

        # Build inverted index: term → {doc_idx: freq}
        self.idf: dict[str, float] = {}
        self.tf: list[dict[str, int]] = []

        df_count: dict[str, int] = defaultdict(int)
        for toks in self.tokenised:
            freq: dict[str, int] = defaultdict(int)
            for t in toks:
                freq[t] += 1
            self.tf.append(dict(freq))
            for t in set(toks):
                df_count[t] += 1

        for term, df in df_count.items():
            self.idf[term] = math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        score = 0.0
        dl = self.dl[doc_idx]
        tf_map = self.tf[doc_idx]
        for t in query_tokens:
            if t not in tf_map:
                continue
            idf = self.idf.get(t, 0.0)
            tf = tf_map[t]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        tokens = _tokenise(query)
        if not tokens:
            return []
        scores = [
            (i, self.score(tokens, i))
            for i in range(self.n)
            if self.score(tokens, i) > 0
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ──────────────────────────────────────────────
# RRF
# ──────────────────────────────────────────────

def _rrf_fuse(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion.
    Returns (chunk_idx, rrf_score) sorted descending.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked):
            scores[idx] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ──────────────────────────────────────────────
# MMR
# ──────────────────────────────────────────────

def _mmr(
    query_vec: np.ndarray,
    candidate_indices: List[int],
    chunk_vecs: np.ndarray,
    top_k: int = 6,
    lambda_: float = 0.6,
) -> List[int]:
    """
    Maximal Marginal Relevance.
    lambda_: trade-off between relevance (1.0) and diversity (0.0).
    """
    if not candidate_indices:
        return []

    selected: List[int] = []
    remaining = list(candidate_indices)

    # query similarity scores
    q = query_vec[0]  # (D,)
    cand_vecs = chunk_vecs[remaining]  # (C, D)
    query_sims = cand_vecs @ q  # (C,)

    while remaining and len(selected) < top_k:
        if not selected:
            # Pick the most relevant first
            best_local = int(np.argmax(query_sims))
        else:
            sel_vecs = chunk_vecs[selected]  # (S, D)
            cand_vecs2 = chunk_vecs[remaining]  # (R, D)
            # max sim to already selected
            redundancy = (cand_vecs2 @ sel_vecs.T).max(axis=1)  # (R,)
            # recompute query sims for remaining
            q_sims_remaining = cand_vecs2 @ q
            mmr_scores = lambda_ * q_sims_remaining - (1 - lambda_) * redundancy
            best_local = int(np.argmax(mmr_scores))

        chosen = remaining.pop(best_local)
        selected.append(chosen)
        # Keep query_sims aligned — drop the chosen entry
        query_sims = np.delete(query_sims, best_local)

    return selected


# ──────────────────────────────────────────────
# HybridRetriever
# ──────────────────────────────────────────────

class HybridRetriever:
    """
    Full hybrid retrieval stack:
    BM25 + FAISS → RRF fusion → MMR diversification
    """

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        corpus = [c.text for c in chunks]
        self.bm25 = TinyBM25(corpus)
        self.faiss_index = FAISSIndex(chunks)
        # Keep the dense vecs for MMR
        self.chunk_vecs = self.faiss_index.vecs  # (N, D)

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        bm25_k: int = 20,
        dense_k: int = 20,
        mmr_lambda: float = 0.65,
    ) -> List[Chunk]:
        """
        Retrieve top_k diverse and relevant chunks for a query.
        """
        # ── BM25 results ───────────────────────
        bm25_results = self.bm25.search(query, top_k=bm25_k)

        # ── Dense results ──────────────────────
        dense_results = self.faiss_index.search(query, top_k=dense_k)

        # ── RRF fusion ─────────────────────────
        fused = _rrf_fuse([bm25_results, dense_results])
        candidate_indices = [idx for idx, _ in fused[: top_k * 3]]

        if not candidate_indices:
            return []

        # ── MMR diversification ────────────────
        q_vec = embed_texts([query])  # (1, D)
        diverse_indices = _mmr(
            q_vec,
            candidate_indices,
            self.chunk_vecs,
            top_k=top_k,
            lambda_=mmr_lambda,
        )

        return [self.chunks[i] for i in diverse_indices]