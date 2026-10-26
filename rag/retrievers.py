"""Three retrievers over the same passage list: dense (MiniLM + FAISS), BM25, and hybrid (RRF)."""
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class DenseRetriever:
    name = "dense"

    def __init__(self, passages):
        self.model = SentenceTransformer(EMBED_MODEL)
        cache = CACHE_DIR / f"embeddings_{len(passages)}.npy"
        if cache.exists():
            emb = np.load(cache)
        else:
            emb = self.model.encode([p["text"] for p in passages], batch_size=64,
                                    normalize_embeddings=True, show_progress_bar=True)
            CACHE_DIR.mkdir(exist_ok=True)
            np.save(cache, emb)
        # Normalized vectors + inner product = cosine similarity.
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb.astype(np.float32))

    def search_many(self, queries, k):
        q = self.model.encode(queries, batch_size=64, normalize_embeddings=True)
        scores, ids = self.index.search(q.astype(np.float32), k)
        return [list(zip(row_ids.tolist(), row_scores.tolist()))
                for row_ids, row_scores in zip(ids, scores)]

    def search(self, query, k):
        return self.search_many([query], k)[0]


class BM25Retriever:
    name = "bm25"

    def __init__(self, passages):
        self.bm25 = BM25Okapi([tokenize(p["text"]) for p in passages])

    def search(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]

    def search_many(self, queries, k):
        return [self.search(q, k) for q in queries]


class HybridRetriever:
    """Reciprocal Rank Fusion: score(p) = sum over retrievers of 1 / (rrf_k + rank)."""
    name = "hybrid"

    def __init__(self, dense, bm25, depth=50, rrf_k=60):
        self.dense, self.bm25, self.depth, self.rrf_k = dense, bm25, depth, rrf_k

    def _fuse(self, *rankings, k):
        scores = {}
        for ranking in rankings:
            for rank, (pid, _) in enumerate(ranking, start=1):
                scores[pid] = scores.get(pid, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(scores.items(), key=lambda x: -x[1])[:k]

    def search_many(self, queries, k):
        dense = self.dense.search_many(queries, self.depth)
        bm25 = self.bm25.search_many(queries, self.depth)
        return [self._fuse(d, b, k=k) for d, b in zip(dense, bm25)]

    def search(self, query, k):
        return self.search_many([query], k)[0]


def build_all(passages):
    dense = DenseRetriever(passages)
    bm25 = BM25Retriever(passages)
    return {"dense": dense, "bm25": bm25, "hybrid": HybridRetriever(dense, bm25)}
