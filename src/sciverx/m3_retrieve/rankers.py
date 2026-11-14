"""BM25, dense và RRF trên đơn vị; gộp kết quả lên mức cha (SPEC 7-M3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from sciverx.m3_retrieve.tokenizer import tokenize


# ------------------------------------------------------------------ BM25

def bm25_scores(query: str, doc_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    from rank_bm25 import BM25Okapi
    if not doc_tokens:
        return np.zeros(0)
    # BM25Okapi chia cho độ dài trung bình; văn bản rỗng được thay bằng một token giả
    safe = [t if t else ["\u0000"] for t in doc_tokens]
    return np.asarray(BM25Okapi(safe, k1=k1, b=b).get_scores(tokenize(query)), dtype=float)


# ------------------------------------------------------------------ dense

class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Ma trận (n, d) đã chuẩn hóa L2."""


class SentenceTransformerEmbedder:  # pragma: no cover - cần sentence-transformers
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True,
                                            convert_to_numpy=True), dtype=np.float32)


def dense_scores(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Cosine trên vector đã chuẩn hóa = tích vô hướng."""
    if doc_matrix.size == 0:
        return np.zeros(0)
    return doc_matrix @ query_vec


# ------------------------------------------------------------------ xếp hạng

def rank_order(scores: np.ndarray, k: int | None = None) -> list[int]:
    """Chỉ số đơn vị theo điểm giảm dần; hòa điểm giữ thứ tự gốc (ổn định, tất định)."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[:k] if k is not None else order


def rrf_fuse(rankings: dict[str, list[int]], constant: int = 60) -> dict[int, float]:
    """score(u) = Σ_m 1 / (constant + rank_m(u)), rank bắt đầu từ 1.
    `constant` là hằng số RRF, không phải top-k."""
    out: dict[int, float] = {}
    for order in rankings.values():
        for rank, idx in enumerate(order, 1):
            out[idx] = out.get(idx, 0.0) + 1.0 / (constant + rank)
    return out


@dataclass
class ParentHit:
    parent: str
    best_unit: int
    score: float
    rank: int


def aggregate_parents(unit_scores: dict[int, float], parents: list[str]) -> list[ParentHit]:
    """Điểm cha = điểm cao nhất của các đơn vị con; mỗi cha chiếm MỘT vị trí."""
    best: dict[str, tuple[float, int]] = {}
    for idx, s in unit_scores.items():
        p = parents[idx]
        if p not in best or (s, -idx) > (best[p][0], -best[p][1]):
            best[p] = (s, idx)
    ordered = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[1][1]))
    return [ParentHit(p, idx, s, r) for r, (p, (s, idx)) in enumerate(ordered, 1)]
