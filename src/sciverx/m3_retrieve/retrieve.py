"""Truy xuất theo claim trong kho của cùng bài (SPEC 7-M3).

Đầu vào runtime CHỈ gồm ClaimInput (claim_id, paper_id, text) và kho M1;
không đọc GoldInfo. Mỗi phương pháp (bm25 | dense | hybrid) lấy top k_units
đơn vị, hybrid hợp nhất bằng RRF, rồi gộp lên mức cha (đã loại trùng).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sciverx.m3_retrieve import parent_key
from sciverx.m3_retrieve.rankers import (aggregate_parents, bm25_scores, dense_scores, rank_order,
                                         rrf_fuse)
from sciverx.m3_retrieve.tokenizer import TOKENIZER_VERSION, tokenize
from sciverx.schemas import ClaimInput, EvidenceUnit, RetrievalCandidate, RetrievalResult
from sciverx.utils.hashing import sha256_obj


@dataclass
class PaperIndex:
    """Kho một bài, một điều kiện bảng."""
    paper_id: str
    condition: str
    units: list[EvidenceUnit]
    tokens: list[list[str]]
    vectors: np.ndarray | None  # (n, d) đã chuẩn hóa; None nếu không dùng dense
    pool_hash: str

    @classmethod
    def build(cls, paper_id: str, condition: str, units: list[EvidenceUnit], embedder=None) -> "PaperIndex":
        vecs = embedder.encode([u.text for u in units]) if embedder is not None and units else None
        return cls(paper_id, condition, units, [tokenize(u.text) for u in units], vecs,
                   pool_hash(units))


def pool_hash(units: list[EvidenceUnit]) -> str:
    return sha256_obj([[u.unit_id, u.text] for u in units])


def _parent_ranks(scores: np.ndarray, order: list[int], parents: list[str]):
    hits = aggregate_parents({i: float(scores[i]) for i in order}, parents)
    return {h.parent: h for h in hits}


def retrieve(claim: ClaimInput, index: PaperIndex, method: str, cfg_m3: dict,
             query_vec: np.ndarray | None = None) -> RetrievalResult:
    if claim.paper_id != index.paper_id:
        raise ValueError("Kho phải là của cùng bài với claim")
    k = cfg_m3.get("k_units", 50)
    c = cfg_m3.get("rrf_constant", 60)
    bm = cfg_m3.get("bm25", {})
    parents = [parent_key(u) for u in index.units]
    per: dict[str, dict] = {}
    unit_scores: dict[int, float] = {}
    if method in ("bm25", "hybrid"):
        s = bm25_scores(claim.text, index.tokens, bm.get("k1", 1.5), bm.get("b", 0.75))
        order = rank_order(s, k)
        per["bm25"] = {"order": order, "parents": _parent_ranks(s, order, parents)}
        if method == "bm25":
            unit_scores = {i: float(s[i]) for i in order}
    if method in ("dense", "hybrid"):
        if query_vec is None or index.vectors is None:
            raise ValueError("dense cần query_vec và vectors của kho")
        s = dense_scores(query_vec, index.vectors)
        order = rank_order(s, k)
        per["dense"] = {"order": order, "parents": _parent_ranks(s, order, parents)}
        if method == "dense":
            unit_scores = {i: float(s[i]) for i in order}
    if method == "hybrid":
        unit_scores = rrf_fuse({m: v["order"] for m, v in per.items()}, c)
    if method not in ("bm25", "dense", "hybrid"):
        raise ValueError(f"method lạ: {method}")

    cands = []
    for hit in aggregate_parents(unit_scores, parents):
        b = per.get("bm25", {}).get("parents", {}).get(hit.parent)
        d = per.get("dense", {}).get("parents", {}).get(hit.parent)
        cands.append(RetrievalCandidate(
            parent_id=hit.parent, best_unit_id=index.units[hit.best_unit].unit_id,
            kind=index.units[hit.best_unit].kind,
            bm25_rank=b.rank if b else None, bm25_score=round(b.score, 6) if b else None,
            dense_rank=d.rank if d else None, dense_score=round(d.score, 6) if d else None,
            rrf_score=round(hit.score, 6) if method == "hybrid" else None, final_rank=hit.rank))
    return RetrievalResult(
        claim_id=claim.claim_id, method=method, table_condition=index.condition,
        params={"tokenizer_version": TOKENIZER_VERSION, "embed_model": cfg_m3.get("embed_model"),
                "rrf_constant": c, "k_units": k, "bm25_k1": bm.get("k1", 1.5), "bm25_b": bm.get("b", 0.75)},
        n_units_in_pool=len(index.units), n_parents_in_pool=len(set(parents)),
        candidates=cands, pool_hash=index.pool_hash)


def units_of_parent(index: PaperIndex, parent: str) -> list[tuple[int, EvidenceUnit]]:
    return [(i, u) for i, u in enumerate(index.units) if parent_key(u) == parent]
