"""Truy xuất theo claim trong kho của cùng bài (SPEC 7-M3).

Đầu vào runtime CHỈ gồm ClaimInput (claim_id, paper_id, text) và kho M1;
không đọc GoldInfo. Mỗi phương pháp (bm25 | dense | hybrid) lấy top k_units
đơn vị, hybrid hợp nhất bằng RRF, rồi gộp lên mức cha (đã loại trùng).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sciverx.m3_retrieve import parent_key
from sciverx.m3_retrieve.rankers import (ParentHit, aggregate_parents, bm25_scores, dense_scores,
                                         rank_order, rrf_fuse)
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


def _rerank_parents(claim_text: str, index: PaperIndex, unit_scores: dict[int, float], parents: list[str],
                    reranker, top_n: int) -> tuple[list[ParentHit], dict[str, tuple[int, float]]]:
    """Chấm lại top_n đơn vị của bước tìm đầu bằng reranker rồi gộp lên cha.
    Cha có đơn vị trong top_n xếp theo điểm rerank; cha còn lại giữ thứ tự bước đầu, xếp sau.
    Trả (danh sách cha, parent → (rerank_rank, rerank_score))."""
    first = sorted(unit_scores, key=lambda i: (-unit_scores[i], i))
    head = first[:top_n]
    rs = reranker.score(claim_text, [index.units[i].text for i in head]) if head else np.zeros(0)
    reranked = aggregate_parents({i: float(s) for i, s in zip(head, rs)}, parents)
    info = {h.parent: (h.rank, h.score) for h in reranked}
    rest = [h for h in aggregate_parents({i: unit_scores[i] for i in first[top_n:]}, parents)
            if h.parent not in info]
    merged = reranked + rest
    return [ParentHit(h.parent, h.best_unit, h.score, r) for r, h in enumerate(merged, 1)], info


def reserve_visual_slots(hits: list[ParentHit], kinds: list[str], k: int, n: int) -> list[ParentHit]:
    """Bảo đảm top-k có ít nhất n cha là bảng/biểu đồ (nếu kho có): kéo cha thị giác
    tốt nhất ở ngoài top-k lên, đẩy cha văn bản xếp cuối trong top-k xuống ngay sau.
    Thứ tự tương đối trong mỗi nhóm được giữ nguyên. n = 0 → không đổi gì."""
    n = min(n, k)
    if n <= 0:
        return hits

    def is_visual(h: ParentHit) -> bool:
        return kinds[h.best_unit] != "paragraph"

    top, tail = list(hits[:k]), list(hits[k:])
    need = n - sum(map(is_visual, top))
    pulled = [h for h in tail if is_visual(h)][:max(0, need)]
    if not pulled:
        return hits
    pushed = []
    for _ in pulled:
        j = max(i for i, h in enumerate(top) if not is_visual(h))
        pushed.insert(0, top.pop(j))
    top += pulled
    top.sort(key=lambda h: h.rank)              # giữ thứ tự gốc giữa các cha còn trong top-k
    rest = pushed + [h for h in tail if h not in pulled]
    return [ParentHit(h.parent, h.best_unit, h.score, r) for r, h in enumerate(top + rest, 1)]


def retrieve(claim: ClaimInput, index: PaperIndex, method: str, cfg_m3: dict,
             query_vec: np.ndarray | None = None, reranker=None) -> RetrievalResult:
    """cfg_m3 (ngoài k_units, rrf_constant, bm25):
    - rrf_weights: {bm25: 1.0, dense: 0.5} — RRF có trọng số cho hybrid (mặc định 1.0 / 1.0);
    - rerank: {model: <cross-encoder>, top_n: 30} — chấm lại top_n đơn vị (mặc định tắt);
    - visual_slots: {n: 2, k?} — giữ ít nhất n bảng/biểu đồ trong top-k (mặc định tắt);
      k bỏ trống = budget.max_parents, tức đúng số cha model sẽ nhận."""
    if claim.paper_id != index.paper_id:
        raise ValueError("Kho phải là của cùng bài với claim")
    k = cfg_m3.get("k_units", 50)
    c = cfg_m3.get("rrf_constant", 60)
    bm = cfg_m3.get("bm25", {})
    weights = cfg_m3.get("rrf_weights") or {}
    rr_cfg = cfg_m3.get("rerank") or {}
    vs = cfg_m3.get("visual_slots") or {}
    if reranker is None and rr_cfg.get("model"):  # pragma: no cover - nạp model thật
        from sciverx.m3_retrieve.rankers import get_reranker
        reranker = get_reranker(rr_cfg["model"])
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
        unit_scores = rrf_fuse({m: v["order"] for m, v in per.items()}, c, weights)
    if method not in ("bm25", "dense", "hybrid"):
        raise ValueError(f"method lạ: {method}")

    fused = {h.parent: h for h in aggregate_parents(unit_scores, parents)}
    rr_info: dict[str, tuple[int, float]] = {}
    if reranker is not None:
        hits, rr_info = _rerank_parents(claim.text, index, unit_scores, parents, reranker,
                                        int(rr_cfg.get("top_n", 30)))
    else:
        hits = list(fused.values())
    # k mặc định = max_parents: giữ chỗ đúng trong vùng mà bước chọn bằng chứng sẽ đưa cho model
    vs_k = int(vs.get("k") or cfg_m3.get("budget", {}).get("max_parents", 0))
    hits = reserve_visual_slots(hits, [u.kind for u in index.units], vs_k, int(vs.get("n", 0)))

    cands = []
    for hit in hits:
        b = per.get("bm25", {}).get("parents", {}).get(hit.parent)
        d = per.get("dense", {}).get("parents", {}).get(hit.parent)
        f = fused.get(hit.parent)
        rr = rr_info.get(hit.parent)
        cands.append(RetrievalCandidate(
            parent_id=hit.parent, best_unit_id=index.units[hit.best_unit].unit_id,
            kind=index.units[hit.best_unit].kind,
            bm25_rank=b.rank if b else None, bm25_score=round(b.score, 6) if b else None,
            dense_rank=d.rank if d else None, dense_score=round(d.score, 6) if d else None,
            rrf_score=round(f.score, 6) if method == "hybrid" and f else None,
            rerank_rank=rr[0] if rr else None, rerank_score=round(rr[1], 6) if rr else None,
            final_rank=hit.rank))
    params = {"tokenizer_version": TOKENIZER_VERSION, "embed_model": cfg_m3.get("embed_model"),
              "rrf_constant": c, "k_units": k, "bm25_k1": bm.get("k1", 1.5), "bm25_b": bm.get("b", 0.75)}
    if weights:
        params["rrf_weights"] = dict(weights)
    if reranker is not None:
        params["rerank_model"] = getattr(reranker, "name", None)
        params["rerank_top_n"] = int(rr_cfg.get("top_n", 30))
    if vs.get("n"):
        params["visual_slots"] = {"k": vs_k, "n": int(vs["n"])}
        if vs.get("k") and cfg_m3.get("budget", {}).get("max_parents") and vs_k > cfg_m3["budget"]["max_parents"]:
            params["visual_slots"]["warning"] = "k > max_parents: bảng được giữ chỗ có thể không đến model"
    return RetrievalResult(
        claim_id=claim.claim_id, method=method, table_condition=index.condition,
        params=params,
        n_units_in_pool=len(index.units), n_parents_in_pool=len(set(parents)),
        candidates=cands, pool_hash=index.pool_hash)


def units_of_parent(index: PaperIndex, parent: str) -> list[tuple[int, EvidenceUnit]]:
    return [(i, u) for i, u in enumerate(index.units) if parent_key(u) == parent]
