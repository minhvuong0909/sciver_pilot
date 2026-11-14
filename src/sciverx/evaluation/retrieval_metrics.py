"""Chỉ số truy xuất ở mức cha, đã loại trùng — SPEC 12.1.

G: tập bằng chứng đúng (bảng theo item, đoạn văn theo section_ids); R_k: top-k cha.
- Hit@k: 1 nếu |G ∩ R_k| ≥ 1
- Recall@k: |G ∩ R_k| / |G|
- VisualComplete@k: 1 nếu mọi bằng chứng thị giác trong G nằm trong R_k
- TextRecall@k: Recall@k chỉ trên section
- MRR: 1 / thứ hạng của bằng chứng đúng đầu tiên (0 nếu không có)
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from sciverx.evaluation.gold_access import (gold_section_parents, gold_visual_parents,
                                            section_matches)
from sciverx.schemas import GoldInfo, RetrievalResult

KS = (1, 3, 5, 10)


def _hit(gold_key: str, retrieved: list[str]) -> bool:
    if gold_key.startswith("section:"):
        return any(section_matches(r, gold_key) for r in retrieved)
    return gold_key in retrieved


def _is_relevant(parent: str, gold_keys: list[str]) -> bool:
    return any(parent == g or section_matches(parent, g) for g in gold_keys)


def claim_metrics(ranked: list[str], gold: GoldInfo, ks=KS) -> dict[str, float]:
    visual = gold_visual_parents(gold)
    text = gold_section_parents(gold)
    all_gold = visual + text
    out: dict[str, float] = {}
    for k in ks:
        top = ranked[:k]
        hits = [g for g in all_gold if _hit(g, top)]
        out[f"hit@{k}"] = float(bool(hits))
        out[f"recall@{k}"] = len(hits) / len(all_gold) if all_gold else 0.0
        out[f"visual_complete@{k}"] = float(all(_hit(g, top) for g in visual)) if visual else 0.0
        out[f"text_recall@{k}"] = (sum(_hit(g, top) for g in text) / len(text)) if text else float("nan")
    first = next((i for i, p in enumerate(ranked, 1) if _is_relevant(p, all_gold)), None)
    out["mrr"] = 1.0 / first if first else 0.0
    return out


def evaluate_retrieval(results: list[RetrievalResult], golds: dict[str, GoldInfo], ks=KS,
                       group_by: dict[str, str] | None = None) -> dict[str, dict[str, float]]:
    """Trung bình theo claim; `group_by` (claim_id → nhóm, vd claim_type) cho thêm bảng con."""
    per: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        m = claim_metrics([c.parent_id for c in r.candidates], golds[r.claim_id], ks)
        m["n_parents_in_pool"] = r.n_parents_in_pool
        per["all"].append(m)
        if group_by:
            per[group_by[r.claim_id]].append(m)
    out = {}
    for g, rows in per.items():
        keys = rows[0].keys()
        agg = {}
        for k in keys:
            vals = [x[k] for x in rows if x[k] == x[k]]  # bỏ NaN (claim không có section chuẩn)
            agg[k] = round(mean(vals), 4) if vals else float("nan")
        agg["n_claims"] = len(rows)
        out[g] = agg
    return out
