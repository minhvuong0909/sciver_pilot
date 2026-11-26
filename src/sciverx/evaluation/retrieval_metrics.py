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


def evidence_kind(gold: GoldInfo) -> str:
    """Nhóm claim theo loại bằng chứng thị giác: table_only | chart_only | table+chart | none."""
    types = {v["item_type"] for v in gold.visual_items}
    if not types:
        return "none"
    has_t, has_c = "table" in types, bool(types - {"table"})
    return "table+chart" if has_t and has_c else ("table_only" if has_t else "chart_only")


def delivered_metrics(selection: dict, gold: GoldInfo) -> dict[str, float]:
    """Chỉ số trên thứ model THẬT SỰ nhận (selection/*.jsonl sau ngân sách M3), không
    phải danh sách xếp hạng. Bảng bị cắt hàng vẫn tính là đã đưa (cắt được ghi riêng).
    - delivered_visual_complete: 1 nếu mọi bảng/biểu đồ chuẩn có trong bằng chứng gửi đi
    - delivered_recall: tỷ lệ bằng chứng chuẩn (bảng, biểu đồ, section) có trong đó"""
    sent = [p["parent_id"] for p in selection.get("parents", [])]
    visual = gold_visual_parents(gold)
    all_gold = visual + gold_section_parents(gold)
    budget = selection.get("budget", {})
    return {
        "delivered_visual_complete": float(all(_hit(g, sent) for g in visual)) if visual else 0.0,
        "delivered_recall": sum(_hit(g, sent) for g in all_gold) / len(all_gold) if all_gold else 0.0,
        "n_parents_sent": float(len(sent)),
        "evidence_tokens": float(budget.get("tokens_used", 0)),
        "truncated": float(bool(budget.get("truncated"))),
        "dropped_for_budget": float(bool(budget.get("dropped_parents"))),
    }


def per_claim_delivered(selections: list[dict], golds: dict[str, GoldInfo]) -> dict[str, dict]:
    return {s["claim_id"]: delivered_metrics(s, golds[s["claim_id"]]) for s in selections}


def summarize_rows(per: dict[str, dict], group_by: dict[str, str] | None = None) -> dict[str, dict]:
    """Trung bình theo claim, nhóm 'all' và (tùy chọn) theo group_by."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for cid, m in per.items():
        groups["all"].append(m)
        if group_by:
            groups[group_by[cid]].append(m)
    out = {}
    for g, rows in groups.items():
        out[g] = {k: round(mean(r[k] for r in rows), 4) for k in rows[0]}
        out[g]["n_claims"] = len(rows)
    return out


def per_claim_metrics(results: list[RetrievalResult], golds: dict[str, GoldInfo], ks=KS) -> dict[str, dict]:
    return {r.claim_id: claim_metrics([c.parent_id for c in r.candidates], golds[r.claim_id], ks)
            for r in results}


def compare_runs(a: dict[str, dict], b: dict[str, dict], paper_of: dict[str, str] | None = None,
                 metrics=("hit@5", "recall@5", "visual_complete@5"), n_boot: int = 1000,
                 seed: int = 2026) -> dict:
    """So B với A trên CÙNG tập claim (giao của hai bên).
    - Chỉ số 0/1 (hit, visual_complete): McNemar chính xác.
    - Mọi chỉ số: hiệu trung bình (B − A) và CI 95% bootstrap theo cụm bài (nếu có paper_of)."""
    from sciverx.evaluation.stats import cluster_bootstrap_diff, mcnemar_exact
    ids = sorted(set(a) & set(b))
    out: dict = {"n_claims": len(ids)}
    for m in metrics:
        va = {c: a[c][m] for c in ids}
        vb = {c: b[c][m] for c in ids}
        row: dict = {"a": round(mean(va.values()), 4) if ids else 0.0,
                     "b": round(mean(vb.values()), 4) if ids else 0.0}
        row["diff"] = round(row["b"] - row["a"], 4)
        if all(v in (0.0, 1.0) for v in [*va.values(), *vb.values()]):
            mc = mcnemar_exact([bool(va[c]) for c in ids], [bool(vb[c]) for c in ids])
            row.update({"b_only_better": mc["c"], "a_only_better": mc["b"], "mcnemar_p": round(mc["p_value"], 4)})
        if paper_of and ids:
            bs = cluster_bootstrap_diff(va, vb, {c: paper_of[c] for c in ids}, n_boot=n_boot, seed=seed)
            row.update({"ci_low": round(bs["ci_low"], 4), "ci_high": round(bs["ci_high"], 4)})
        out[m] = row
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
