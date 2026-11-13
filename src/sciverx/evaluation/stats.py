"""Thống kê so sánh cặp — SPEC 12.3.

- McNemar chính xác (nhị thức hai phía trên các cặp bất đồng).
- Khoảng tin cậy 95% của hiệu accuracy bằng bootstrap theo CỤM BÀI BÁO: lấy
  mẫu lại bài (có hoàn lại), giữ mọi claim của bài; 1.000 lần, seed 2026.
- flips: các claim đổi đúng/sai giữa hai điều kiện, để phân tích nguyên nhân.
"""

from __future__ import annotations

import random
from math import comb


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """b = số cặp A đúng B sai; c = A sai B đúng. p hai phía = min(1, 2·P(X ≤ min(b,c))), X~Bin(b+c, 0.5)."""
    if len(a_correct) != len(b_correct):
        raise ValueError("hai điều kiện phải trên cùng tập claim")
    b = sum(x and not y for x, y in zip(a_correct, b_correct))
    c = sum(y and not x for x, y in zip(a_correct, b_correct))
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"b": b, "c": c, "p_value": min(1.0, p)}


def cluster_bootstrap_diff(a_correct: dict[str, bool], b_correct: dict[str, bool], cluster_of: dict[str, str],
                           n_boot: int = 1000, seed: int = 2026, alpha: float = 0.05) -> dict:
    """Hiệu accuracy (B - A) và khoảng tin cậy percentile theo cụm."""
    ids = sorted(a_correct)
    if set(ids) != set(b_correct):
        raise ValueError("hai điều kiện phải trên cùng tập claim")
    clusters: dict[str, list[str]] = {}
    for cid in ids:
        clusters.setdefault(cluster_of[cid], []).append(cid)
    keys = sorted(clusters)

    def diff(sample_ids: list[str]) -> float:
        return (sum(b_correct[c] for c in sample_ids) - sum(a_correct[c] for c in sample_ids)) / len(sample_ids)

    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        chosen = [cid for k in (rng.choice(keys) for _ in keys) for cid in clusters[k]]
        stats.append(diff(chosen))
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"diff": diff(ids), "ci_low": lo, "ci_high": hi, "n_clusters": len(keys),
            "mean_claims_per_cluster": len(ids) / len(keys), "n_boot": n_boot, "seed": seed}


def flips(a_correct: dict[str, bool], b_correct: dict[str, bool]) -> list[dict]:
    """Claim đổi kết quả; kèm cột trống để người phân tích điền (SPEC 12.3)."""
    out = []
    for cid in sorted(a_correct):
        if a_correct[cid] != b_correct[cid]:
            out.append({"claim_id": cid, "a_correct": a_correct[cid], "b_correct": b_correct[cid],
                        "observed_errors": "", "suspected_causes": "", "certainty": ""})
    return out
