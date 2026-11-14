"""Chỉ số toàn chuỗi cho kết luận Entailed/Refuted — SPEC 12.2, 12.3.

Dự đoán None (không đọc được câu trả lời) tính là sai và được báo riêng.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, pstdev

from sciverx.schemas import GoldInfo, VerifierOutput


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def accuracy_f1(preds: list[bool | None], labels: list[bool]) -> dict[str, float]:
    n = len(labels)
    correct = sum(p is not None and p == y for p, y in zip(preds, labels))
    # macro-F1 trên hai lớp; None tính là dự đoán sai lớp
    tp_t = sum(p is True and y for p, y in zip(preds, labels))
    fp_t = sum(p is True and not y for p, y in zip(preds, labels))
    fn_t = sum(p is not True and y for p, y in zip(preds, labels))
    tp_f = sum(p is False and not y for p, y in zip(preds, labels))
    fp_f = sum(p is False and y for p, y in zip(preds, labels))
    fn_f = sum(p is not False and not y for p, y in zip(preds, labels))
    return {"n": n, "accuracy": correct / n if n else 0.0,
            "macro_f1": (_f1(tp_t, fp_t, fn_t) + _f1(tp_f, fp_f, fn_f)) / 2,
            "unparsed_pct": 100 * sum(p is None for p in preds) / n if n else 0.0}


def by_sample(outputs: list[VerifierOutput], golds: dict[str, GoldInfo], use: str = "pred_new"
              ) -> dict[str, dict[str, bool | None]]:
    """sample_name → claim_id → dự đoán."""
    out: dict[str, dict[str, bool | None]] = defaultdict(dict)
    for o in outputs:
        out[o.sample_name][o.claim_id] = getattr(o, use)
    return out


def majority(preds: list[bool | None]) -> bool | None:
    votes = Counter(p for p in preds if p is not None)
    if not votes:
        return None
    (top, n), *rest = votes.most_common()
    return None if rest and rest[0][1] == n else top


def condition_summary(outputs: list[VerifierOutput], golds: dict[str, GoldInfo], use: str = "pred_new",
                      seed_samples: tuple[str, ...] = ("t1_s1", "t1_s2", "t1_s3")) -> dict:
    per = by_sample(outputs, golds, use)
    ids = sorted(set().union(*(p.keys() for p in per.values()))) if per else []
    labels = [golds[c].label for c in ids]
    res: dict = {"n_claims": len(ids)}
    seeds = [s for s in seed_samples if s in per]
    accs = [accuracy_f1([per[s].get(c) for c in ids], labels)["accuracy"] for s in seeds]
    if accs:
        res["seed_accuracy_mean"] = mean(accs)
        res["seed_accuracy_std"] = pstdev(accs) if len(accs) > 1 else 0.0
        consistent = sum(len({per[s].get(c) for s in seeds}) == 1 for c in ids)
        res["consistent_pct"] = 100 * consistent / len(ids) if ids else 0.0
        res["majority"] = accuracy_f1([majority([per[s].get(c) for s in seeds]) for c in ids], labels)
    if "t0" in per:
        res["t0"] = accuracy_f1([per["t0"].get(c) for c in ids], labels)
    trunc = [o for o in outputs if o.finish_reason == "length"]
    res["truncated_pct"] = 100 * len(trunc) / len(outputs) if outputs else 0.0
    return res


def grouped_accuracy(preds: dict[str, bool | None], golds: dict[str, GoldInfo], key) -> dict[str, dict]:
    """Accuracy theo nhóm, vd key=lambda g: g.claim_type hoặc nhãn."""
    groups: dict[str, list[str]] = defaultdict(list)
    for cid in preds:
        groups[str(key(golds[cid]))].append(cid)
    return {g: accuracy_f1([preds[c] for c in ids], [golds[c].label for c in ids]) for g, ids in groups.items()}
