"""Báo cáo kết luận của verifier: so các điều kiện trong một lượt chạy (vd B0, IMG, G1).

Đọc `<run_dir>/<điều kiện>/verifier_outputs.jsonl` (B0 do `b0-import` ghi). Với mỗi
điều kiện và mỗi cách chấm (pred_new chính, pred_old để đối chiếu paper):
- accuracy, macro-F1 từng lượt sinh (t0, t1_s1..3, b0); TB ± SD qua 3 seed; % nhất
  quán; accuracy bỏ phiếu đa số;
- % ok / parse_error / empty / error, % bị cắt;
- accuracy theo claim_type và theo nhãn (ở "lượt chính": t0, nếu không có thì b0).
So với điều kiện tham chiếu (mặc định IMG) trên CÙNG claim: McNemar chính xác và CI 95%
bootstrap theo bài, cho từng "góc nhìn" chung (t0, từng seed, majority); B0 (một lượt,
temperature 1.0, không seed) được so với từng lượt temperature 1.0 và với t0 của bên kia.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev

from sciverx.evaluation.stats import cluster_bootstrap_diff, mcnemar_exact
from sciverx.evaluation.verdict_metrics import accuracy_f1, condition_summary, majority
from sciverx.schemas import GoldInfo, VerifierOutput
from sciverx.utils.io import read_jsonl

SEEDS = ("t1_s1", "t1_s2", "t1_s3")
ORDER = ("B0", "IMG", "G0", "G1", "E0", "E1")


def load_conditions(run_dir: Path) -> dict[str, list[VerifierOutput]]:
    found = {p.parent.name: read_jsonl(p, VerifierOutput) for p in Path(run_dir).glob("*/verifier_outputs.jsonl")}
    return {k: found[k] for k in sorted(found, key=lambda c: (ORDER.index(c) if c in ORDER else 99, c))}


def views(outputs: list[VerifierOutput], use: str) -> dict[str, dict[str, bool | None]]:
    """Tên góc nhìn → claim_id → dự đoán. Gồm từng lượt sinh và 'majority' (≥ 3 seed)."""
    v: dict[str, dict[str, bool | None]] = {}
    for o in outputs:
        v.setdefault(o.sample_name, {})[o.claim_id] = getattr(o, use)
    seeds = [s for s in SEEDS if s in v]
    if len(seeds) >= 3:
        ids = set.intersection(*(set(v[s]) for s in seeds))
        v["majority"] = {c: majority([v[s][c] for s in seeds]) for c in ids}
    return v


def main_view(v: dict) -> str:
    for name in ("t0", "b0", "majority", *SEEDS):
        if name in v:
            return name
    return next(iter(v))


def _acc_by(preds: dict[str, bool | None], golds: dict[str, GoldInfo], key) -> dict[str, dict]:
    groups: dict[str, list[str]] = {}
    for c in preds:
        groups.setdefault(str(key(golds[c])), []).append(c)
    return {g: accuracy_f1([preds[c] for c in ids], [golds[c].label for c in ids]) for g, ids in sorted(groups.items())}


def summarize_condition(outputs: list[VerifierOutput], golds: dict[str, GoldInfo]) -> dict:
    out: dict = {"n_outputs": len(outputs)}
    base = condition_summary(outputs, golds, "pred_new", SEEDS)
    out["status_pct"] = base.get("status_pct", {})
    out["truncated_pct"] = base.get("truncated_pct", 0.0)
    for use in ("pred_new", "pred_old"):
        v = views(outputs, use)
        res: dict = {"samples": {name: accuracy_f1(list(p.values()), [golds[c].label for c in p])
                                 for name, p in v.items()}}
        accs = [res["samples"][s]["accuracy"] for s in SEEDS if s in res["samples"]]
        if len(accs) > 1:
            res["seed_mean"], res["seed_sd"] = mean(accs), pstdev(accs)
            ids = set.intersection(*(set(v[s]) for s in SEEDS if s in v))
            res["consistent_pct"] = 100 * sum(len({v[s][c] for s in SEEDS if s in v}) == 1 for c in ids) / len(ids)
        mv = main_view(v)
        res["main_view"] = mv
        res["by_claim_type"] = _acc_by(v[mv], golds, lambda g: g.claim_type)
        res["by_label"] = _acc_by(v[mv], golds, lambda g: "Entailed" if g.label else "Refuted")
        out[use] = res
    return out


def compare(a_out: list[VerifierOutput], b_out: list[VerifierOutput], golds: dict[str, GoldInfo],
            paper_of: dict[str, str] | None, use: str = "pred_new", n_boot: int = 1000, seed: int = 2026) -> dict:
    """So B với A (A = tham chiếu). Trả {góc_nhìn: {acc_a, acc_b, diff, ci, mcnemar}}."""
    va, vb = views(a_out, use), views(b_out, use)
    pairs = [(n, n) for n in va if n in vb]
    if "b0" in vb and "b0" not in va:      # B0 (một lượt, temp 1.0) so với từng lượt của tham chiếu
        pairs += [(n, "b0") for n in (*SEEDS, "t0") if n in va]
    if "b0" in va and "b0" not in vb:
        pairs += [("b0", n) for n in (*SEEDS, "t0") if n in vb]
    out = {}
    for na, nb in pairs:
        ids = sorted(set(va[na]) & set(vb[nb]))
        if not ids:
            continue
        ca = {c: va[na][c] is not None and va[na][c] == golds[c].label for c in ids}
        cb = {c: vb[nb][c] is not None and vb[nb][c] == golds[c].label for c in ids}
        row = {"n": len(ids), "acc_a": sum(ca.values()) / len(ids), "acc_b": sum(cb.values()) / len(ids)}
        row["diff"] = row["acc_b"] - row["acc_a"]
        mc = mcnemar_exact([ca[c] for c in ids], [cb[c] for c in ids])
        row.update({"a_only_correct": mc["b"], "b_only_correct": mc["c"], "mcnemar_p": mc["p_value"]})
        if paper_of:
            bs = cluster_bootstrap_diff(ca, cb, {c: paper_of[c] for c in ids}, n_boot=n_boot, seed=seed)
            row.update({"ci_low": bs["ci_low"], "ci_high": bs["ci_high"]})
        out[na if na == nb else f"{na}~{nb}"] = row
    return out


def build_report(run_dir: Path, golds: dict[str, GoldInfo], paper_of: dict[str, str] | None,
                 reference: str = "IMG", n_boot: int = 1000) -> dict:
    conds = load_conditions(run_dir)
    rep: dict = {"run_dir": str(run_dir), "reference": reference, "conditions": {}, "compare": {}}
    for name, outs in conds.items():
        rep["conditions"][name] = summarize_condition(outs, golds)
    if reference in conds:
        for name, outs in conds.items():
            if name == reference:
                continue
            for use in ("pred_new", "pred_old"):
                rep["compare"][f"{name}_vs_{reference}/{use}"] = compare(
                    conds[reference], outs, golds, paper_of, use, n_boot=n_boot)
    return rep


def _p(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def to_markdown(rep: dict) -> str:
    L = [f"# Báo cáo verifier\n\nLượt chạy: `{rep['run_dir']}` · tham chiếu: **{rep['reference']}**\n",
         "## 1. Accuracy (pred_new; pred_old trong ngoặc)\n",
         "| Điều kiện | t0 | TB 3 seed ± SD | Majority | B0 | % parse_error | % rỗng | % lỗi | % bị cắt |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, c in rep["conditions"].items():
        n, o = c["pred_new"], c["pred_old"]

        def cell(view, n=n, o=o):
            s, t = n["samples"].get(view), o["samples"].get(view)
            return f"{_p(s['accuracy'])} ({_p(t['accuracy'])})" if s else "—"
        seeds = (f"{_p(n['seed_mean'])} ± {_p(n['seed_sd'])} ({_p(o['seed_mean'])})"
                 if "seed_mean" in n else "—")
        st = c["status_pct"]
        L.append(f"| {name} | {cell('t0')} | {seeds} | {cell('majority')} | {cell('b0')} | "
                 f"{st.get('parse_error', 0):.1f} | {st.get('empty', 0):.1f} | {st.get('error', 0):.1f} | "
                 f"{c['truncated_pct']:.1f} |")
    L += ["\n## 2. Theo loại claim và nhãn (pred_new, lượt chính)\n"]
    types = sorted({t for c in rep["conditions"].values() for t in c["pred_new"]["by_claim_type"]})
    L += ["| Điều kiện (lượt) | " + " | ".join(types) + " | Entailed | Refuted |",
          "|---" * (len(types) + 3) + "|"]
    for name, c in rep["conditions"].items():
        n = c["pred_new"]
        row = [_p(n["by_claim_type"].get(t, {}).get("accuracy")) for t in types]
        lab = [_p(n["by_label"].get(k, {}).get("accuracy")) for k in ("Entailed", "Refuted")]
        L.append(f"| {name} ({n['main_view']}) | " + " | ".join(row + lab) + " |")
    L += [f"\n## 3. So với {rep['reference']} trên cùng claim\n",
          "| So sánh | Góc nhìn | n | Acc tham chiếu | Acc | Δ (điểm) | CI 95% (bootstrap theo bài) | McNemar p |",
          "|---|---|---|---|---|---|---|---|"]
    for key, rows in rep["compare"].items():
        for view, r in rows.items():
            ci = f"[{_p(r['ci_low'])}, {_p(r['ci_high'])}]" if "ci_low" in r else "—"
            L.append(f"| {key} | {view} | {r['n']} | {_p(r['acc_a'])} | {_p(r['acc_b'])} | "
                     f"{100 * r['diff']:+.1f} | {ci} | {r['mcnemar_p']:.4f} |")
    L += ["\nĐọc: Δ > 0 nghĩa là điều kiện tốt hơn tham chiếu. Chỉ coi là khác biệt khi CI không chứa 0 "
          "và p < 0,05. pred_old là cách chấm của paper; dùng để đối chiếu, không dùng kết luận."]
    return "\n".join(L) + "\n"
