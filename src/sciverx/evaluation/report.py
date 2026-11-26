"""Stage `evaluate` / `report`: đọc đầu ra của lượt chạy và GoldInfo, ghi chỉ số vào eval/.

Hiện có: chỉ số truy xuất (X2) cho mọi file retrieval/{cond}_{method}.jsonl, tách
theo claim_type. So sánh G1↔E1 chỉ trên PAIRED_RETR (danh sách trong
paired_retr.txt nếu có). Chỉ số kết luận dùng verdict_metrics/stats khi đã có
verifier_outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

from sciverx.evaluation.gold_access import load_gold
from sciverx.evaluation.retrieval_metrics import (compare_runs, evaluate_retrieval, evidence_kind,
                                                  per_claim_delivered, per_claim_metrics, summarize_rows)
from sciverx.m1_prepare import work_dir
from sciverx.schemas import RetrievalResult
from sciverx.utils.io import read_jsonl
from sciverx.utils.manifest import find_or_make_run_id


def run_dir_of(cfg: dict) -> Path:
    if cfg.get("eval", {}).get("run_dir"):
        return Path(cfg["eval"]["run_dir"])
    out = cfg["paths"]["outputs"]
    return Path(out) / find_or_make_run_id(out, cfg.get("experiment", "x2_retrieval"), cfg["split"], cfg)


def evaluate_retrieval_dir(run_dir: Path, gold_path: Path, paired: set[str] | None = None,
                           paper_of: dict[str, str] | None = None) -> dict:
    """Chỉ số cho từng file retrieval/{cond}_{tên}.jsonl, tách theo claim_type và theo
    loại bằng chứng (ev:table_only, ev:chart_only, ev:table+chart). Thêm khóa
    "compare/{A}__vs__{B}": so từng cặp file cùng điều kiện bảng (McNemar, bootstrap theo bài)."""
    golds = load_gold(gold_path)
    by_type = {cid: g.claim_type for cid, g in golds.items()}
    by_ev = {cid: "ev:" + evidence_kind(g) for cid, g in golds.items()}
    out = {}
    per: dict[str, dict] = {}
    delivered: dict[str, dict] = {}
    for f in sorted((run_dir / "retrieval").glob("*.jsonl")):
        res = read_jsonl(f, RetrievalResult)
        out[f.stem] = evaluate_retrieval(res, golds, group_by=by_type)
        out[f.stem].update({k: v for k, v in evaluate_retrieval(res, golds, group_by=by_ev).items() if k != "all"})
        if paired is not None:
            out[f.stem + "@PAIRED_RETR"] = evaluate_retrieval([r for r in res if r.claim_id in paired], golds)
        per[f.stem] = per_claim_metrics(res, golds)
        sel_path = run_dir / "selection" / f.name
        if sel_path.exists():                                    # thứ model thật sự nhận
            delivered[f.stem] = per_claim_delivered(read_jsonl(sel_path), golds)
            out[f"delivered/{f.stem}"] = summarize_rows(delivered[f.stem], by_type)
    names = sorted(per)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a.split("_", 1)[0] == b.split("_", 1)[0]:        # cùng điều kiện bảng (G1, E1…)
                out[f"compare/{a}__vs__{b}"] = compare_runs(per[a], per[b], paper_of)
                if a in delivered and b in delivered:
                    out[f"compare-delivered/{a}__vs__{b}"] = compare_runs(
                        delivered[a], delivered[b], paper_of,
                        metrics=("delivered_visual_complete", "delivered_recall"))
    return out


def write_paired_sets(cfg: dict) -> dict[str, int]:
    """Ghi paired_retr.txt / paired_oracle.txt vào thư mục làm việc; trả kích thước."""
    from sciverx.evaluation.paired_sets import paired_oracle, paired_retr
    from sciverx.m1_prepare import tables_path
    from sciverx.schemas import ClaimInput, TableRecord
    wd = work_dir(cfg)
    golds = load_gold(wd / "gold.jsonl")
    paper_of = {c.claim_id: c.paper_id for c in read_jsonl(wd / "claims_input.jsonl", ClaimInput)}
    g0 = read_jsonl(tables_path(cfg, "G0"), TableRecord) if tables_path(cfg, "G0").exists() else []
    sizes = {"n_claims": len(golds)}
    if g0:
        pr = paired_retr(golds, paper_of, g0)
        (wd / "paired_retr.txt").write_text("\n".join(pr) + "\n", encoding="utf-8")
        sizes["PAIRED_RETR"] = len(pr)
        if tables_path(cfg, "E0").exists():
            po = paired_oracle(golds, paper_of, g0, read_jsonl(tables_path(cfg, "E0"), TableRecord))
            (wd / "paired_oracle.txt").write_text("\n".join(po) + "\n", encoding="utf-8")
            sizes["PAIRED_ORACLE"] = len(po)
    return sizes


def run_evaluate(cfg: dict, args) -> int:
    rd = run_dir_of(cfg)
    print(f"[evaluate] tập ghép cặp: {write_paired_sets(cfg)}")
    paired_p = work_dir(cfg) / "paired_retr.txt"
    paired = set(paired_p.read_text().split()) if paired_p.exists() else None
    from sciverx.schemas import ClaimInput
    claims_p = work_dir(cfg) / "claims_input.jsonl"
    paper_of = {c.claim_id: c.paper_id for c in read_jsonl(claims_p, ClaimInput)} if claims_p.exists() else None
    metrics = evaluate_retrieval_dir(rd, work_dir(cfg) / "gold.jsonl", paired, paper_of)
    out = rd / "eval" / "retrieval_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, groups in metrics.items():
        if name.startswith("compare-delivered/"):
            vc = groups["delivered_visual_complete"]
            print(f"[evaluate] {name}: n={groups['n_claims']} đủ bảng khi đến model Δ={vc['diff']} "
                  f"[{vc.get('ci_low')}, {vc.get('ci_high')}] p={vc.get('mcnemar_p')}")
            continue
        if name.startswith("compare/"):
            vc = groups["visual_complete@5"]
            print(f"[evaluate] {name}: n={groups['n_claims']} recall@5 Δ={groups['recall@5']['diff']} "
                  f"visual@5 Δ={vc['diff']} [{vc.get('ci_low')}, {vc.get('ci_high')}] p={vc.get('mcnemar_p')}")
            continue
        if name.startswith("delivered/"):
            a = groups["all"]
            by = " ".join(f"{t}={groups[t]['delivered_visual_complete']}"
                          for t in ("direct", "analytical", "parallel", "sequential") if t in groups)
            print(f"[evaluate] {name}: n={a['n_claims']} đủ bảng khi đến model={a['delivered_visual_complete']} "
                  f"({by}) token TB={a['evidence_tokens']} bị cắt={a['truncated']}")
            continue
        a = groups["all"]
        print(f"[evaluate] {name}: n={a['n_claims']} hit@5={a['hit@5']} recall@5={a['recall@5']} "
              f"visual@5={a['visual_complete@5']} mrr={a['mrr']}")
    return 0


def run_report(cfg: dict, args) -> int:
    """So kết luận của verifier giữa các điều kiện trong lượt chạy (vd B0, IMG, G1).
    Ghi eval/verdict_report.json và eval/verdict_report.md."""
    from sciverx.evaluation.verdict_report import build_report, to_markdown
    from sciverx.schemas import ClaimInput
    rd = run_dir_of(cfg)
    if not list(rd.glob("*/verifier_outputs.jsonl")):
        print(f"[report] chưa có verifier_outputs.jsonl trong {rd}")
        return 2
    golds = load_gold(work_dir(cfg) / "gold.jsonl")
    paper_of = {c.claim_id: c.paper_id for c in read_jsonl(work_dir(cfg) / "claims_input.jsonl", ClaimInput)}
    ev = cfg.get("eval", {})
    rep = build_report(rd, golds, paper_of, ev.get("reference", "IMG"), ev.get("bootstrap_n", 1000))
    out = rd / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(rep)
    (out / "verdict_report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"[report] → {out / 'verdict_report.md'}")
    return 0
