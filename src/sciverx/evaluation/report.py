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
from sciverx.evaluation.retrieval_metrics import evaluate_retrieval
from sciverx.m1_prepare import work_dir
from sciverx.schemas import RetrievalResult
from sciverx.utils.io import read_jsonl
from sciverx.utils.manifest import make_run_id


def run_dir_of(cfg: dict) -> Path:
    if cfg.get("eval", {}).get("run_dir"):
        return Path(cfg["eval"]["run_dir"])
    return Path(cfg["paths"]["outputs"]) / make_run_id(cfg.get("experiment", "x2_retrieval"), cfg["split"], cfg)


def evaluate_retrieval_dir(run_dir: Path, gold_path: Path, paired: set[str] | None = None) -> dict:
    golds = load_gold(gold_path)
    by_type = {cid: g.claim_type for cid, g in golds.items()}
    out = {}
    for f in sorted((run_dir / "retrieval").glob("*.jsonl")):
        res = read_jsonl(f, RetrievalResult)
        out[f.stem] = evaluate_retrieval(res, golds, group_by=by_type)
        if paired is not None:
            out[f.stem + "@PAIRED_RETR"] = evaluate_retrieval([r for r in res if r.claim_id in paired], golds)
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
    metrics = evaluate_retrieval_dir(rd, work_dir(cfg) / "gold.jsonl", paired)
    out = rd / "eval" / "retrieval_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, groups in metrics.items():
        a = groups["all"]
        print(f"[evaluate] {name}: n={a['n_claims']} hit@1={a['hit@1']} hit@3={a['hit@3']} "
              f"recall@3={a['recall@3']} visual@3={a['visual_complete@3']} mrr={a['mrr']}")
    return 0


def run_report(cfg: dict, args) -> int:  # pragma: no cover - gom bảng ở GĐ B
    print("[report] chưa làm: tổng hợp báo cáo Mức B (bảng chỉ số, CI, McNemar)")
    return 3
