"""Stage `retrieve`: chạy BM25 / dense / hybrid trên kho G1 và E1 cho mọi claim.

Chỉ đọc claims_input.jsonl (không đọc gold.jsonl). Ghi vào thư mục lượt chạy:
retrieval/{cond}_{method}.jsonl và selection/{cond}_{method}.jsonl (cha được
chọn theo ngân sách), kèm manifest.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from sciverx.m1_prepare import tables_path, work_dir
from sciverx.m1_prepare.index import load_indexes
from sciverx.m3_retrieve.budget import select_evidence
from sciverx.m3_retrieve.retrieve import retrieve
from sciverx.schemas import ClaimInput, TableRecord
from sciverx.utils.io import read_jsonl, write_jsonl
from sciverx.utils.manifest import build_manifest, make_run_id, open_run_dir


def qwen_token_counter(model: str) -> Callable[[str], int]:  # pragma: no cover
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    return lambda text: len(tok(text, add_special_tokens=False)["input_ids"])


def run_retrieval(cfg: dict, claims: list[ClaimInput], indexes: dict, tables: dict,
                  cond: str, method: str, count, embedder=None) -> tuple[list, list]:
    m3 = {**cfg["m3"], "embed_model": cfg["m1"]["units"]["embed_model"]}
    allow_mixed = cfg.get("selection", {}).get("allow_mixed", False)
    results, selections = [], []
    qvecs = embedder.encode([c.text for c in claims]) if embedder is not None else None
    for i, c in enumerate(claims):
        idx = indexes[c.paper_id]
        res = retrieve(c, idx, method, m3, qvecs[i] if qvecs is not None else None)
        sel = select_evidence(res, idx, tables.get(c.paper_id, {}), m3["budget"], count,
                              allow_mixed=allow_mixed)
        results.append(res)
        selections.append({"claim_id": c.claim_id, "method": method, "table_condition": cond,
                           "parents": [asdict(p) for p in sel.parents], "budget": sel.budget})
    return results, selections


def run(cfg: dict, args) -> int:  # pragma: no cover - cần model thật; logic đã test riêng
    claims = read_jsonl(work_dir(cfg) / "claims_input.jsonl", ClaimInput)
    if cfg.get("limit"):
        claims = claims[: cfg["limit"]]
    methods = cfg["m3"]["methods"]
    need_dense = any(m in ("dense", "hybrid") for m in methods)
    embedder = None
    if need_dense:
        from sciverx.m3_retrieve.rankers import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder(cfg["m1"]["units"]["embed_model"])
    count = qwen_token_counter(cfg["verifier"]["model"])
    exp = cfg.get("experiment", "x2_retrieval")
    run_id = make_run_id(exp, cfg["split"], cfg)
    run_dir = open_run_dir(cfg["paths"]["outputs"], build_manifest(run_id, cfg, [c.claim_id for c in claims]))
    for cond in ("G1", "E1"):
        if not tables_path(cfg, cond).exists():
            continue
        tables: dict[str, dict[str, TableRecord]] = {}
        for r in read_jsonl(tables_path(cfg, cond), TableRecord):
            tables.setdefault(r.paper_id, {})[r.table_id] = r
        indexes = load_indexes(cfg, cond, need_dense)
        for method in methods:
            res, sel = run_retrieval(cfg, claims, indexes, tables, cond, method, count,
                                     embedder if method != "bm25" else None)
            write_jsonl(Path(run_dir) / "retrieval" / f"{cond}_{method}.jsonl", res)
            write_jsonl(Path(run_dir) / "selection" / f"{cond}_{method}.jsonl", sel)
            print(f"[retrieve] {cond}/{method}: {len(res)} claim")
    print(f"[retrieve] → {run_dir}")
    return 0
