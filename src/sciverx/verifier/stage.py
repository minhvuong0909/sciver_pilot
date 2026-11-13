"""Stage `verify` (và `align` / `numeric` / `package` cho S-align) trên máy GPU.

Cấu hình thí nghiệm chọn:
  verifier.setting: S-oracle | S-retr | S-align4 | S-align45
  verifier.table_conditions: [IMG, G0, G1, E0, E1] (S-oracle) hoặc [G1, E1]
  verifier.unified: true → dùng unified_v1 ở S-oracle (X1u)
  m3.methods: phương pháp truy xuất cho S-retr/S-align (mặc định hybrid)
  verifier.sample_names: lượt sinh cần chạy (mặc định cả 4)
Ghi vào outputs/runs/<run_id>/: verifier_requests.jsonl, verifier_outputs.jsonl,
packages.jsonl (S-align). GoldInfo chỉ được đọc để dựng OracleEvidenceLocator ở
S-oracle (bỏ label).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from sciverx.data.loader import load_paper
from sciverx.evaluation.gold_access import load_gold
from sciverx.m1_prepare import tables_path, units_path, work_dir
from sciverx.m3_retrieve.retrieve import PaperIndex
from sciverx.pipeline import run_claim
from sciverx.schemas import ClaimInput, EvidenceUnit, OracleEvidenceLocator, TableRecord
from sciverx.utils.io import read_jsonl, write_jsonl
from sciverx.utils.manifest import build_manifest, make_run_id, open_run_dir
from sciverx.verifier.build_inputs import BuiltInput, build_oracle
from sciverx.verifier.run_vllm import preprocess_image, run_samples


def _tables_by_paper(cfg: dict, cond: str) -> dict[str, dict[str, TableRecord]]:
    out: dict[str, dict[str, TableRecord]] = {}
    for r in read_jsonl(tables_path(cfg, cond), TableRecord):
        out.setdefault(r.paper_id, {})[r.table_id] = r
    return out


def build_all_inputs(cfg: dict, claims: list[ClaimInput], cond: str, llm, count: Callable[[str], int],
                     embedder=None) -> tuple[list[BuiltInput], list]:
    """Trả (inputs, packages). S-oracle đọc vị trí bằng chứng (không label) từ gold.jsonl."""
    vcfg = cfg["verifier"]
    setting = vcfg["setting"]
    root = cfg["paths"]["data_root"]
    inputs, packages = [], []
    if setting == "S-oracle":
        locs = {cid: OracleEvidenceLocator.from_gold(g) for cid, g in load_gold(work_dir(cfg) / "gold.jsonl").items()}
        tables = _tables_by_paper(cfg, cond) if cond != "IMG" else {}
        for c in claims:
            paper = load_paper(root, f"papers/{c.paper_id}.json")
            inputs.append(build_oracle(c, locs[c.claim_id], paper, cond, tables.get(c.paper_id, {}), root,
                                       unified=vcfg.get("unified", False)))
        return inputs, packages
    method = vcfg.get("method", "hybrid")
    tables = _tables_by_paper(cfg, cond)
    units = read_jsonl(units_path(cfg, cond), EvidenceUnit)
    by_paper: dict[str, list[EvidenceUnit]] = {}
    for u in units:
        by_paper.setdefault(u.paper_id, []).append(u)
    qvecs = embedder.encode([c.text for c in claims]) if embedder is not None and method != "bm25" else None
    indexes: dict[str, PaperIndex] = {}
    for i, c in enumerate(claims):
        if c.paper_id not in indexes:
            indexes[c.paper_id] = PaperIndex.build(c.paper_id, cond, by_paper.get(c.paper_id, []),
                                                   embedder if method != "bm25" else None)
        paper = load_paper(root, f"papers/{c.paper_id}.json")
        sections = {str(s["section_id"]): s.get("text") or "" for s in paper["sections"]}
        r = run_claim(c, indexes[c.paper_id], tables.get(c.paper_id, {}), sections, setting, method, cfg,
                      count, llm, qvecs[i] if qvecs is not None else None, root)
        inputs.append(r.verifier_input)
        if r.package is not None:
            packages.append(r.package)
    return inputs, packages


def run_verify(cfg: dict, args, llm=None, count: Callable[[str], int] | None = None, embedder=None,
               load_image=preprocess_image) -> int:
    vcfg = cfg["verifier"]
    claims = read_jsonl(work_dir(cfg) / "claims_input.jsonl", ClaimInput)
    if cfg.get("limit"):
        claims = claims[: cfg["limit"]]
    if llm is None:  # pragma: no cover - GPU
        from sciverx.verifier.run_vllm import VllmClient
        llm = VllmClient(vcfg["model"], vcfg.get("max_model_len", 32768), vcfg.get("limit_mm_per_prompt"))
    if count is None:  # pragma: no cover
        from sciverx.m3_retrieve.stage import qwen_token_counter
        count = qwen_token_counter(vcfg["model"])
    if embedder is None and vcfg["setting"] != "S-oracle" and vcfg.get("method", "hybrid") != "bm25":  # pragma: no cover
        from sciverx.m3_retrieve.rankers import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder(cfg["m1"]["units"]["embed_model"])
    names = vcfg.get("sample_names")
    samples = [s for s in vcfg["samples"] if not names or s["name"] in names]
    exp = cfg.get("experiment", "verify")
    run_id = make_run_id(exp, cfg["split"], cfg)
    run_dir = Path(open_run_dir(cfg["paths"]["outputs"], build_manifest(run_id, cfg, [c.claim_id for c in claims])))
    system = vcfg.get("system", "P0")
    for cond in vcfg["table_conditions"]:
        inputs, packages = build_all_inputs(cfg, claims, cond, llm, count, embedder)
        write_jsonl(run_dir / cond / "verifier_requests.jsonl", [b.request for b in inputs])
        if packages:
            write_jsonl(run_dir / cond / "packages.jsonl", packages)
        outs = run_samples(llm, inputs, samples, vcfg.get("max_tokens", 10240), run_id=run_id, system=system,
                           setting=vcfg["setting"], table_condition=cond, load_image=load_image)
        write_jsonl(run_dir / cond / "verifier_outputs.jsonl", outs)
        print(f"[verify] {vcfg['setting']}/{cond}: {len(inputs)} claim × {len(samples)} lượt")
    print(f"[verify] → {run_dir}")
    return 0
