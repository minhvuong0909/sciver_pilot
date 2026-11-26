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

import time
from collections import Counter
from pathlib import Path
from typing import Callable

from sciverx.data.loader import load_paper
from sciverx.evaluation.gold_access import load_gold
from sciverx.m1_prepare import tables_path, units_path, work_dir
from sciverx.m3_retrieve.retrieve import PaperIndex
from sciverx.pipeline import run_claim
from sciverx.schemas import ClaimInput, EvidenceUnit, OracleEvidenceLocator, TableRecord, VerifierOutput
from sciverx.utils.io import read_jsonl, write_jsonl
from sciverx.utils.manifest import build_manifest, find_or_make_run_id, open_run_dir, update_manifest
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
        llm = VllmClient(vcfg["model"], vcfg.get("max_model_len", 32768), vcfg.get("limit_mm_per_prompt"),
                         revision=vcfg.get("revision"))
    if count is None:  # pragma: no cover
        from sciverx.m3_retrieve.stage import qwen_token_counter
        count = qwen_token_counter(vcfg["model"])
    if embedder is None and vcfg["setting"] != "S-oracle" and vcfg.get("method", "hybrid") != "bm25":  # pragma: no cover
        from sciverx.m3_retrieve.rankers import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder(cfg["m1"]["units"]["embed_model"])
    names = vcfg.get("sample_names")
    samples = [s for s in vcfg["samples"] if not names or s["name"] in names]
    exp = cfg.get("experiment", "verify")
    run_id = find_or_make_run_id(cfg["paths"]["outputs"], exp, cfg["split"], cfg)
    run_dir = Path(open_run_dir(cfg["paths"]["outputs"], build_manifest(run_id, cfg, [c.claim_id for c in claims])))
    system = vcfg.get("system", "P0")
    started = time.monotonic()
    counts: Counter = Counter()
    for cond in vcfg["table_conditions"]:
        inputs, packages = build_all_inputs(cfg, claims, cond, llm, count, embedder)
        write_jsonl(run_dir / cond / "verifier_requests.jsonl", [b.request for b in inputs])
        if packages:
            write_jsonl(run_dir / cond / "packages.jsonl", packages)
        out_path = run_dir / cond / "verifier_outputs.jsonl"
        prev = load_previous_outputs(out_path)
        done = {k: o.request_hash for k, o in prev.items() if o.status != "error"}
        new = run_samples(llm, inputs, samples, vcfg.get("max_tokens", 10240), run_id=run_id, system=system,
                          setting=vcfg["setting"], table_condition=cond, load_image=load_image,
                          max_retries=vcfg.get("max_retries", 1), done=done,
                          chunk_size=vcfg.get("chunk_size", 64),
                          on_chunk=lambda chunk, p=out_path: append_jsonl(p, chunk))
        # Ghi lại bản gọn: mỗi (claim, lượt) một dòng, bản mới thắng bản cũ, theo thứ tự input × lượt
        merged = {**prev, **{(o.claim_id, o.sample_name): o for o in new}}
        order = [(b.request.claim_id, s["name"]) for s in samples for b in inputs]
        outs = [merged[k] for k in order if k in merged]
        write_jsonl(out_path, outs)
        counts.update(error_counts(outs))
        print(f"[verify] {vcfg['setting']}/{cond}: {len(inputs)} claim × {len(samples)} lượt "
              f"(mới {len(new)}, dùng lại {len(outs) - len(new)})")
    update_manifest(run_dir, model=vcfg["model"], model_revision=vcfg.get("revision"),
                    seeds=[s["seed"] for s in samples], duration_s=round(time.monotonic() - started, 1),
                    error_counts=dict(counts), runtime=getattr(llm, "runtime", {}))
    print(f"[verify] → {run_dir}  (đếm: {dict(counts)})")
    return 0


def append_jsonl(path: Path, records: list) -> None:
    """Ghi nối ngay sau mỗi lô (flush + fsync) để GPU ngắt giữa chừng không mất lô đã xong."""
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_previous_outputs(path: Path) -> dict[tuple[str, str], VerifierOutput]:
    """Đầu ra đã có từ lần chạy trước (cùng run_id): (claim_id, sample_name) → bản ghi
    cuối cùng. Dòng hỏng ở cuối file (ghi dở khi máy tắt) bị bỏ qua."""
    out: dict[tuple[str, str], VerifierOutput] = {}
    if not path.exists():
        return out
    import json
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            o = VerifierOutput.model_validate(json.loads(line))
        except Exception:  # noqa: BLE001 - dòng ghi dở
            continue
        out[(o.claim_id, o.sample_name)] = o
    return out


def error_counts(outs: list) -> dict[str, int]:
    """Số đầu ra theo status, số bị cắt (finish_reason = length) và tổng số lần retry."""
    c = Counter(o.status for o in outs)
    c["truncated"] = sum(o.finish_reason == "length" for o in outs)
    c["retries"] = sum(o.attempts - 1 for o in outs)
    c["total"] = len(outs)
    return dict(c)
