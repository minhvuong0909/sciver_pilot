"""Chuỗi xử lý một claim ở các thiết lập truy xuất: M3 → (M4 → M5) → M6 → request verifier.

Đầu vào CHỈ gồm ClaimInput và kho của bài (PaperIndex, TableRecord, section
text). Không có tham số nào nhận GoldInfo — test chống rò rỉ dựa vào điều đó.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sciverx.llm import LLMClient
from sciverx.m3_retrieve.budget import select_evidence
from sciverx.m3_retrieve.retrieve import PaperIndex, retrieve
from sciverx.m4_align.align import AlignResult, Candidate, align_claim
from sciverx.m5_numeric.relations import TableContext, build_checks
from sciverx.m6_package.package import build_package, evidence_text, numeric_text
from sciverx.schemas import ClaimInput, EvidencePackage, RetrievalResult, TableRecord
from sciverx.verifier.build_inputs import BuiltInput, build_aligned, build_retrieved

Counter_ = Callable[[str], int]


@dataclass
class ClaimRun:
    retrieval: RetrievalResult
    selection: dict
    align: AlignResult | None
    package: EvidencePackage | None
    verifier_input: BuiltInput


def run_claim(claim: ClaimInput, index: PaperIndex, tables: dict[str, TableRecord], sections: dict[str, str],
              setting: str, method: str, cfg: dict, count: Counter_, llm: LLMClient | None = None,
              query_vec=None, data_root: str = ".") -> ClaimRun:
    """setting ∈ S-retr | S-align4 | S-align45. tables: table_id → TableRecord (điều kiện đang chạy);
    sections: section_id → text của bài."""
    m3 = {**cfg["m3"], "embed_model": cfg.get("m1", {}).get("units", {}).get("embed_model")}
    res = retrieve(claim, index, method, m3, query_vec)
    sel = select_evidence(res, index, tables, m3["budget"], count,
                          allow_mixed=cfg.get("selection", {}).get("allow_mixed", False))
    parents = [vars(p) for p in sel.parents]
    selection = {"claim_id": claim.claim_id, "parents": parents, "budget": sel.budget}
    if setting == "S-retr":
        return ClaimRun(res, selection, None, None, build_retrieved(claim, parents, data_root))

    if llm is None:
        raise ValueError("S-align cần model cho M4")
    cands = []
    for p in sel.parents:
        kind, pid = p.parent_id.split(":", 1)
        if kind == "table":
            cands.append(Candidate(p.parent_id, p.rank, table=tables[pid], rows=p.rows))
        elif kind == "section":
            cands.append(Candidate(p.parent_id, p.rank, section_text=sections.get(pid, "")))
    m4 = cfg.get("m4", {})
    al = align_claim(claim, cands, llm, m4.get("max_retries", 2), m4.get("max_align_tokens", 12000), count)
    contexts = {p.parent_id: TableContext(tables[p.parent_id.split(":", 1)[1]], p.rows)
                for p in sel.parents if p.kind == "table"}
    num = build_checks(claim.text, al.evidence or al.candidates, contexts)
    prov = {"model": getattr(llm, "model_name", None), "prompt_id": "unified_v1" if setting == "S-align4"
            else "unified_v1_numeric", "run_id": cfg.get("run_id"), "config_hash": cfg.get("config_hash")}
    pkg = build_package(claim, setting, index.condition, res, al, num.checks, sel.budget, prov,
                        num.missing_information)
    rows = {p.parent_id.split(":", 1)[1]: p.rows for p in sel.parents if p.kind == "table" and p.rows}
    ev = evidence_text(pkg, tables, rows)
    vin = build_aligned(claim, ev, numeric_text(pkg) if setting == "S-align45" else None)
    return ClaimRun(res, selection, al, pkg, vin)
