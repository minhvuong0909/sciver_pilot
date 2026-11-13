"""M6 — đóng gói EvidencePackage (SPEC 7-M6).

Ghép kết quả M3 (truy xuất, ngân sách), M4 (căn chỉnh) và M5 (số liệu) thành
một bản ghi kiểm định được và truy được về nguồn. Dựng văn bản bằng chứng cho
verifier; numeric_checks được mô tả (toán hạng, nguồn, phép tính), KHÔNG kèm
kết quả hay kết luận. Báo: % hợp lệ schema, % truy được về nguồn, phân bố
alignment_status.
"""

from __future__ import annotations

from collections import Counter

from pydantic import ValidationError

from sciverx.m4_align.align import AlignResult
from sciverx.m5_numeric.calculator import describe
from sciverx.schemas import (AlignedEvidence, CellRef, ClaimInput, EvidencePackage, NumericCheck,
                             RetrievalResult, SpanRef, TableRecord)
from sciverx.tables.linearize import TEXT_LABEL as TEXT_TABLE_LABEL, linearize_table


def build_package(claim: ClaimInput, setting: str, table_condition: str, retrieval: RetrievalResult,
                  align: AlignResult, checks: list[NumericCheck], budget: dict, provenance: dict,
                  numeric_missing: list[str] | None = None) -> EvidencePackage:
    if setting == "S-align4":
        checks = []  # M5 chỉ có ở S-align45
    missing = list(align.missing) + (list(numeric_missing or []) if setting == "S-align45" else [])
    if align.truncated:
        missing.append(f"M4 bỏ ứng viên vì ngân sách: {', '.join(align.truncated)}")
    return EvidencePackage(
        claim_id=claim.claim_id, claim_text=claim.text, paper_id=claim.paper_id,
        evidence_setting=setting, table_condition=table_condition, retrieval=retrieval,
        evidence=align.evidence, alignment_status=align.status, alignment_candidates=align.candidates,
        numeric_checks=checks, missing_information=missing, budget=budget, errors=list(align.errors),
        provenance={**provenance, "align_prompt_id": align.prompt_id, "align_prompt_hash": align.prompt_hash})


# ------------------------------------------------------------------ truy về nguồn

def ref_errors(pkg: EvidencePackage, tables: dict[str, TableRecord], sections: dict[str, str]) -> list[str]:
    """Mọi CellRef phải trỏ tới ô có thật (đúng chữ); mọi SpanRef phải khớp chữ trong section.
    tables: table_record_id → TableRecord; sections: section_id → text."""
    errs = []
    ev_all: list[AlignedEvidence] = list(pkg.evidence) + list(pkg.alignment_candidates)
    cells: list[CellRef] = [c for e in ev_all for c in e.cells]
    spans: list[SpanRef] = [s for e in ev_all for s in e.spans]
    for chk in pkg.numeric_checks:
        cells += list(chk.comparison_set)
        for op in chk.evidence_operands:
            if isinstance(op.ref, CellRef):
                cells.append(op.ref)
            elif isinstance(op.ref, SpanRef):
                spans.append(op.ref)
    for c in cells:
        rec = tables.get(c.table_record_id)
        cell = next((x for x in rec.cells if x.cell_id == c.cell_id), None) if rec else None
        if cell is None or cell.text_raw != c.text_raw:
            errs.append(f"CellRef {c.table_record_id}/{c.cell_id}")
    for s in spans:
        text = sections.get(s.section_id)
        if text is None or text[s.char_start:s.char_end] != s.text:
            errs.append(f"SpanRef {s.section_id}[{s.char_start}:{s.char_end}]")
    return errs


def package_report(raw_packages: list[dict], tables: dict[str, TableRecord],
                   sections_by_paper: dict[str, dict[str, str]]) -> dict:
    valid, traceable, statuses, problems = 0, 0, Counter(), []
    for raw in raw_packages:
        try:
            pkg = EvidencePackage.model_validate(raw)
        except ValidationError as exc:
            problems.append({"claim_id": raw.get("claim_id"), "error": str(exc).splitlines()[0]})
            continue
        valid += 1
        statuses[pkg.alignment_status] += 1
        errs = ref_errors(pkg, tables, sections_by_paper.get(pkg.paper_id, {}))
        if errs:
            problems.append({"claim_id": pkg.claim_id, "error": "; ".join(errs[:5])})
        else:
            traceable += 1
    n = len(raw_packages) or 1
    return {"n": len(raw_packages), "schema_valid_pct": round(100 * valid / n, 2),
            "traceable_pct": round(100 * traceable / n, 2),
            "alignment_status": dict(statuses), "problems": problems}


# ------------------------------------------------------------------ văn bản cho verifier

def evidence_text(pkg: EvidencePackage, tables: dict[str, TableRecord], rows_by_table: dict[str, list[int]] | None = None) -> str:
    """Bằng chứng đã căn chỉnh: các ô/câu được chọn, kèm bảng cha theo ngân sách.
    tables: table_id → TableRecord của điều kiện đang chạy."""
    rows_by_table = rows_by_table or {}
    parts = []
    chosen = pkg.evidence or pkg.alignment_candidates  # ambiguous: đưa mọi phương án
    for ev in chosen:
        kind, pid = ev.evidence_id.split(":", 1)
        if ev.type == "table" and pid in tables:
            sel = "; ".join(f"{c.row_header} / {c.col_header} = {c.text_norm or c.text_raw}" for c in ev.cells)
            body = linearize_table(tables[pid], rows=rows_by_table.get(pid))
            parts.append(f"[Table {pid}] selected cells: {sel or '-'}\n{TEXT_TABLE_LABEL}\n{body}")
        else:
            sents = " ".join(s.text for s in ev.spans)
            parts.append(f"[Section {pid}] {sents}")
    if not parts:
        parts.append("(no aligned evidence found)")
    if pkg.missing_information:
        parts.append("Missing information: " + "; ".join(pkg.missing_information))
    return "\n\n".join(parts)


def numeric_text(pkg: EvidencePackage) -> str:
    """Khối "Numeric checks to consider" cho unified_v1_numeric — không kết quả, không kết luận."""
    return describe([c.model_copy(update={"computed": None}) for c in pkg.numeric_checks]) or "(none)"
