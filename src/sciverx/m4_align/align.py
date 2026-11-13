"""M4 — căn chỉnh claim với bằng chứng (SPEC 7-M4).

Đầu vào: ClaimInput.text + các cha đã chọn theo ngân sách M3. Bảng in kèm
cell_id; đoạn văn tách thành câu có ID. Gọi model (temperature 0, prompt
align_v1), kiểm định JSON bằng Pydantic, thử lại tối đa max_retries lần, sau
đó `parse_error` (khác `not_found`). Code kiểm: evidence_id thuộc ứng viên,
cell_id có trong TableRecord, câu có thật; row_header/col_header lấy từ bảng.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import BaseModel, ValidationError

from sciverx.llm import LLMClient, LLMRequest
from sciverx.m4_align.prompts import ALIGN_PROMPT_HASH, ALIGN_PROMPT_ID, ALIGN_V1
from sciverx.schemas import AlignedEvidence, CellRef, ClaimInput, SpanRef, TableRecord
from sciverx.tables.grid import build_grid, merged_column_headers, row_header
from sciverx.tables.linearize import linearize_table

# Câu kết thúc ở .!? theo sau là khoảng trắng hoặc hết chuỗi (không cắt "92.4", "e.g.x").
_SENT = re.compile(r"\S.*?(?:[.!?]+(?=\s|$)|$)", re.S)


# ------------------------------------------------------------------ đầu vào

@dataclass
class Candidate:
    """Một cha ứng viên đưa vào M4."""
    evidence_id: str           # "table:1" | "section:4.1"
    rank: int
    table: TableRecord | None = None
    rows: list[int] | None = None          # hàng được đưa vào (bảng bị cắt theo ngân sách)
    section_text: str | None = None
    unit_id: str | None = None

    def sentences(self) -> list[tuple[str, int, int]]:
        """[(sentence_id, char_start, char_end)] trong section_text."""
        out = []
        for m in _SENT.finditer(self.section_text or ""):
            s, e = m.start(), m.end()
            while s < e and self.section_text[s].isspace():
                s += 1
            while e > s and self.section_text[e - 1].isspace():
                e -= 1
            if e > s:
                out.append((f"s{len(out)}", s, e))
        return out

    def render(self) -> str:
        kind, pid = self.evidence_id.split(":", 1)
        if self.table is not None:
            body = linearize_table(self.table, rows=self.rows, with_ids=True)
            return f"<evidence_id: {self.evidence_id}> Table {pid}\n{body}"
        sents = " ".join(f"[{sid}] {self.section_text[a:b]}" for sid, a, b in self.sentences())
        return f"<evidence_id: {self.evidence_id}> Section {pid}\n{sents}"


# ------------------------------------------------------------------ đầu ra của model

class _Item(BaseModel):
    evidence_id: str
    cell_ids: list[str] = []
    sentence_ids: list[str] = []
    reason: str = ""


class AlignResponse(BaseModel):
    status: Literal["matched", "partial", "ambiguous", "not_found"]
    evidence: list[_Item] = []
    alternatives: list[_Item] = []
    missing: list[str] = []


@dataclass
class AlignResult:
    claim_id: str
    status: str
    evidence: list[AlignedEvidence] = field(default_factory=list)
    candidates: list[AlignedEvidence] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    truncated: list[str] = field(default_factory=list)
    prompt_id: str = ALIGN_PROMPT_ID
    prompt_hash: str = ALIGN_PROMPT_HASH
    attempts: int = 0


def extract_json(text: str) -> dict:
    """Lấy object JSON đầu tiên trong câu trả lời (bỏ ```json ... ```)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("không có JSON")
    return json.loads(m.group(0))


# ------------------------------------------------------------------ kiểm ID bằng code

def _cell_ref(rec: TableRecord, cell_id: str) -> CellRef:
    cell = next((c for c in rec.cells if c.cell_id == cell_id), None)
    if cell is None:
        raise ValueError(f"cell_id {cell_id} không có trong {rec.table_record_id}")
    grid = build_grid(rec.cells)
    col_heads = merged_column_headers(grid, rec.n_header_rows)
    return CellRef(table_record_id=rec.table_record_id, condition=rec.condition, cell_id=cell_id,
                   row_header=row_header(grid, cell.row), col_header=col_heads[cell.col],
                   text_raw=cell.text_raw, text_norm=cell.text_norm)


def _to_aligned(item: _Item, cands: dict[str, Candidate], paper_id: str) -> AlignedEvidence:
    if item.evidence_id not in cands:
        raise ValueError(f"evidence_id {item.evidence_id} không thuộc ứng viên")
    cand = cands[item.evidence_id]
    kind, pid = item.evidence_id.split(":", 1)
    if cand.table is not None:
        if item.sentence_ids:
            raise ValueError("bảng không có sentence_ids")
        cells = [_cell_ref(cand.table, cid) for cid in item.cell_ids]
        return AlignedEvidence(evidence_id=item.evidence_id, type="table", cells=cells,
                               from_candidate_rank=cand.rank)
    if item.cell_ids:
        raise ValueError("đoạn văn không có cell_ids")
    sents = {sid: (a, b) for sid, a, b in cand.sentences()}
    spans = []
    for sid in item.sentence_ids:
        if sid not in sents:
            raise ValueError(f"sentence_id {sid} không có trong {item.evidence_id}")
        a, b = sents[sid]
        spans.append(SpanRef(unit_id=cand.unit_id or f"{paper_id}::paragraph::{pid}::0", section_id=pid,
                             char_start=a, char_end=b, text=cand.section_text[a:b]))
    return AlignedEvidence(evidence_id=item.evidence_id, type="paragraph", spans=spans,
                           from_candidate_rank=cand.rank)


def validate_response(resp: AlignResponse, cands: dict[str, Candidate], paper_id: str
                      ) -> tuple[list[AlignedEvidence], list[AlignedEvidence]]:
    ev = [_to_aligned(i, cands, paper_id) for i in resp.evidence]
    alt = [_to_aligned(i, cands, paper_id) for i in resp.alternatives]
    if resp.status in ("matched", "partial") and not ev:
        raise ValueError(f"status {resp.status} nhưng không có evidence")
    return ev, alt


# ------------------------------------------------------------------ chạy

def fit_candidates(cands: list[Candidate], max_tokens: int, count: Callable[[str], int]
                   ) -> tuple[list[Candidate], list[str]]:
    """Giữ ứng viên theo thứ hạng tới max_align_tokens; phần bị bỏ được ghi lại."""
    kept, dropped, used = [], [], 0
    for c in sorted(cands, key=lambda c: c.rank):
        n = count(c.render())
        if used + n > max_tokens:
            dropped.append(c.evidence_id)
            continue
        kept.append(c)
        used += n
    return kept, dropped


def align_claim(claim: ClaimInput, cands: list[Candidate], llm: LLMClient, max_retries: int = 2,
                max_tokens: int = 12000, count: Callable[[str], int] = lambda t: len(t.split()),
                gen_tokens: int = 1024) -> AlignResult:
    res = AlignResult(claim_id=claim.claim_id, status="not_found")
    kept, res.truncated = fit_candidates(cands, max_tokens, count)
    if not kept:
        res.missing.append("không có ứng viên nào trong ngân sách")
        return res
    by_id = {c.evidence_id: c for c in kept}
    prompt = ALIGN_V1.substitute(claim=claim.text, evidence="\n\n".join(c.render() for c in kept))
    for attempt in range(max_retries + 1):
        res.attempts = attempt + 1
        out = llm.generate([LLMRequest(prompt=prompt, temperature=0.0, seed=0, max_tokens=gen_tokens)])[0]
        try:
            parsed = AlignResponse.model_validate(extract_json(out.text))
            ev, alt = validate_response(parsed, by_id, claim.paper_id)
        except (ValueError, ValidationError) as exc:
            res.errors.append(f"attempt {attempt + 1}: {str(exc).splitlines()[0][:200]}")
            continue
        res.status, res.evidence, res.candidates = parsed.status, ev, alt
        res.missing += parsed.missing
        if alt and res.status == "matched":  # nhiều phương án ngang nhau
            res.status = "ambiguous"
        if res.status == "ambiguous" and ev:
            res.candidates = ev + alt
            res.evidence = []
        return res
    res.status = "parse_error"
    return res
