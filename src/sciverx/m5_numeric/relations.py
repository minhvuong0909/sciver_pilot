"""Dựng NumericCheck từ claim và bằng chứng đã căn chỉnh (SPEC 7-M5).

Nguyên tắc:
- claim_operands (số NÊU trong claim) tách khỏi evidence_operands (số LẤY từ
  bằng chứng); giá trị trong claim không bao giờ dùng để xác nhận chính nó.
- Một claim có thể có nhiều NumericCheck.
- percentage_point_diff ≠ relative_change (80% → 90%: +10 điểm %, +12,5% tương đối).
  Khi câu chữ không phân biệt được, ghi missing_information thay vì đoán.
- extremum / universal / comparison có comparison_set và comparison_set_status;
  chỉ `complete` khi đã bao đủ mọi hàng của cột trong bảng (bảng không bị cắt).
- ratio_multiple xác định quantity_kind (thời gian thì tỷ số đảo chiều).
Luật Python trước; LLM chỉ khi luật không quyết được (chưa bật ở đây).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sciverx.m5_numeric.parse import Quantity, cell_number, quantities, time_spec
from sciverx.m5_numeric.tolerance import tolerance_rule
from sciverx.schemas import AlignedEvidence, CellRef, NumericCheck, Operand, TableRecord
from sciverx.tables.grid import build_grid, merged_column_headers, row_header

_LOWER_BETTER = re.compile(r"\b(error|loss|latency|time|runtime|perplexity|ppl|wer|cer|mae|mse|rmse|"
                           r"fid|memory|cost|ms|seconds?)\b", re.I)
_HIGHER_BETTER = re.compile(r"\b(acc|accuracy|f1|bleu|rouge|precision|recall|auc|map|em|score|"
                            r"throughput|speed|fps|success)\b", re.I)
_THROUGHPUT = re.compile(r"\b(throughput|tokens?/s|samples?/s|fps|qps|speed)\b", re.I)
_TIME = re.compile(r"\b(latency|time|runtime|ms|seconds?|hours?|minutes?)\b", re.I)

_EXTREMUM = re.compile(r"\b(best|highest|lowest|largest|smallest|top|state-of-the-art|sota|"
                       r"outperforms? all|maximum|minimum)\b", re.I)
_UNIVERSAL = re.compile(r"\b(all|every|each|consistently|across (?:all )?(?:the )?\w+)\b", re.I)
_COMPARE = re.compile(r"\b(outperforms?|better than|worse than|higher than|lower than|exceeds?|surpass(?:es)?|"
                      r"beats?|than)\b", re.I)
_CHANGE = re.compile(r"\b(improv\w*|increas\w*|decreas\w*|reduc\w*|drops?|gains?|boost\w*|declin\w*|"
                     r"rais\w*|lower\w*|higher)\b", re.I)
_RELATIVE = re.compile(r"\b(relative(?:ly)?|relative improvement)\b", re.I)
_RATIO = re.compile(r"(\d+(?:\.\d+)?)\s*(?:×|x\b|times\b|-?fold\b)", re.I)
_AVERAGE = re.compile(r"\b(average|mean|on average)\b", re.I)
_SUM = re.compile(r"\b(total|sum|combined|overall number)\b", re.I)


@dataclass
class TableContext:
    record: TableRecord
    rows_included: list[int] | None = None   # None = cả bảng có trong bằng chứng


@dataclass
class NumericResult:
    checks: list[NumericCheck] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)


def _claim_op(q: Quantity) -> Operand:
    return Operand(source="claim", ref=None, value_text=q.text, value=str(q.value), unit=q.unit, scale=q.scale)


def _ev_op(c: CellRef) -> Operand | None:
    text = c.text_norm or c.text_raw
    v = cell_number(text)
    if v is None:
        return None
    unit = "%" if "%" in text or "%" in c.col_header else None
    return Operand(source="evidence_cell", ref=c, value_text=text, value=str(v), unit=unit)


def direction_of(header: str) -> str:
    if "(lower is better)" in header or "↓" in header:
        return "lower_better"
    if "(higher is better)" in header or "↑" in header:
        return "higher_better"
    if _LOWER_BETTER.search(header):
        return "lower_better"
    if _HIGHER_BETTER.search(header):
        return "higher_better"
    return "unknown"


def quantity_kind_of(text: str) -> str:
    if _THROUGHPUT.search(text):
        return "throughput"
    if _TIME.search(text):
        return "time"
    return "unknown"


def column_set(ctx: TableContext, col_cell: CellRef) -> tuple[list[CellRef], str]:
    """Mọi ô thân bảng cùng cột với `col_cell`; complete chỉ khi cả bảng có trong bằng chứng."""
    rec = ctx.record
    grid = build_grid(rec.cells)
    target = next(c for c in rec.cells if c.cell_id == col_cell.cell_id)
    heads = merged_column_headers(grid, rec.n_header_rows)
    rows = range(rec.n_header_rows, grid.n_rows)
    out, seen = [], set()
    for r in rows:
        cell = grid.cell_at(r, target.col)
        if cell is None or cell.cell_id in seen or cell.is_header:
            continue
        seen.add(cell.cell_id)
        out.append(CellRef(table_record_id=rec.table_record_id, condition=rec.condition, cell_id=cell.cell_id,
                           row_header=row_header(grid, r), col_header=heads[target.col],
                           text_raw=cell.text_raw, text_norm=cell.text_norm))
    complete = ctx.rows_included is None or set(rows) <= set(ctx.rows_included)
    return out, "complete" if complete else "partial"


def build_checks(claim_text: str, evidence: list[AlignedEvidence],
                 tables: dict[str, TableContext] | None = None) -> NumericResult:
    """tables: evidence_id ("table:1") → TableContext, cần cho tập so sánh."""
    tables = tables or {}
    res = NumericResult()
    qs = quantities(claim_text)
    cells = [c for e in evidence for c in e.cells]
    ev_ops = [op for op in (_ev_op(c) for c in cells) if op is not None]
    header_text = " ".join({c.col_header for c in cells})
    direction = direction_of(header_text) if cells else None
    tspec = time_spec(claim_text)
    k = 0

    def add(relation: str, claim_ops: list[Operand], ev: list[Operand], **kw) -> NumericCheck:
        nonlocal k
        k += 1
        q0 = next((q for q in qs if str(q.value) == (claim_ops[0].value if claim_ops else None)), None)
        qual = q0.qualifier if q0 else "none"
        prec = q0.precision_decimals if q0 else None
        chk = NumericCheck(check_id=f"c{k}", relation=relation, claim_operands=claim_ops,
                           evidence_operands=ev, time=tspec, direction=direction, qualifier=qual,
                           precision_decimals=prec, tolerance_rule=tolerance_rule(qual, prec, relation),
                           decided_by="rule", unit=(claim_ops[0].unit if claim_ops else None),
                           comparison_set_status=kw.pop("comparison_set_status", "not_applicable"), **kw)
        res.checks.append(chk)
        return chk

    ratio = _RATIO.search(claim_text)
    pct_change = [q for q in qs if q.unit in ("%", "pp")] if _CHANGE.search(claim_text) else []
    used: set[int] = set()

    if ratio:
        q = next(q for q in qs if q.unit == "x")
        used.add(id(q))
        kind = quantity_kind_of(claim_text + " " + header_text)
        add("ratio_multiple", [_claim_op(q)], ev_ops, quantity_kind=kind)
        if kind == "unknown":
            res.missing_information.append("ratio_multiple: không rõ đại lượng là thời gian hay tốc độ")
    for q in pct_change:
        if id(q) in used:
            continue
        used.add(id(q))
        if q.unit == "pp":
            add("percentage_point_diff", [_claim_op(q)], ev_ops)
        elif _RELATIVE.search(claim_text):
            add("relative_change", [_claim_op(q)], ev_ops)
        else:
            chk = add("relative_change", [_claim_op(q)], ev_ops)
            res.missing_information.append(
                f"{chk.check_id}: claim ghi '{q.text}' nhưng không rõ là điểm % hay % tương đối")
    if _EXTREMUM.search(claim_text) or _UNIVERSAL.search(claim_text) or (
            _COMPARE.search(claim_text) and not pct_change and not ratio):
        rel = ("extremum" if _EXTREMUM.search(claim_text)
               else "universal" if _UNIVERSAL.search(claim_text) else "comparison")
        comp, status = [], "unknown"
        tcells = [(e, c) for e in evidence if e.type == "table" for c in e.cells]
        if tcells and tcells[0][0].evidence_id in tables:
            comp, status = column_set(tables[tcells[0][0].evidence_id], tcells[0][1])
        elif rel == "comparison" and len(ev_ops) >= 2:
            comp, status = [op.ref for op in ev_ops if isinstance(op.ref, CellRef)], "complete"
        if status != "complete":
            res.missing_information.append(f"{rel}: tập so sánh chưa đủ ({status})")
        add(rel, [], ev_ops, comparison_set=comp, comparison_set_status=status)
    if _AVERAGE.search(claim_text) or _SUM.search(claim_text):
        q = next((q for q in qs if id(q) not in used), None)
        if q is not None:
            used.add(id(q))
            add("average" if _AVERAGE.search(claim_text) else "sum", [_claim_op(q)], ev_ops)
    for q in qs:  # số còn lại: so giá trị trực tiếp
        if id(q) in used:
            continue
        add("percent_value" if q.unit == "%" else "value_equals", [_claim_op(q)], ev_ops)
    return res
