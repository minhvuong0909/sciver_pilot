"""Chuẩn hóa bảng G0→G1, E0→E1 — SPEC M1.4. Cùng một hàm cho G và E.

Luôn giữ `text_raw`, ghi `text_norm`. Hàm là idempotent: chuẩn hóa lại kết
quả không đổi gì. Mọi thay đổi ghi log {table_record_id, cell_id, before,
after, rules}.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from sciverx.schemas import SourceCell, TableRecord
from sciverx.tables.html_cells import FOOTNOTE_CHARS

NA_TOKENS = {"-", "–", "—", "‒", "−", "n/a", "N/A", "na", "NA"}
_UNICODE_MINUS = re.compile(r"[−‒–—](?=\d)")
_PM = re.compile(r"\s*(?:±|\+/-|\+-)\s*")
_PCT_SPACE = re.compile(r"(\d)\s+%")
_THOUSANDS = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")
_DECIMAL_COMMA = re.compile(r"^-?\d+,\d+$")
_NUM_WITH_MARKS = re.compile(rf"^(.*\d(?:\s*%)?)\s*([{re.escape(FOOTNOTE_CHARS)}]+)$")
_ARROW_LEGEND = {"↑": "(higher is better)", "↓": "(lower is better)"}


@dataclass
class NormalizeResult:
    record: TableRecord
    log: list[dict] = field(default_factory=list)


def _column_has_decimal_comma(cells: list[SourceCell]) -> set[int]:
    cols = set()
    for c in cells:
        t = c.text_raw.strip()
        if not c.is_header and _DECIMAL_COMMA.match(t) and not _THOUSANDS.match(t):
            cols.add(c.col)
    return cols


def normalize_text(text: str, is_header: bool, decimal_comma_col: bool = False
                   ) -> tuple[str, list[str], list[str]]:
    """Chuẩn hóa một ô. Trả (text_norm, rules_applied, flags)."""
    rules: list[str] = []
    flags: list[str] = []
    t = text.strip()

    if t in NA_TOKENS:
        return ("N/A", ["na_token"] if t != "N/A" else [], flags)

    # Tách ký hiệu chú thích sau số trước, xử lý phần số, rồi gắn lại "x [*]".
    marks = ""
    m = _NUM_WITH_MARKS.match(t)
    if m:
        t, marks = m.group(1).rstrip(), m.group(2)
        rules.append("footnote_mark")

    new = _UNICODE_MINUS.sub("-", t)
    if new != t:
        rules.append("unicode_minus")
        t = new

    new = _PCT_SPACE.sub(r"\1%", t)
    if new != t:
        rules.append("percent_space")
        t = new

    if _PM.search(t):
        new = _PM.sub(" ± ", t).strip()
        if new != t:
            rules.append("plus_minus")
            t = new

    if _THOUSANDS.match(t):
        if decimal_comma_col:
            flags.append("ambiguous_separator")
        else:
            t = t.replace(",", "")
            rules.append("thousands_separator")

    if marks:
        t = f"{t} [{marks}]"

    if is_header:
        for arrow, legend in _ARROW_LEGEND.items():
            if arrow in t and legend not in t:
                t = t.replace(arrow, f"{arrow} {legend}", 1)
                rules.append("header_arrow_legend")
    return t, rules, flags


def find_footnotes(caption: str, marks: set[str]) -> dict[str, str]:
    """Tìm nội dung chú thích trong caption: "* denotes ..." hoặc "†: ...".
    Chỉ trả các ký hiệu có trong `marks`; không tìm thấy thì bỏ qua."""
    out: dict[str, str] = {}
    for mark in sorted(marks, key=len, reverse=True):
        pat = re.compile(rf"(?:^|\s|\(){re.escape(mark)}\s*[:\-]?\s*([^*†‡§¶]+?)(?=(?:\s[{re.escape(FOOTNOTE_CHARS)}])|$|\))")
        m = pat.search(caption)
        if m and m.group(1).strip(" .;"):
            out[mark] = m.group(1).strip(" .;")
    return out


def normalize_table(record: TableRecord) -> NormalizeResult:
    if record.condition not in ("G0", "E0"):
        raise ValueError(f"Chỉ chuẩn hóa G0/E0, nhận {record.condition}")
    out_cond = "G1" if record.condition == "G0" else "E1"
    out_id = TableRecord.make_id(record.paper_id, record.table_id, out_cond)
    dc_cols = _column_has_decimal_comma(record.cells)
    log: list[dict] = []
    new_cells = []
    all_marks: set[str] = set()
    for cell in record.cells:
        norm, rules, flags = normalize_text(cell.text_raw, cell.is_header, cell.col in dc_cols)
        all_marks.update(cell.footnote_marks)
        if norm != cell.text_raw or flags:
            log.append({"table_record_id": out_id, "cell_id": cell.cell_id,
                        "before": cell.text_raw, "after": norm, "rules": rules + flags})
        new_cells.append(cell.model_copy(update={"text_norm": norm}))
    footnotes = dict(record.footnotes)
    for mark, content in find_footnotes(record.caption, all_marks).items():
        footnotes.setdefault(mark, content)
    rec = record.model_copy(update={
        "table_record_id": out_id, "condition": out_cond, "cells": new_cells,
        "footnotes": footnotes, "normalization_log_id": f"{out_id}::normalize"})
    return NormalizeResult(rec, log)


def load_terms(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip().lower() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def is_format_dependent(claim_text: str, terms: list[str]) -> bool:
    """Cờ format_dependent: claim chứa từ chỉ thứ hạng (theo ranh giới từ)."""
    low = claim_text.lower()
    return any(re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", low) for t in terms)
