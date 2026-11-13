"""SPEC 12.4 test_normalize_rules: từng luật M1.4, các ca mơ hồ giữ nguyên, idempotent."""

import pytest

from sciverx.schemas import SourceCell, TableRecord
from sciverx.tables.linearize import TEXT_LABEL, linearize_table, verifier_table_block
from sciverx.tables.normalize import (find_footnotes, is_format_dependent, normalize_table,
                                      normalize_text)


def nt(text, header=False, dc=False):
    return normalize_text(text, header, dc)[0]


@pytest.mark.parametrize("raw,expected", [
    ("−0.5", "-0.5"), ("–3", "-3"), ("—12.1", "-12.1"), ("a–b", "a–b"),   # chỉ trước chữ số
    ("–", "N/A"), ("—", "N/A"), ("-", "N/A"), ("n/a", "N/A"), ("N/A", "N/A"),
    ("93.2*", "93.2 [*]"), ("93.2†‡", "93.2 [†‡]"), ("85.3 %*", "85.3% [*]"),
    ("1,234", "1234"), ("12,345.6", "12345.6"), ("1,234*", "1234 [*]"),
    ("93.2±0.1", "93.2 ± 0.1"), ("93.2 +/- 0.1", "93.2 ± 0.1"), ("93.2+-0.1", "93.2 ± 0.1"),
    ("85.3 %", "85.3%"),
])
def test_cell_rules(raw, expected):
    assert nt(raw) == expected


@pytest.mark.parametrize("raw", [
    "1,23",        # không khớp mẫu nghìn đầy đủ
    "12,34,567",   # kiểu Ấn Độ, không khớp
    "93,2",        # dấu phẩy thập phân
    "BERT-base", "top-1", "GPT-4o", "x↑",  # ô thân có mũi tên: giữ nguyên
])
def test_ambiguous_or_non_matching_kept(raw):
    assert nt(raw) == raw


def test_thousands_kept_when_column_uses_decimal_comma():
    text, rules, flags = normalize_text("1,234", False, decimal_comma_col=True)
    assert text == "1,234" and "ambiguous_separator" in flags and not rules


def test_header_arrow_legend_only_in_header():
    assert nt("Acc ↑", header=True) == "Acc ↑ (higher is better)"
    assert nt("Time (s) ↓", header=True) == "Time (s) ↓ (lower is better)"
    assert nt("Acc ↑", header=False) == "Acc ↑"


@pytest.mark.parametrize("raw,header", [
    ("−0.5", False), ("93.2*", False), ("1,234*", False), ("93.2±0.1", False),
    ("85.3 %", False), ("Acc ↑", True), ("–", False), ("1,234", False), ("plain", False),
])
def test_idempotent(raw, header):
    once = nt(raw, header)
    assert nt(once, header) == once


def _record(cells, caption="Table 2: Results. * denotes p < 0.05.", n_header=1, cond="G0"):
    return TableRecord(table_record_id=f"p::T2::{cond}", paper_id="p", table_id="T2",
                       condition=cond, extractor="gold_json", caption=caption,
                       cells=cells, n_header_rows=n_header, status="ok")


def _c(cid, r, c, text, header=False, **kw):
    return SourceCell(cell_id=cid, row=r, col=c, is_header=header, text_raw=text, **kw)


def test_normalize_table_keeps_raw_logs_and_footnotes():
    cells = [_c("r0c0", 0, 0, "Model", True), _c("r0c1", 0, 1, "Acc ↑", True),
             _c("r1c0", 1, 0, "Ours"), _c("r1c1", 1, 1, "93.2*", footnote_marks=["*"], bold=True),
             _c("r2c0", 2, 0, "Base"), _c("r2c1", 2, 1, "−1.0")]
    res = normalize_table(_record(cells))
    rec = res.record
    assert rec.condition == "G1" and rec.table_record_id == "p::T2::G1"
    assert [c.text_raw for c in rec.cells] == [c.text_raw for c in cells]
    assert rec.cells[3].text_norm == "93.2 [*]"
    assert rec.footnotes == {"*": "denotes p < 0.05"}
    changed = {e["cell_id"]: e["rules"] for e in res.log}
    assert changed == {"r0c1": ["header_arrow_legend"], "r1c1": ["footnote_mark"],
                       "r2c1": ["unicode_minus"]}
    assert all(e["table_record_id"] == "p::T2::G1" for e in res.log)


def test_decimal_comma_column_detected_per_column():
    cells = [_c("r0c0", 0, 0, "1,234"), _c("r0c1", 0, 1, "1,234"),
             _c("r1c0", 1, 0, "93,2"), _c("r1c1", 1, 1, "5")]
    res = normalize_table(_record(cells, n_header=0))
    norms = {c.cell_id: c.text_norm for c in res.record.cells}
    assert norms["r0c0"] == "1,234"   # cột 0 có dấu phẩy thập phân → giữ
    assert norms["r0c1"] == "1234"
    assert any("ambiguous_separator" in e["rules"] for e in res.log)


def test_normalize_only_raw_conditions():
    with pytest.raises(ValueError):
        normalize_table(_record([_c("r0c0", 0, 0, "x")], cond="G1"))


def test_find_footnotes():
    assert find_footnotes("Results. † uses extra data. ‡ reproduced.", {"†", "‡"}) == {
        "†": "uses extra data", "‡": "reproduced"}
    assert find_footnotes("No notes here.", {"*"}) == {}


def test_format_dependent_terms():
    terms = ["best", "highest", "outperform", "state-of-the-art", "bold"]
    assert is_format_dependent("Our model achieves the best F1.", terms)
    assert is_format_dependent("It reaches State-of-the-art results.", terms)
    assert not is_format_dependent("A best-effort baseline is used.", terms)
    assert not is_format_dependent("Accuracy is 93.2 on SQuAD.", terms)
    assert not is_format_dependent("It outperforms nothing", ["outperform"])  # ranh giới từ


def test_linearize_merged_headers_and_bold_off_by_default():
    cells = [_c("r0c0", 0, 0, "Model", True, rowspan=2), _c("r0c1", 0, 1, "SQuAD", True, colspan=2),
             _c("r1c1", 1, 1, "EM", True), _c("r1c2", 1, 2, "F1", True),
             _c("r2c0", 2, 0, "Ours"), _c("r2c1", 2, 1, "86.0", bold=True),
             _c("r2c2", 2, 2, "92.4*", footnote_marks=["*"])]
    rec = normalize_table(_record(cells, n_header=2)).record
    text = linearize_table(rec)
    assert text.splitlines() == ["Table 2: Results. * denotes p < 0.05.",
                                 "Model | SQuAD / EM | SQuAD / F1",
                                 "Ours | 86.0 | 92.4 [*]",
                                 "Notes: [*] denotes p < 0.05"]
    assert "**86.0**" in linearize_table(rec, include_bold=True)
    assert "[r2c1] 86.0" in linearize_table(rec, with_ids=True)
    assert verifier_table_block(rec).startswith(TEXT_LABEL + "\n")


def test_linearize_selected_rows_repeats_rowspan():
    cells = [_c("r0c0", 0, 0, "Task", True), _c("r0c1", 0, 1, "Score", True),
             _c("r1c0", 1, 0, "QA", rowspan=2), _c("r1c1", 1, 1, "1"), _c("r2c1", 2, 1, "2")]
    rec = normalize_table(_record(cells, caption="")).record
    assert linearize_table(rec, rows=[2]).splitlines() == ["Task | Score", "QA | 2"]
