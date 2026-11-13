"""SPEC 12.4 test_units_limit và test_pool_parity. Dùng bộ đếm theo khoảng trắng
(số từ + 2) thay cho tokenizer của model embedding."""

import pytest

from sciverx.m1_prepare.units import build_units, table_units, whitespace_counter as wc
from sciverx.schemas import TableRecord
from sciverx.tables.html_cells import parse_html_table


def _table(cond, n_rows=3, n_cols=3, status="ok", caption="Table 1: Results."):
    html = "<table><thead><tr>" + "".join(f"<th>h{c}</th>" for c in range(n_cols)) + "</tr></thead>"
    html += "".join("<tr>" + "".join(f"<td>r{r}c{c}</td>" for c in range(n_cols)) + "</tr>"
                    for r in range(n_rows)) + "</table>"
    cells, nh = parse_html_table(html)
    if status != "ok":
        cells, nh = [], 0
    return TableRecord(table_record_id=f"p::1::{cond}", paper_id="p", table_id="1", condition=cond,
                       extractor="x", caption=caption, cells=cells, n_header_rows=nh, status=status)


PAPER = {"sections": [
    {"section_id": "1", "section_name": "Intro", "text": "Short para.\n\nSecond para here."},
    {"section_id": "2", "section_name": "Long",
     "text": "One two three four five. Six seven eight nine ten. Eleven twelve thirteen fourteen fifteen."},
    {"section_id": "3", "section_name": "Empty", "text": ""}],
    "image_paths": {"5": {"caption": "Figure 1: curve"}}}


def test_paragraph_offsets_and_split_by_sentence():
    units, log = build_units("p", PAPER, [], limit=10, count=wc)
    paras = [u for u in units if u.kind == "paragraph"]
    for u in paras:
        sec = next(s for s in PAPER["sections"] if s["section_id"] == u.parent_id)
        assert u.text.endswith(sec["text"][u.char_start:u.char_end])
        assert u.text.startswith(sec["section_name"] + ": ")
    long_parts = [u for u in paras if u.parent_id == "2"]
    assert len(long_parts) == 3 and long_parts[0].text == "Long: One two three four five."
    assert [u.unit_id for u in paras][:2] == ["p::paragraph::1::0", "p::paragraph::1::1"]


def test_sentence_longer_than_limit_split_by_words_and_logged():
    paper = {"sections": [{"section_id": "1", "section_name": "S", "text": " ".join(f"w{i}" for i in range(20))}]}
    units, log = build_units("p", paper, [], limit=8, count=wc)
    assert all(u.n_wordpieces <= 8 for u in units) and len(units) > 1
    assert log and log[0]["action"] == "split_sentence_by_words"
    assert " ".join(u.text.split(": ", 1)[1] for u in units) == paper["sections"][0]["text"]


def test_table_full_when_it_fits():
    units = table_units(_table("G1"), limit=100, count=wc, log=[])
    assert [u.kind for u in units] == ["table_full"]
    assert units[0].text.splitlines()[:2] == ["Table 1: Results.", "h0 | h1 | h2"]


def test_table_rows_repeat_caption_and_header():
    units = table_units(_table("G1", n_rows=6), limit=15, count=wc, log=[])
    assert len(units) > 1 and all(u.kind == "table_rows" for u in units)
    for u in units:
        lines = u.text.splitlines()
        assert lines[:2] == ["Table 1: Results.", "h0 | h1 | h2"] and u.n_wordpieces <= 15
    covered = [r for u in units for r in range(u.row_range[0], u.row_range[1] + 1)]
    assert covered == list(range(1, 7))


def test_wide_row_split_by_columns_keeps_row_header():
    log = []
    units = table_units(_table("G1", n_rows=1, n_cols=12, caption="T"), limit=14, count=wc, log=log)
    assert len(units) > 1 and log[0]["action"] == "split_row_by_columns"
    assert all(u.text.splitlines()[2].startswith("r0c0") for u in units)
    assert all(u.n_wordpieces <= 14 for u in units)


def test_empty_table_gets_caption_only_unit():
    units = table_units(_table("E1", status="no_table_found"), limit=50, count=wc, log=[])
    assert len(units) == 1 and units[0].kind == "table_rows" and units[0].text == "Table 1: Results."
    assert units[0].unit_id == "p::table_rows::1::E1-caption"


def test_units_limit_raises_if_violated():
    with pytest.raises(ValueError):
        build_units("p", {"sections": []}, [_table("G1", caption=" ".join(["x"] * 50))], limit=10, count=wc)


def test_pool_parity_same_parents_and_paragraphs():
    g, _ = build_units("p", PAPER, [_table("G1", n_rows=6)], limit=15, count=wc)
    e, _ = build_units("p", PAPER, [_table("E1", status="no_table_found")], limit=15, count=wc)
    assert {u.parent_id for u in g} == {u.parent_id for u in e}
    assert [u for u in g if u.kind == "paragraph"] == [u for u in e if u.kind == "paragraph"]
    assert all(u.n_wordpieces <= 15 for u in g + e)


def test_chart_captions_only_when_mixed():
    a, _ = build_units("p", PAPER, [], limit=50, count=wc)
    b, _ = build_units("p", PAPER, [], limit=50, count=wc, allow_mixed=True)
    assert not any(u.kind == "chart_caption" for u in a)
    assert [u.parent_id for u in b if u.kind == "chart_caption"] == ["5"]
