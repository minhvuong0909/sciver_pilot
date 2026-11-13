"""SPEC 12.4 test_budget: cùng ngân sách cho mọi phương pháp; ghi đúng dropped_parents."""

import pytest

from sciverx.m1_prepare.units import build_units, whitespace_counter as wc
from sciverx.m3_retrieve.budget import select_evidence
from sciverx.m3_retrieve.retrieve import PaperIndex, retrieve
from sciverx.schemas import ClaimInput, TableRecord
from sciverx.tables.html_cells import parse_html_table

from test_m3_retrieve import BagEmbedder


def _table(tid, n_rows, caption):
    html = "<table><thead><tr><th>Model</th><th>F1</th></tr></thead>" + "".join(
        f"<tr><td>m{r}</td><td>{90 + r / 10:.1f}</td></tr>" for r in range(n_rows)) + "</table>"
    cells, nh = parse_html_table(html)
    return TableRecord(table_record_id=f"p::{tid}::G1", paper_id="p", table_id=tid, condition="G1",
                       extractor="x", caption=caption, cells=cells, n_header_rows=nh, status="ok")


PAPER = {"sections": [
    {"section_id": "1", "section_name": "Intro", "text": "We evaluate F1 of models."},
    {"section_id": "4", "section_name": "Results", "text": "Model m3 reaches 90.3 F1. " * 3}]}
TABLES = {"1": _table("1", 4, "Table 1: F1 of models."), "2": _table("2", 40, "Table 2: Big F1 table.")}
CLAIM = ClaimInput(claim_id="validation-1", split="validation", paper_id="p", text="m3 reaches 90.3 F1")


def _index(limit=40):
    units, _ = build_units("p", PAPER, list(TABLES.values()), limit=limit, count=wc)
    return PaperIndex.build("p", "G1", units, BagEmbedder())


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_same_budget_for_every_method(method):
    idx = _index()
    res = retrieve(CLAIM, idx, method, {"k_units": 50}, BagEmbedder().encode([CLAIM.text])[0])
    cfg = {"max_evidence_tokens": 60, "max_parents": 3, "max_images": 2}
    sel = select_evidence(res, idx, TABLES, cfg, wc)
    assert sel.budget["tokens_used"] <= 60
    assert sel.budget["tokens_used"] == sum(wc(p.text) for p in sel.parents)
    assert len(sel.parents) <= 3
    assert {k: sel.budget[k] for k in ("max_evidence_tokens", "max_parents", "max_images")} == cfg


def test_dropped_parents_recorded_when_tokens_run_out():
    idx = _index()
    res = retrieve(CLAIM, idx, "bm25", {"k_units": 50})
    first = res.candidates[0].parent_id
    big = 10_000
    sel_all = select_evidence(res, idx, TABLES, {"max_evidence_tokens": big, "max_parents": 3}, wc)
    assert sel_all.budget["dropped_parents"] == [] and len(sel_all.parents) == 3
    need_first = wc(sel_all.parents[0].text)
    sel = select_evidence(res, idx, TABLES, {"max_evidence_tokens": need_first, "max_parents": 3}, wc)
    assert [p.parent_id for p in sel.parents] == [first]
    assert sel.budget["dropped_parents"] == [c.parent_id for c in res.candidates[1:3]]


def test_oversized_table_truncated_to_best_rows_with_header():
    idx = _index()
    res = retrieve(CLAIM, idx, "bm25", {"k_units": 50})
    res.candidates = [c for c in res.candidates if c.parent_id == "table:2"]
    sel = select_evidence(res, idx, TABLES, {"max_evidence_tokens": 50, "max_parents": 3}, wc)
    p = sel.parents[0]
    assert p.truncated and sel.budget["truncated"] and p.rows
    lines = p.text.splitlines()
    assert lines[:2] == ["Table 2: Big F1 table.", "Model | F1"]
    assert wc(p.text) <= 50 and len(lines) - 2 == len(p.rows) < 40
