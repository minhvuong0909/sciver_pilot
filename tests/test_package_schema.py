"""SPEC 12.4 test_package_schema: mọi package hợp lệ; mọi ref tồn tại; parse_error khác not_found."""

import json
from pathlib import Path

from sciverx.llm import ScriptedLLM
from sciverx.m4_align.align import AlignResult, Candidate, align_claim
from sciverx.m5_numeric.relations import TableContext, build_checks
from sciverx.m6_package.package import build_package, evidence_text, numeric_text, package_report, ref_errors
from sciverx.schemas import ClaimInput, RetrievalCandidate, RetrievalResult, TableRecord
from sciverx.tables.html_cells import parse_html_table

SECTION = "We train on SQuAD. Ours reaches 92.4 F1! BERT is lower."
CLAIM = ClaimInput(claim_id="validation-1", split="validation", paper_id="p",
                   text="Ours improves F1 by 1.5 points over BERT.")


def _table():
    cells, nh = parse_html_table("<table><thead><tr><th>Model</th><th>F1</th></tr></thead>"
                                 "<tr><td>BERT</td><td>90.9</td></tr><tr><td>Ours</td><td>92.4</td></tr></table>")
    return TableRecord(table_record_id="p::1::G1", paper_id="p", table_id="1", condition="G1",
                       extractor="x", caption="Table 1: QA.", cells=cells, n_header_rows=nh, status="ok")


TABLE = _table()
RETR = RetrievalResult(claim_id=CLAIM.claim_id, method="hybrid", table_condition="G1", params={},
                       n_units_in_pool=4, n_parents_in_pool=2, pool_hash="h",
                       candidates=[RetrievalCandidate(parent_id="table:1", best_unit_id="u", kind="table_full",
                                                      final_rank=1)])
PROV = {"model": "mock", "run_id": "r", "config_hash": "c"}


def _package(setting="S-align45"):
    cands = [Candidate("table:1", 1, table=TABLE),
             Candidate("section:4.1", 2, section_text=SECTION, unit_id="p::paragraph::4.1::0")]
    ans = json.dumps({"status": "matched", "evidence": [
        {"evidence_id": "table:1", "cell_ids": ["r1c1", "r2c1"]},
        {"evidence_id": "section:4.1", "sentence_ids": ["s1"]}]})
    al = align_claim(CLAIM, cands, ScriptedLLM(script=[ans]))
    num = build_checks(CLAIM.text, al.evidence, {"table:1": TableContext(TABLE)})
    return build_package(CLAIM, setting, "G1", RETR, al, num.checks, {"tokens_used": 50}, PROV,
                         num.missing_information)


def test_package_valid_and_traceable():
    pkg = _package()
    assert pkg.alignment_status == "matched" and pkg.numeric_checks[0].relation == "percentage_point_diff"
    assert ref_errors(pkg, {TABLE.table_record_id: TABLE}, {"4.1": SECTION}) == []
    rep = package_report([json.loads(pkg.model_dump_json())], {TABLE.table_record_id: TABLE},
                         {"p": {"4.1": SECTION}})
    assert rep["schema_valid_pct"] == 100.0 and rep["traceable_pct"] == 100.0
    assert pkg.provenance["align_prompt_id"] == "align_v1"


def test_s_align4_has_no_numeric_checks():
    assert _package("S-align4").numeric_checks == []


def test_broken_refs_and_invalid_schema_reported():
    pkg = _package()
    bad_table = TABLE.model_copy(update={"cells": [c for c in TABLE.cells if c.cell_id != "r2c1"]})
    assert any("r2c1" in e for e in ref_errors(pkg, {TABLE.table_record_id: bad_table}, {"4.1": SECTION}))
    assert ref_errors(pkg, {TABLE.table_record_id: TABLE}, {"4.1": "different text"})
    rep = package_report([{"claim_id": "x", "alignment_status": "weird"}], {}, {})
    assert rep["schema_valid_pct"] == 0.0 and rep["problems"]


def test_parse_error_package_differs_from_not_found():
    al_pe = AlignResult(claim_id=CLAIM.claim_id, status="parse_error", errors=["bad json"])
    al_nf = AlignResult(claim_id=CLAIM.claim_id, status="not_found", missing=["nothing relevant"])
    pe = build_package(CLAIM, "S-align4", "G1", RETR, al_pe, [], {}, PROV)
    nf = build_package(CLAIM, "S-align4", "G1", RETR, al_nf, [], {}, PROV)
    assert pe.alignment_status == "parse_error" and pe.errors == ["bad json"]
    assert nf.alignment_status == "not_found" and nf.errors == []
    assert "(no aligned evidence found)" in evidence_text(nf, {"1": TABLE})


def test_verifier_text_has_cells_but_no_results():
    pkg = _package()
    ev = evidence_text(pkg, {"1": TABLE})
    assert "[Table 1] selected cells: BERT / F1 = 90.9; Ours / F1 = 92.4" in ev
    assert "[Section 4.1] Ours reaches 92.4 F1!" in ev
    nt = numeric_text(pkg)
    assert "percentage_point_diff" in nt and "1.5" in nt
    assert "computed" not in nt and "1.5 =" not in nt


def test_mock_fixture_packages_still_valid():
    fix = Path(__file__).parent / "fixtures" / "packages_mock.jsonl"
    rows = [json.loads(x) for x in fix.read_text(encoding="utf-8").splitlines()]
    rep = package_report(rows, {}, {})
    assert rep["schema_valid_pct"] == 100.0
