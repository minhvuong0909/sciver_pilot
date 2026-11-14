"""SPEC 12.4 test_m5_relations: 80→90 là +10 điểm % và +12,5% tương đối; "at least 12%"
khác "about 12%"; "2× faster" với thời gian khác với throughput; claim_operands tách
khỏi evidence_operands."""

from decimal import Decimal

import pytest

from sciverx.m5_numeric.calculator import compute, describe, with_computed
from sciverx.m5_numeric.parse import cell_number, quantities, time_spec
from sciverx.m5_numeric.relations import TableContext, build_checks
from sciverx.m5_numeric.tolerance import within_tolerance
from sciverx.schemas import AlignedEvidence, CellRef, TableRecord
from sciverx.tables.html_cells import parse_html_table


def cref(cid, rh, ch, text):
    return CellRef(table_record_id="p::1::G1", condition="G1", cell_id=cid, row_header=rh, col_header=ch,
                   text_raw=text, text_norm=text)


def table_ev(*cells):
    return [AlignedEvidence(evidence_id="table:1", type="table", cells=list(cells), from_candidate_rank=1)]


ACC = table_ev(cref("r1c1", "Baseline", "Acc (%)", "80.0"), cref("r2c1", "Ours", "Acc (%)", "90.0"))


def rel(res):
    return [c.relation for c in res.checks]


def test_percentage_points_vs_relative_change():
    pp = build_checks("Our method improves accuracy by 10 percentage points over the baseline.", ACC)
    assert rel(pp) == ["percentage_point_diff"] and not pp.missing_information
    assert compute(pp.checks[0])["value"] == "10.0"
    r = build_checks("Accuracy increases by 12.5% relative to the baseline.", ACC)
    assert rel(r) == ["relative_change"]
    assert Decimal(compute(r.checks[0])["value"]) == Decimal("12.5")


def test_ambiguous_percent_change_is_flagged_not_guessed():
    r = build_checks("Our method improves accuracy by 10% over the baseline.", ACC)
    assert rel(r) == ["relative_change"]
    assert any("điểm % hay % tương đối" in m for m in r.missing_information)


def test_at_least_differs_from_about():
    a = build_checks("The model reaches at least 12% recall.", [])
    b = build_checks("The model reaches about 12% recall.", [])
    assert a.checks[0].qualifier == "at_least" and b.checks[0].qualifier == "approximately"
    assert a.checks[0].tolerance_rule != b.checks[0].tolerance_rule
    assert within_tolerance(Decimal(12), Decimal("12.9"), "at_least", 0)
    assert not within_tolerance(Decimal(12), Decimal("12.9"), "exact", 0)
    assert within_tolerance(Decimal(12), Decimal("12.9"), "approximately", 0)
    assert not within_tolerance(Decimal(12), Decimal("11.0"), "at_least", 0)


def test_ratio_time_vs_throughput():
    t = build_checks("Our system is 2× faster than the baseline.",
                     table_ev(cref("r1c1", "Baseline", "Latency (ms)", "120"), cref("r2c1", "Ours", "Latency (ms)", "60")))
    s = build_checks("Our system is 2× faster than the baseline.",
                     table_ev(cref("r1c1", "Baseline", "Throughput (tok/s)", "50"), cref("r2c1", "Ours", "Throughput (tok/s)", "100")))
    assert t.checks[0].quantity_kind == "time" and s.checks[0].quantity_kind == "throughput"
    assert compute(t.checks[0])["formula"].endswith("(time: inverse)")
    assert compute(t.checks[0])["value"] == compute(s.checks[0])["value"] == "2"
    u = build_checks("Ours is 3-fold better.", table_ev(cref("r1c1", "A", "Score", "1")))
    assert u.checks[0].relation == "ratio_multiple" and u.checks[0].quantity_kind == "unknown"
    assert u.missing_information


def test_claim_operands_separate_from_evidence():
    r = build_checks("Ours reaches 90.0 accuracy.", ACC)
    c = r.checks[0]
    assert c.relation == "value_equals"
    assert [o.source for o in c.claim_operands] == ["claim"] and c.claim_operands[0].ref is None
    assert all(o.source == "evidence_cell" and o.ref is not None for o in c.evidence_operands)
    assert c.claim_operands[0].value == "90.0"


def test_extremum_comparison_set_complete_vs_partial():
    cells, nh = parse_html_table("<table><thead><tr><th>Model</th><th>BLEU ↑</th></tr></thead>"
                                 "<tr><td>A</td><td>27.1</td></tr><tr><td>B</td><td>28.4</td></tr>"
                                 "<tr><td>Ours</td><td>29.3</td></tr></table>")
    rec = TableRecord(table_record_id="p::1::G1", paper_id="p", table_id="1", condition="G1", extractor="x",
                      caption="", cells=cells, n_header_rows=nh, status="ok")
    ev = table_ev(cref("r3c1", "Ours", "BLEU ↑", "29.3"))
    full = build_checks("Ours achieves the highest BLEU.", ev, {"table:1": TableContext(rec)})
    c = full.checks[0]
    assert c.relation == "extremum" and c.comparison_set_status == "complete"
    assert [x.cell_id for x in c.comparison_set] == ["r1c1", "r2c1", "r3c1"]
    assert c.direction == "higher_better"
    part = build_checks("Ours achieves the highest BLEU.", ev, {"table:1": TableContext(rec, rows_included=[3])})
    assert part.checks[0].comparison_set_status == "partial" and part.missing_information
    uni = build_checks("Our approach improves recall on every dataset.", ev, {"table:1": TableContext(rec)})
    assert "universal" in rel(uni)


def test_multiple_checks_and_time():
    r = build_checks("From 2019 to 2021, accuracy rose from 80% to 90%.", ACC)
    assert len(r.checks) >= 2 and r.checks[0].time.kind == "range"


def test_parse_helpers():
    qs = quantities("We use 12k samples, ~3.5 hours and 1,234 tokens in 2020.")
    assert [(str(q.value), q.unit) for q in qs] == [("12000", None), ("3.5", "hour"), ("1234", None)]
    assert qs[1].qualifier == "approximately"
    assert time_spec("in Q3 2021").start == "2021-Q3"
    assert cell_number("93.2 ± 0.1 [*]") == Decimal("93.2") and cell_number("−1.5") == Decimal("-1.5")


def test_describe_has_no_results_or_verdict():
    r = build_checks("Our method improves accuracy by 10 percentage points over the baseline.", ACC)
    text = describe(with_computed(r.checks))
    assert "percentage_point_diff" in text and "80.0" in text and "Baseline / Acc (%)" in text
    for leak in ("computed", "true", "false", "correct", "supported"):
        assert leak not in text.lower()
