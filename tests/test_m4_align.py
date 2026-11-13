import json

from sciverx.llm import ScriptedLLM
from sciverx.m4_align.align import Candidate, align_claim, fit_candidates
from sciverx.schemas import ClaimInput, TableRecord
from sciverx.tables.html_cells import parse_html_table

CLAIM = ClaimInput(claim_id="validation-1", split="validation", paper_id="p",
                   text="Ours reaches 92.4 F1 on SQuAD, above BERT.")


def _table():
    cells, nh = parse_html_table("<table><thead><tr><th>Model</th><th>SQuAD F1</th></tr></thead>"
                                 "<tr><td>BERT</td><td>90.9</td></tr><tr><td>Ours</td><td>92.4</td></tr></table>")
    return TableRecord(table_record_id="p::1::G1", paper_id="p", table_id="1", condition="G1",
                       extractor="x", caption="Table 1: QA.", cells=cells, n_header_rows=nh, status="ok")


CANDS = [Candidate("table:1", 1, table=_table()),
         Candidate("section:4.1", 2, section_text="We train on SQuAD. Ours reaches 92.4 F1! BERT is lower.",
                   unit_id="p::paragraph::4.1::0")]


def _ans(**kw):
    return json.dumps({"status": "matched", "evidence": [], "alternatives": [], "missing": [], **kw})


def test_render_prints_ids():
    assert "[r2c1] 92.4" in CANDS[0].render()
    assert "[s1] Ours reaches 92.4 F1!" in CANDS[1].render()


def test_matched_with_cells_and_sentences_headers_from_table():
    llm = ScriptedLLM(script=["```json\n" + _ans(evidence=[
        {"evidence_id": "table:1", "cell_ids": ["r2c1", "r1c1"], "reason": "Ours/BERT F1"},
        {"evidence_id": "section:4.1", "sentence_ids": ["s1"]}]) + "\n```"])
    r = align_claim(CLAIM, CANDS, llm)
    assert r.status == "matched" and r.attempts == 1
    c = r.evidence[0].cells[0]
    assert (c.row_header, c.col_header, c.text_raw) == ("Ours", "SQuAD F1", "92.4")
    s = r.evidence[1].spans[0]
    assert s.text == "Ours reaches 92.4 F1!" and CANDS[1].section_text[s.char_start:s.char_end] == s.text
    assert llm.requests[0].temperature == 0.0


def test_invalid_ids_retry_then_parse_error():
    bad = _ans(evidence=[{"evidence_id": "table:1", "cell_ids": ["r9c9"]}])
    llm = ScriptedLLM(script=[bad, "not json", _ans(evidence=[{"evidence_id": "table:7"}])])
    r = align_claim(CLAIM, CANDS, llm, max_retries=2)
    assert r.status == "parse_error" and r.attempts == 3 and len(r.errors) == 3
    assert r.status != "not_found"


def test_retry_recovers():
    llm = ScriptedLLM(script=["{oops", _ans(evidence=[{"evidence_id": "table:1", "cell_ids": ["r2c1"]}])])
    r = align_claim(CLAIM, CANDS, llm)
    assert r.status == "matched" and r.attempts == 2 and len(r.errors) == 1


def test_not_found_and_ambiguous():
    r = align_claim(CLAIM, CANDS, ScriptedLLM(script=[_ans(status="not_found", missing=["no dev set"])]))
    assert r.status == "not_found" and r.missing == ["no dev set"]
    amb = _ans(evidence=[{"evidence_id": "table:1", "cell_ids": ["r2c1"]}],
               alternatives=[{"evidence_id": "section:4.1", "sentence_ids": ["s1"]}])
    r = align_claim(CLAIM, CANDS, ScriptedLLM(script=[amb]))
    assert r.status == "ambiguous" and r.evidence == [] and len(r.candidates) == 2


def test_matched_without_evidence_is_invalid():
    r = align_claim(CLAIM, CANDS, ScriptedLLM(script=[_ans()]), max_retries=0)
    assert r.status == "parse_error"


def test_fit_candidates_drops_by_rank_and_logs():
    kept, dropped = fit_candidates(CANDS, max_tokens=25, count=lambda t: len(t.split()))
    assert [c.evidence_id for c in kept] == ["table:1"] and dropped == ["section:4.1"]
