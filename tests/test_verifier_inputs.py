"""Dựng đầu vào verifier và chạy với model giả. SPEC 12.4 test_no_label_anywhere (phần verifier)."""

import json

import pytest

from sciverx.data.loader import load_paper, load_split
from sciverx.data.select import select_claims
from sciverx.llm import ScriptedLLM
from sciverx.schemas import OracleEvidenceLocator, TableRecord
from sciverx.tables.html_cells import parse_html_table
from sciverx.verifier.build_inputs import build_aligned, build_oracle, build_retrieved
from sciverx.verifier.run_vllm import run_samples

SAMPLES = [{"name": "t1_s1", "temperature": 1.0, "seed": 1}, {"name": "t0", "temperature": 0.0, "seed": 0}]


def _setup(mini_sciver):
    rows = load_split(mini_sciver, "validation")
    claims, golds = select_claims(rows, "validation")
    tables = {}
    for tid in ("1", "2"):
        cells, nh = parse_html_table("<table><thead><tr><th>Model</th><th>F1</th></tr></thead>"
                                     "<tr><td>Ours</td><td>92.4</td></tr></table>")
        tables[tid] = TableRecord(table_record_id=f"x::{tid}::G1", paper_id="x", table_id=tid,
                                  condition="G1", extractor="t", caption=f"Table {tid}: Results.",
                                  cells=cells, n_header_rows=nh, status="ok")
    return claims, golds, tables


def test_oracle_img_uses_original_prompt_and_images(mini_sciver):
    claims, golds, tables = _setup(mini_sciver)
    c, g = claims[0], golds[0]        # direct, 1 bảng
    paper = load_paper(mini_sciver, f"papers/{c.paper_id}.json")
    b = build_oracle(c, OracleEvidenceLocator.from_gold(g), paper, "IMG", tables, mini_sciver)
    assert b.request.prompt_id == "sciver_p1" and len(b.images) == 1
    assert "based on the image and the caption" in b.request.text
    assert "Experiments:\n\nResults:\nOur model reaches 92.4 F1." in b.request.text  # prepare_context


def test_oracle_text_mode_inserts_table_block(mini_sciver):
    claims, golds, tables = _setup(mini_sciver)
    c, g = claims[1], golds[1]        # parallel, 2 bảng
    paper = load_paper(mini_sciver, f"papers/{c.paper_id}.json")
    b = build_oracle(c, OracleEvidenceLocator.from_gold(g), paper, "G1", tables, mini_sciver)
    assert b.request.prompt_id == "sciver_p2_text" and b.images == []
    assert b.request.text.startswith("Table content (as text):\nitem1 Table 1: Results.")
    u = build_oracle(c, OracleEvidenceLocator.from_gold(g), paper, "G1", tables, mini_sciver, unified=True)
    assert u.request.prompt_id == "unified_v1"


def test_build_inputs_refuses_gold_info(mini_sciver):
    claims, golds, tables = _setup(mini_sciver)
    paper = load_paper(mini_sciver, f"papers/{claims[0].paper_id}.json")
    with pytest.raises(TypeError):
        build_oracle(claims[0], golds[0], paper, "IMG", tables, mini_sciver)


def test_no_label_anywhere_in_requests(mini_sciver):
    claims, golds, tables = _setup(mini_sciver)
    for c, g in zip(claims, golds):
        paper = load_paper(mini_sciver, f"papers/{c.paper_id}.json")
        for cond in ("IMG", "G1"):
            for unified in (False, True):
                r = build_oracle(c, OracleEvidenceLocator.from_gold(g), paper, cond, tables, mini_sciver,
                                 unified).request
                dumped = json.dumps(r.model_dump()).lower()
                for leak in ("label", "entailed", "refuted", "perturbed", "leak", "claim_type"):
                    assert leak not in dumped


def test_retrieved_and_aligned_prompts():
    from sciverx.schemas import ClaimInput
    c = ClaimInput(claim_id="validation-0", split="validation", paper_id="p", text="Ours gets 92.4 F1.")
    parents = [{"parent_id": "table:1", "text": "Table 1: R.\nModel | F1\nOurs | 92.4", "truncated": False},
               {"parent_id": "section:4.1", "text": "Results: Ours reaches 92.4 F1.", "truncated": True}]
    r = build_retrieved(c, parents, ".")
    assert r.request.prompt_id == "unified_v1"
    assert "[Table 1]\nTable content (as text):" in r.request.text
    assert "[Section 4.1] (truncated)" in r.request.text
    a4 = build_aligned(c, "EVIDENCE")
    a45 = build_aligned(c, "EVIDENCE", "- compare 92.4 (claim) with Table 1 r1c1")
    assert a4.request.prompt_id == "unified_v1" and a45.request.prompt_id == "unified_v1_numeric"
    assert a4.request.request_hash != a45.request.request_hash


def test_run_samples_with_mock_model(mini_sciver):
    claims, golds, tables = _setup(mini_sciver)
    paper = load_paper(mini_sciver, f"papers/{claims[0].paper_id}.json")
    b = build_oracle(claims[0], OracleEvidenceLocator.from_gold(golds[0]), paper, "G1", tables, mini_sciver)
    llm = ScriptedLLM(script=["Therefore, the final answer is: Answer: yes", "Answer: no"])
    outs = run_samples(llm, [b], SAMPLES, 64, run_id="r", system="P0", setting="S-oracle",
                       table_condition="G1", load_image=lambda p: b"")
    assert [(o.sample_name, o.pred_new, o.seed) for o in outs] == [("t1_s1", True, 1), ("t0", False, 0)]
    assert [r.seed for r in llm.requests] == [1, 0]          # seed được truyền vào request
    assert all(o.request_hash == b.request.request_hash for o in outs)
