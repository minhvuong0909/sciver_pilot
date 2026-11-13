import pytest
from pydantic import ValidationError

from sciverx import SCHEMA_VERSION
from sciverx import schemas as S


def _cell(cell_id="r0c0", row=0, col=0, **kw):
    base = dict(cell_id=cell_id, row=row, col=col, is_header=False, text_raw="1")
    base.update(kw)
    return S.SourceCell(**base)


def _retrieval():
    return S.RetrievalResult(
        claim_id="validation-1", method="hybrid", table_condition="G1",
        params={"rrf_constant": 60}, n_units_in_pool=3, n_parents_in_pool=2,
        candidates=[S.RetrievalCandidate(parent_id="T1", best_unit_id="u1", kind="table_rows",
                                         rrf_score=0.03, final_rank=1)],
        pool_hash="abc")


def test_every_record_carries_schema_version():
    c = S.ClaimInput(claim_id="validation-1", split="validation", paper_id="p", text="t")
    assert c.schema_version == SCHEMA_VERSION == "1.1"
    assert '"schema_version":"1.1"' in c.model_dump_json()


def test_claim_input_rejects_label_and_unknown_fields():
    with pytest.raises(ValidationError):
        S.ClaimInput(claim_id="x", split="validation", paper_id="p", text="t", label=True)


def test_claim_input_rejects_unknown_split():
    with pytest.raises(ValidationError):
        S.ClaimInput(claim_id="x", split="train", paper_id="p", text="t")


def test_oracle_locator_has_no_label():
    gold = S.GoldInfo(claim_id="x", label=True, claim_type="direct", section_ids=["S4"],
                      visual_items=[{"item_id": "S4.T2", "item_type": "table", "image_path": "a.png"}])
    loc = S.OracleEvidenceLocator.from_gold(gold)
    assert "label" not in loc.model_dump()
    assert loc.visual_items == gold.visual_items
    loc.visual_items[0]["item_id"] = "changed"
    assert gold.visual_items[0]["item_id"] == "S4.T2"  # bản sao, không dùng chung
    with pytest.raises(ValidationError):
        S.OracleEvidenceLocator(**gold.model_dump())


def test_source_cell_span_must_be_positive():
    assert _cell(rowspan=2, colspan=3).colspan == 3
    with pytest.raises(ValidationError):
        _cell(rowspan=0)


def test_table_record_roundtrip_and_id():
    rid = S.TableRecord.make_id("p1", "S4.T2", "G0")
    assert rid == "p1::S4.T2::G0"
    t = S.TableRecord(table_record_id=rid, paper_id="p1", table_id="S4.T2", condition="G0",
                      extractor="gold_json", caption="Table 2", cells=[_cell()],
                      n_header_rows=1, status="ok")
    again = S.TableRecord.model_validate_json(t.model_dump_json())
    assert again == t
    with pytest.raises(ValidationError):
        S.TableRecord(**{**t.model_dump(), "condition": "IMG"})


def test_operand_ref_accepts_cell_or_span():
    cell = S.CellRef(table_record_id="p::T::G1", condition="G1", cell_id="r1c2",
                     row_header="BERT", col_header="F1", text_raw="93.2")
    op = S.Operand(source="evidence_cell", ref=cell, value_text="93.2", value="93.2")
    again = S.Operand.model_validate_json(op.model_dump_json())
    assert isinstance(again.ref, S.CellRef)
    span = S.SpanRef(unit_id="u", section_id="S4", char_start=0, char_end=4, text="text")
    again = S.Operand.model_validate_json(
        S.Operand(source="evidence_span", ref=span, value_text="4").model_dump_json())
    assert isinstance(again.ref, S.SpanRef)


def test_numeric_check_relation_is_closed():
    kw = dict(check_id="c1", claim_operands=[], evidence_operands=[],
              comparison_set_status="not_applicable", qualifier="none", decided_by="rule")
    S.NumericCheck(relation="percentage_point_diff", **kw)
    with pytest.raises(ValidationError):
        S.NumericCheck(relation="guess", **kw)


def test_parse_error_distinct_from_not_found():
    base = dict(claim_id="validation-1", claim_text="t", paper_id="p", evidence_setting="S-align4",
                table_condition="G1", retrieval=_retrieval(), evidence=[],
                missing_information=[], budget={}, errors=[], provenance={})
    a = S.EvidencePackage(alignment_status="parse_error", **base)
    b = S.EvidencePackage(alignment_status="not_found", **base)
    assert a.alignment_status != b.alignment_status
    assert S.EvidencePackage.model_validate_json(a.model_dump_json()) == a


def test_verifier_output_accepts_img_condition():
    out = S.VerifierOutput(claim_id="x", run_id="r", system="P0", evidence_setting="S-oracle",
                           table_condition="IMG", prompt_id="sciver_p1", request_hash="h",
                           sample_name="t0", temperature=0.0, seed=0, response="Answer: yes",
                           finish_reason="stop", n_images=1)
    assert out.pred_new is None
