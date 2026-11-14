from sciverx.evaluation.paired_sets import paired_oracle, paired_retr
from sciverx.evaluation.report import evaluate_retrieval_dir
from sciverx.schemas import GoldInfo, RetrievalCandidate, RetrievalResult, TableRecord
from sciverx.utils.io import write_jsonl


def _t(pid, tid, status="ok", cond="G0"):
    return TableRecord(table_record_id=f"{pid}::{tid}::{cond}", paper_id=pid, table_id=tid, condition=cond,
                       extractor="x", caption="", cells=[], n_header_rows=0, status=status)


def _g(cid, items):
    return GoldInfo(claim_id=cid, label=True, claim_type="direct", section_ids=["4"],
                    visual_items=[{"item_id": i, "item_type": "table", "image_path": ""} for i in items])


GOLDS = {"c1": _g("c1", ["1"]), "c2": _g("c2", ["2"]), "c3": _g("c3", ["1"])}
PAPER = {"c1": "p", "c2": "p", "c3": "q"}


def test_paired_retr_requires_every_table_of_the_paper():
    g = [_t("p", "1"), _t("p", "2"), _t("q", "1"), _t("q", "2", "gold_unavailable")]
    assert paired_retr(GOLDS, PAPER, g) == ["c1", "c2"]


def test_paired_oracle_requires_g_ok_and_e_present_for_evidence_tables():
    g = [_t("p", "1"), _t("p", "2", "error"), _t("q", "1")]
    e = [_t("p", "1", "no_table_found", "E0"), _t("p", "2", cond="E0")]
    assert paired_oracle(GOLDS, PAPER, g, e) == ["c1"]


def test_evaluate_retrieval_dir(tmp_path):
    write_jsonl(tmp_path / "gold.jsonl", GOLDS.values())
    res = [RetrievalResult(claim_id=c, method="bm25", table_condition="G1", params={}, n_units_in_pool=3,
                           n_parents_in_pool=2, pool_hash="h",
                           candidates=[RetrievalCandidate(parent_id=p, best_unit_id="u", kind="x", final_rank=i + 1)
                                       for i, p in enumerate(["table:1", "section:4"])]) for c in GOLDS]
    write_jsonl(tmp_path / "run" / "retrieval" / "G1_bm25.jsonl", res)
    m = evaluate_retrieval_dir(tmp_path / "run", tmp_path / "gold.jsonl", paired={"c1"})
    assert m["G1_bm25"]["all"]["hit@1"] == round(2 / 3, 4) and m["G1_bm25"]["direct"]["n_claims"] == 3
    assert m["G1_bm25@PAIRED_RETR"]["all"]["n_claims"] == 1
