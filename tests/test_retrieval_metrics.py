"""SPEC 12.4 test_retrieval_metrics: ví dụ có nhiều bằng chứng, tính tay."""

import pytest

from sciverx.evaluation.retrieval_metrics import claim_metrics, evaluate_retrieval
from sciverx.schemas import GoldInfo, RetrievalCandidate, RetrievalResult

GOLD = GoldInfo(claim_id="v-1", label=True, claim_type="parallel", section_ids=["4.1", "5"],
                visual_items=[{"item_id": "1", "item_type": "table", "image_path": "a"},
                              {"item_id": "3", "item_type": "table", "image_path": "b"}])
# G = {table:1, table:3, section:4.1, section:5}
RANKED = ["section:2", "table:1", "section:5.2", "table:7", "table:3", "section:4.1.1"]


def test_claim_metrics_hand_computed():
    m = claim_metrics(RANKED, GOLD)
    assert m["hit@1"] == 0.0 and m["recall@1"] == 0.0
    assert m["hit@3"] == 1.0 and m["recall@3"] == pytest.approx(2 / 4)   # table:1, section:5 (5.2 ⊂ 5)
    assert m["visual_complete@3"] == 0.0 and m["visual_complete@5"] == 1.0
    assert m["recall@5"] == pytest.approx(3 / 4)
    assert m["recall@10"] == pytest.approx(4 / 4)                       # 4.1.1 ⊂ 4.1
    assert m["text_recall@3"] == pytest.approx(1 / 2) and m["text_recall@10"] == 1.0
    assert m["mrr"] == pytest.approx(1 / 2)


def test_section_prefix_does_not_overmatch():
    g = GOLD.model_copy(update={"section_ids": ["4"], "visual_items": []})
    assert claim_metrics(["section:41", "section:4"], g)["mrr"] == pytest.approx(1 / 2)


def test_no_relevant_gives_zero_mrr():
    assert claim_metrics(["section:9"], GOLD)["mrr"] == 0.0


def test_evaluate_retrieval_groups_and_nan_text():
    def res(cid, parents):
        return RetrievalResult(claim_id=cid, method="hybrid", table_condition="G1", params={},
                               n_units_in_pool=10, n_parents_in_pool=6, pool_hash="h",
                               candidates=[RetrievalCandidate(parent_id=p, best_unit_id="u", kind="x",
                                                              final_rank=i + 1) for i, p in enumerate(parents)])
    g2 = GoldInfo(claim_id="v-2", label=False, claim_type="direct", section_ids=[],
                  visual_items=[{"item_id": "2", "item_type": "table", "image_path": "c"}])
    out = evaluate_retrieval([res("v-1", RANKED), res("v-2", ["table:2"])], {"v-1": GOLD, "v-2": g2},
                             group_by={"v-1": "parallel", "v-2": "direct"})
    assert out["all"]["n_claims"] == 2 and out["all"]["hit@1"] == 0.5
    assert out["direct"]["mrr"] == 1.0
    assert out["all"]["text_recall@10"] == 1.0  # v-2 không có section chuẩn nên bị bỏ khỏi trung bình


def test_evidence_kind_groups():
    from sciverx.evaluation.retrieval_metrics import evidence_kind
    from sciverx.schemas import GoldInfo

    def g(types):
        return GoldInfo(claim_id="c", label=True, claim_type="direct", section_ids=[],
                        visual_items=[{"item_id": str(i), "item_type": t, "image_path": ""}
                                      for i, t in enumerate(types)])
    assert evidence_kind(g(["table"])) == "table_only"
    assert evidence_kind(g(["table", "table"])) == "table_only"
    assert evidence_kind(g(["chart"])) == "chart_only"
    assert evidence_kind(g(["table", "chart"])) == "table+chart"
    assert evidence_kind(g([])) == "none"


def test_compare_runs_paired_mcnemar_and_bootstrap():
    from sciverx.evaluation.retrieval_metrics import compare_runs
    a = {f"c{i}": {"hit@5": 1.0, "recall@5": 0.5, "visual_complete@5": 0.0} for i in range(10)}
    b = {f"c{i}": {"hit@5": 1.0, "recall@5": 1.0, "visual_complete@5": 1.0 if i < 8 else 0.0}
         for i in range(10)}
    b["extra"] = {"hit@5": 1.0, "recall@5": 1.0, "visual_complete@5": 1.0}   # chỉ so trên phần giao
    paper = {f"c{i}": f"p{i % 5}" for i in range(10)}
    r = compare_runs(a, b, paper, n_boot=200)
    assert r["n_claims"] == 10
    vc = r["visual_complete@5"]
    assert vc["diff"] == 0.8 and vc["b_only_better"] == 8 and vc["a_only_better"] == 0
    assert vc["mcnemar_p"] < 0.01 and vc["ci_low"] > 0
    assert r["recall@5"]["diff"] == 0.5 and "mcnemar_p" not in r["recall@5"]   # không phải 0/1
    assert r["hit@5"]["diff"] == 0.0 and r["hit@5"]["mcnemar_p"] == 1.0


def test_delivered_metrics_use_what_the_model_receives():
    from sciverx.evaluation.retrieval_metrics import delivered_metrics, summarize_rows
    from sciverx.schemas import GoldInfo
    gold = GoldInfo(claim_id="c", label=True, claim_type="parallel", section_ids=["4"],
                    visual_items=[{"item_id": "1", "item_type": "table", "image_path": ""},
                                  {"item_id": "2", "item_type": "table", "image_path": ""}])
    sel_one = {"claim_id": "c", "parents": [{"parent_id": "section:4.1"}, {"parent_id": "table:1"}],
               "budget": {"tokens_used": 900, "truncated": False, "dropped_parents": []}}
    sel_both = {"claim_id": "c", "parents": [{"parent_id": "table:2"}, {"parent_id": "table:1"}],
                "budget": {"tokens_used": 1500, "truncated": True, "dropped_parents": ["section:4"]}}
    a, b = delivered_metrics(sel_one, gold), delivered_metrics(sel_both, gold)
    assert a["delivered_visual_complete"] == 0.0 and a["delivered_recall"] == 2 / 3   # tiểu mục 4.1 khớp section 4
    assert b["delivered_visual_complete"] == 1.0 and b["truncated"] == 1.0 and b["dropped_for_budget"] == 1.0
    s = summarize_rows({"x": a, "y": b}, {"x": "parallel", "y": "direct"})
    assert s["all"]["delivered_visual_complete"] == 0.5 and s["parallel"]["n_claims"] == 1
