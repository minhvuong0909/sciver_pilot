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
