"""SPEC 12.4: test_tokenizer, test_rrf, test_parent_dedup và truy xuất tất định."""

import numpy as np
import pytest

from sciverx.m3_retrieve import parent_key
from sciverx.m3_retrieve.rankers import aggregate_parents, rank_order, rrf_fuse
from sciverx.m3_retrieve.retrieve import PaperIndex, retrieve
from sciverx.m3_retrieve.tokenizer import tokenize
from sciverx.schemas import ClaimInput, EvidenceUnit


def test_tokenizer_keeps_scientific_tokens():
    toks = tokenize("GPT-4o reaches 93.2 F1@5 and top-1 accuracy of 12.5% on the SQuAD v1.1 set.")
    for t in ("gpt-4o", "93.2", "f1@5", "top-1", "12.5%", "squad", "v1.1"):
        assert t in toks
    assert "the" not in toks and "and" not in toks
    assert tokenize("Accuracy:93.2") == ["accuracy", "93.2"]


def test_rrf_hand_computed():
    # bm25: [a=0, b=1, c=2], dense: [c=2, a=0]
    fused = rrf_fuse({"bm25": [0, 1, 2], "dense": [2, 0]}, constant=60)
    assert fused[0] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1] == pytest.approx(1 / 62)
    assert fused[2] == pytest.approx(1 / 63 + 1 / 61)
    assert rank_order(np.array([fused[0], fused[1], fused[2]])) == [0, 2, 1]


def test_rank_order_is_stable_on_ties():
    assert rank_order(np.array([1.0, 2.0, 2.0, 0.5]), k=3) == [1, 2, 0]


def test_parent_dedup_five_units_one_slot():
    parents = ["table:2"] * 5 + ["section:4", "table:1"]
    scores = {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.6, 4: 0.5, 5: 0.85, 6: 0.1}
    hits = aggregate_parents(scores, parents)
    assert [h.parent for h in hits] == ["table:2", "section:4", "table:1"]
    assert hits[0].best_unit == 0 and [h.rank for h in hits] == [1, 2, 3]


def _u(i, kind, parent, text):
    return EvidenceUnit(unit_id=f"p::{kind}::{parent}::{i}", paper_id="p", kind=kind, parent_id=parent,
                        table_condition="G1" if kind != "paragraph" else None, text=text,
                        n_wordpieces=len(text.split()))


UNITS = [
    _u(0, "paragraph", "1", "Intro: We study question answering."),
    _u(1, "table_rows", "1", "Table 1: QA results. Model | F1 BERT | 90.9"),
    _u(2, "table_rows", "1", "Table 1: QA results. Model | F1 Ours | 92.4"),
    _u(3, "paragraph", "4", "Results: Ours reaches 92.4 F1 on SQuAD."),
    _u(4, "table_full", "2", "Table 2: Speed. Model | ms Ours | 40"),
]


class BagEmbedder:
    name = "bag"
    vocab = ["f1", "92.4", "ours", "speed", "ms", "squad", "bert"]

    def encode(self, texts):
        m = np.array([[float(w in tokenize(t)) for w in self.vocab] for t in texts]) + 1e-3
        return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_parent_keys_disambiguate_section_and_table():
    assert parent_key(UNITS[0]) == "section:1" and parent_key(UNITS[1]) == "table:1"


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_retrieve_dedups_parents_and_is_deterministic(method):
    emb = BagEmbedder()
    idx = PaperIndex.build("p", "G1", UNITS, emb)
    claim = ClaimInput(claim_id="validation-1", split="validation", paper_id="p",
                       text="Ours achieves 92.4 F1 on SQuAD.")
    qv = emb.encode([claim.text])[0]
    cfg = {"k_units": 50, "rrf_constant": 60, "bm25": {"k1": 1.5, "b": 0.75}}
    a = retrieve(claim, idx, method, cfg, qv)
    b = retrieve(claim, idx, method, cfg, qv)
    assert a == b
    parents = [c.parent_id for c in a.candidates]
    assert len(parents) == len(set(parents)) == 4
    assert a.n_units_in_pool == 5 and a.n_parents_in_pool == 4
    assert parents[0] in ("table:1", "section:4")
    if method == "hybrid":
        top = a.candidates[0]
        assert top.bm25_rank is not None and top.dense_rank is not None and top.rrf_score > 0


def test_retrieve_rejects_other_paper():
    idx = PaperIndex.build("p", "G1", UNITS)
    with pytest.raises(ValueError):
        retrieve(ClaimInput(claim_id="x", split="validation", paper_id="q", text="t"), idx, "bm25", {})


# ------------------------------------------------------------------ cải thiện truy xuất (X2b)

def test_weighted_rrf_downweights_weak_ranker():
    rank = {"bm25": [0, 1], "dense": [1, 0]}
    equal = rrf_fuse(rank, constant=60)
    assert equal[0] == pytest.approx(equal[1])                 # 1:1 → hòa
    w = rrf_fuse(rank, constant=60, weights={"bm25": 1.0, "dense": 0.5})
    assert w[0] == pytest.approx(1 / 61 + 0.5 / 62) and w[0] > w[1]
    assert rrf_fuse(rank, 60, {"dense": 0.0}) == {0: pytest.approx(1 / 61), 1: pytest.approx(1 / 62)}


class FakeReranker:
    name = "fake"

    def __init__(self, prefer: str):
        self.prefer = prefer
        self.calls = []

    def score(self, query, texts):
        self.calls.append(list(texts))
        return np.array([1.0 if self.prefer in t else 0.0 for t in texts])


def _claim(text="Ours achieves 92.4 F1 on SQuAD."):
    return ClaimInput(claim_id="validation-1", split="validation", paper_id="p", text=text)


def test_rerank_reorders_parents_and_records_scores():
    idx = PaperIndex.build("p", "G1", UNITS)
    base = retrieve(_claim(), idx, "bm25", {})
    rr = FakeReranker(prefer="Speed")                         # đẩy table:2 lên đầu
    res = retrieve(_claim(), idx, "bm25", {"rerank": {"top_n": 10}}, reranker=rr)
    assert res.candidates[0].parent_id == "table:2" != base.candidates[0].parent_id
    assert res.candidates[0].rerank_rank == 1 and res.candidates[0].rerank_score == 1.0
    assert [c.final_rank for c in res.candidates] == [1, 2, 3, 4]
    assert sorted(c.parent_id for c in res.candidates) == sorted(c.parent_id for c in base.candidates)
    assert res.params["rerank_model"] == "fake" and res.params["rerank_top_n"] == 10


def test_rerank_top_n_limits_rescored_units():
    idx = PaperIndex.build("p", "G1", UNITS)
    rr = FakeReranker(prefer="Speed")
    res = retrieve(_claim(), idx, "bm25", {"rerank": {"top_n": 2}}, reranker=rr)
    assert len(rr.calls[0]) == 2                               # chỉ chấm lại 2 đơn vị đầu
    assert len({c.parent_id for c in res.candidates}) == len(res.candidates) == 4
    assert all(c.rerank_rank is None for c in res.candidates[2:] if c.rerank_score is None)


def test_visual_slots_pull_tables_into_top_k():
    from sciverx.m3_retrieve.rankers import ParentHit
    from sciverx.m3_retrieve.retrieve import reserve_visual_slots
    kinds = ["paragraph", "paragraph", "paragraph", "table_rows", "chart_caption"]
    hits = [ParentHit(f"p{i}", i, 1.0 - i / 10, i + 1) for i in range(5)]
    out = reserve_visual_slots(hits, kinds, k=3, n=2)
    assert [h.parent for h in out] == ["p0", "p3", "p4", "p1", "p2"]
    assert [h.rank for h in out] == [1, 2, 3, 4, 5]
    assert reserve_visual_slots(hits, kinds, k=3, n=0) == hits
    assert [h.parent for h in reserve_visual_slots(hits, kinds, k=5, n=2)] == [h.parent for h in hits]
    assert [h.parent for h in reserve_visual_slots(hits, kinds, k=2, n=9)][:2] == ["p3", "p4"]  # n > k


def test_retrieve_with_visual_slots_param_recorded():
    idx = PaperIndex.build("p", "G1", UNITS)
    res = retrieve(_claim("Intro question answering study"), idx, "bm25", {"visual_slots": {"k": 1, "n": 1}})
    assert res.candidates[0].kind != "paragraph"
    assert res.params["visual_slots"] == {"k": 1, "n": 1}


def test_variants_of_defaults_and_validation():
    from sciverx.m3_retrieve.stage import variants_of
    assert variants_of({"methods": ["bm25", "hybrid"]}) == [{"name": "bm25", "method": "bm25"},
                                                            {"name": "hybrid", "method": "hybrid"}]
    v = [{"name": "bm25-vs2", "method": "bm25", "visual_slots": {"k": 5, "n": 2}}]
    assert variants_of({"methods": ["bm25"], "variants": v}) == v
    with pytest.raises(ValueError):
        variants_of({"variants": [{"name": "a_b", "method": "bm25"}]})
    with pytest.raises(ValueError):
        variants_of({"variants": [{"name": "a", "method": "bm25"}, {"name": "a", "method": "dense"}]})


def test_x2b_config_loads():
    from pathlib import Path

    from sciverx.config import load_config
    from sciverx.m3_retrieve.stage import variants_of
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/experiments/x2b_retrieval_variants.yaml")
    names = [v["name"] for v in variants_of(cfg["m3"])]
    assert "bm25" in names and "bm25-rerank-vs2" in names and cfg["m3"]["rrf_constant"] == 60
