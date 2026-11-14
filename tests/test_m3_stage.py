"""Index cache và chạy truy xuất trên bộ dữ liệu thu nhỏ (embedder giả, bộ đếm khoảng trắng)."""

import argparse

import numpy as np

from sciverx import cli
from sciverx.m1_prepare import tables_path, units_path, work_dir
from sciverx.m1_prepare.index import build_indexes, index_dir, load_vectors, save_vectors
from sciverx.m1_prepare.units import build_units, whitespace_counter as wc
from sciverx.m3_retrieve.stage import run_retrieval
from sciverx.schemas import ClaimInput, EvidenceUnit, TableRecord
from sciverx.utils.io import read_jsonl, write_jsonl

from test_m3_retrieve import BagEmbedder


def _prepare(mini_sciver, tmp_path):
    cfg = {"split": "validation", "seed_global": 2026,
           "paths": {"data_root": str(mini_sciver), "work": str(tmp_path / "w")},
           "m1": {"gold": {"audit_random_n": 1}, "units": {"embed_model": "bag", "max_wordpieces": 40}},
           "m3": {"methods": ["bm25", "dense", "hybrid"], "k_units": 50, "rrf_constant": 60,
                  "bm25": {"k1": 1.5, "b": 0.75},
                  "budget": {"max_evidence_tokens": 200, "max_parents": 3, "max_images": 2}}}
    ns = argparse.Namespace()
    for st in ("select", "sources", "gold", "normalize"):
        cli.resolve_stage(st)(cfg, ns)
    from sciverx.data.loader import load_paper
    units = []
    for pid in ("2409.00001v1", "2409.00002v2"):
        recs = [r for r in read_jsonl(tables_path(cfg, "G1"), TableRecord) if r.paper_id == pid]
        u, _ = build_units(pid, load_paper(mini_sciver, f"papers/{pid}.json"), recs, 40, wc)
        units += u
    write_jsonl(units_path(cfg, "G1"), units)
    return cfg


def test_vector_cache_invalidated_on_content_change(tmp_path):
    u = [EvidenceUnit(unit_id="p::paragraph::1::0", paper_id="p", kind="paragraph", parent_id="1",
                      text="a", n_wordpieces=1)]
    save_vectors(tmp_path, u, np.ones((1, 3)), "m")
    assert load_vectors(tmp_path, u, "m").shape == (1, 3)
    assert load_vectors(tmp_path, u, "other-model") is None
    changed = [u[0].model_copy(update={"text": "b"})]
    assert load_vectors(tmp_path, changed, "m") is None


def test_run_retrieval_end_to_end_on_mini(mini_sciver, tmp_path):
    cfg = _prepare(mini_sciver, tmp_path)
    units = read_jsonl(units_path(cfg, "G1"), EvidenceUnit)
    emb = BagEmbedder()
    save_vectors(index_dir(cfg, "G1"), units, emb.encode([u.text for u in units]), "bag")
    indexes = build_indexes(units, "G1", load_vectors(index_dir(cfg, "G1"), units, "bag"))
    claims = read_jsonl(work_dir(cfg) / "claims_input.jsonl", ClaimInput)
    tables = {}
    for r in read_jsonl(tables_path(cfg, "G1"), TableRecord):
        tables.setdefault(r.paper_id, {})[r.table_id] = r
    for method in ("bm25", "dense", "hybrid"):
        res, sel = run_retrieval(cfg, claims, indexes, tables, "G1", method, wc,
                                 emb if method != "bm25" else None)
        assert [r.claim_id for r in res] == [c.claim_id for c in claims]
        for r in res:
            assert r.method == method and r.candidates
            assert len({c.parent_id for c in r.candidates}) == len(r.candidates)
        for s in sel:
            assert s["budget"]["tokens_used"] <= 200 and len(s["parents"]) <= 3
