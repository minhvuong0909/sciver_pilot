"""SPEC 12.4 test_no_leakage_requests: ở S-retr / S-align, xáo trộn ngẫu nhiên mọi trường
gold (nhãn, section, claim_type, các câu perturbed...) → pool_hash, kết quả truy xuất (tất
định) và request_hash gửi tới M4/verifier KHÔNG đổi. Dùng model giả, không so văn bản sinh ra."""

import argparse
import json
import random

import pytest

from sciverx import cli
from sciverx.data.loader import load_paper
from sciverx.llm import ScriptedLLM
from sciverx.m1_prepare import tables_path, work_dir
from sciverx.m1_prepare.units import build_units, whitespace_counter as wc
from sciverx.m3_retrieve.retrieve import PaperIndex
from sciverx.pipeline import run_claim
from sciverx.schemas import ClaimInput, TableRecord
from sciverx.utils.io import read_jsonl

GOLD_FIELDS = ("label", "section", "claim_type", "origin_statement", "perturbed_statement",
               "perturbed_explanation", "request_id")


def _shuffle_gold(root, seed):
    rows = json.loads((root / "valset.json").read_text())
    rng = random.Random(seed)
    for f in GOLD_FIELDS:
        vals = [r[f] for r in rows]
        rng.shuffle(vals)
        for r, v in zip(rows, vals):
            r[f] = v
    for r in rows:  # claim_type phải khớp kiểu item để dữ liệu còn hợp lệ
        r["claim_type"] = "direct" if "item" in r else "parallel"
        r["label"] = not r["label"] if rng.random() < 0.5 else r["label"]
    (root / "valset.json").write_text(json.dumps(rows))


ALIGN_ANSWER = json.dumps({"status": "matched", "evidence": [{"evidence_id": "table:1", "cell_ids": ["r2c1"]}]})


def _run_all(mini_sciver, work, setting):
    cfg = {"split": "validation", "seed_global": 2026,
           "paths": {"data_root": str(mini_sciver), "work": str(work)},
           "m1": {"gold": {"audit_random_n": 1}, "units": {"max_wordpieces": 60}},
           "m3": {"k_units": 50, "rrf_constant": 60, "bm25": {"k1": 1.5, "b": 0.75},
                  "budget": {"max_evidence_tokens": 400, "max_parents": 3, "max_images": 2}},
           "m4": {"max_retries": 2, "max_align_tokens": 2000}}
    for st in ("select", "sources", "gold", "normalize"):
        cli.resolve_stage(st)(cfg, argparse.Namespace())
    claims = read_jsonl(work_dir(cfg) / "claims_input.jsonl", ClaimInput)
    g1 = read_jsonl(tables_path(cfg, "G1"), TableRecord)
    out = []
    llm = ScriptedLLM(responder=lambda req: ALIGN_ANSWER)
    for c in claims:
        paper = load_paper(mini_sciver, f"papers/{c.paper_id}.json")
        tabs = {t.table_id: t for t in g1 if t.paper_id == c.paper_id}
        units, _ = build_units(c.paper_id, paper, list(tabs.values()), 60, wc)
        idx = PaperIndex.build(c.paper_id, "G1", units)
        sections = {s["section_id"]: s["text"] for s in paper["sections"]}
        r = run_claim(c, idx, tabs, sections, setting, "bm25", cfg, wc, llm)
        out.append((c.claim_id, r.retrieval.pool_hash, r.retrieval.model_dump_json(),
                    r.verifier_input.request.request_hash))
    m4_hashes = [hash(req.prompt) for req in llm.requests]
    return out, m4_hashes


@pytest.mark.parametrize("setting", ["S-retr", "S-align4", "S-align45"])
def test_shuffling_gold_does_not_change_requests(mini_sciver, tmp_path, setting):
    base, base_m4 = _run_all(mini_sciver, tmp_path / "a", setting)
    for seed in (1, 2):
        _shuffle_gold(mini_sciver, seed)
        again, again_m4 = _run_all(mini_sciver, tmp_path / f"b{seed}", setting)
        assert again == base
        assert again_m4 == base_m4
    assert base and all(len(x[3]) == 64 for x in base)


def test_pipeline_has_no_gold_parameter():
    import inspect
    params = inspect.signature(run_claim).parameters
    assert not any("gold" in p.lower() or "label" in p.lower() for p in params)
