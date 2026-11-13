"""Stage verify chạy hết chuỗi trên bộ dữ liệu thu nhỏ với model giả (không GPU)."""

import argparse
import json

import pytest

from sciverx import cli
from sciverx.data.loader import load_paper
from sciverx.llm import ScriptedLLM
from sciverx.m1_prepare import tables_path, units_path
from sciverx.m1_prepare.units import build_units, whitespace_counter as wc
from sciverx.schemas import EvidencePackage, TableRecord, VerifierOutput, VerifierRequest
from sciverx.utils.io import read_jsonl, write_jsonl

ALIGN = json.dumps({"status": "matched", "evidence": [{"evidence_id": "table:1", "cell_ids": ["r2c1"]}]})


def responder(req):
    return ALIGN if "Reply with JSON only" in req.prompt else "Therefore, the final answer is: Answer: yes"


def _cfg(mini_sciver, tmp_path, setting, conds, unified=False):
    cfg = {"split": "validation", "seed_global": 2026, "experiment": f"t_{setting}",
           "paths": {"data_root": str(mini_sciver), "work": str(tmp_path / "w"), "outputs": str(tmp_path / "o")},
           "m1": {"gold": {"audit_random_n": 1}, "units": {"max_wordpieces": 60, "embed_model": "bag"}},
           "m3": {"k_units": 50, "rrf_constant": 60, "bm25": {"k1": 1.5, "b": 0.75},
                  "budget": {"max_evidence_tokens": 400, "max_parents": 3, "max_images": 2}},
           "m4": {"max_retries": 1, "max_align_tokens": 2000},
           "verifier": {"setting": setting, "table_conditions": conds, "method": "bm25", "unified": unified,
                        "model": "mock", "max_tokens": 32,
                        "samples": [{"name": "t1_s1", "temperature": 1.0, "seed": 1},
                                    {"name": "t0", "temperature": 0.0, "seed": 0}]}}
    for st in ("select", "sources", "gold", "normalize"):
        cli.resolve_stage(st)(cfg, argparse.Namespace())
    units = []
    for pid in ("2409.00001v1", "2409.00002v2"):
        recs = [r for r in read_jsonl(tables_path(cfg, "G1"), TableRecord) if r.paper_id == pid]
        units += build_units(pid, load_paper(mini_sciver, f"papers/{pid}.json"), recs, 60, wc)[0]
    write_jsonl(units_path(cfg, "G1"), units)
    return cfg


@pytest.mark.parametrize("setting,conds,unified", [
    ("S-oracle", ["IMG", "G1"], False), ("S-oracle", ["G1"], True),
    ("S-retr", ["G1"], False), ("S-align4", ["G1"], False), ("S-align45", ["G1"], False)])
def test_verify_stage_end_to_end(mini_sciver, tmp_path, setting, conds, unified):
    cfg = _cfg(mini_sciver, tmp_path, setting, conds, unified)
    llm = ScriptedLLM(responder=responder)
    fn = cli.resolve_stage("verify")
    assert fn(cfg, argparse.Namespace(), llm=llm, count=wc, load_image=lambda p: b"") == 0
    run_dir = next((tmp_path / "o").iterdir())
    for cond in conds:
        reqs = read_jsonl(run_dir / cond / "verifier_requests.jsonl", VerifierRequest)
        outs = read_jsonl(run_dir / cond / "verifier_outputs.jsonl", VerifierOutput)
        assert len(outs) == 2 * len(reqs) == 6
        assert all(o.pred_new is True and o.evidence_setting == setting for o in outs)
        assert {r.request_hash for r in reqs} == {o.request_hash for o in outs}
        if setting.startswith("S-align"):
            pkgs = read_jsonl(run_dir / cond / "packages.jsonl", EvidencePackage)
            assert len(pkgs) == 3 and all(p.evidence_setting == setting for p in pkgs)
            expected = "unified_v1_numeric" if setting == "S-align45" else "unified_v1"
            assert {r.prompt_id for r in reqs} == {expected}
    assert (run_dir / "manifest.json").exists()
