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
        assert all(o.status == "ok" for o in outs)
        if setting.startswith("S-align"):
            pkgs = read_jsonl(run_dir / cond / "packages.jsonl", EvidencePackage)
            assert len(pkgs) == 3 and all(p.evidence_setting == setting for p in pkgs)
            expected = "unified_v1_numeric" if setting == "S-align45" else "unified_v1"
            assert {r.prompt_id for r in reqs} == {expected}
    assert (run_dir / "manifest.json").exists()
    m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert m["model"] == "mock" and m["seeds"] == [1, 0] and m["duration_s"] is not None
    assert m["error_counts"]["total"] == 6 * len(conds) and m["error_counts"]["ok"] == 6 * len(conds)
    assert m["error_counts"]["retries"] == 0 and m["error_counts"]["truncated"] == 0
    assert m["config_hash"] and m["claim_ids"]                 # update_manifest không xóa phần cũ


class _Killed(BaseException):
    """Giả lập GPU/máy bị ngắt: không phải Exception nên retry không bắt."""


class _DieAfter:
    """Model giả trả lời được `n_ok` lô rồi 'chết'. Ghi lại số request mỗi lần gọi."""
    model_name = "mock/die"

    def __init__(self, n_ok: int):
        self.n_ok = n_ok
        self.batches: list[int] = []

    def generate(self, reqs):
        from sciverx.llm import LLMResponse
        if len(self.batches) >= self.n_ok:
            raise _Killed()
        self.batches.append(len(reqs))
        return [LLMResponse(text=responder(r)) for r in reqs]


def test_verify_resumes_after_interruption_without_losing_chunks(mini_sciver, tmp_path):
    cfg = _cfg(mini_sciver, tmp_path, "S-oracle", ["G1"])
    cfg["verifier"]["chunk_size"] = 2                       # 3 claim × 2 lượt → 4 lô
    fn = cli.resolve_stage("verify")
    dying = _DieAfter(n_ok=2)
    with pytest.raises(_Killed):
        fn(cfg, argparse.Namespace(), llm=dying, count=wc, load_image=lambda p: b"")
    run_dir = next((tmp_path / "o").iterdir())
    saved = read_jsonl(run_dir / "G1" / "verifier_outputs.jsonl", VerifierOutput)
    assert len(saved) == 3 == sum(dying.batches)            # 2 lô đã xong được ghi ngay ra đĩa
    again = ScriptedLLM(responder=responder)
    assert fn(cfg, argparse.Namespace(), llm=again, count=wc, load_image=lambda p: b"") == 0
    outs = read_jsonl(run_dir / "G1" / "verifier_outputs.jsonl", VerifierOutput)
    assert len(again.requests) == 3                         # chỉ chạy phần còn thiếu
    assert len(outs) == 6 and len({(o.claim_id, o.sample_name) for o in outs}) == 6
    assert [o.sample_name for o in outs] == ["t1_s1"] * 3 + ["t0"] * 3


def test_verify_rerun_skips_done_but_retries_errors(mini_sciver, tmp_path):
    cfg = _cfg(mini_sciver, tmp_path, "S-oracle", ["G1"])
    fn = cli.resolve_stage("verify")
    assert fn(cfg, argparse.Namespace(), llm=ScriptedLLM(responder=responder), count=wc,
              load_image=lambda p: b"") == 0
    run_dir = next((tmp_path / "o").iterdir())
    p = run_dir / "G1" / "verifier_outputs.jsonl"
    outs = read_jsonl(p, VerifierOutput)
    outs[0] = outs[0].model_copy(update={"status": "error", "error": "OOM", "response": ""})
    write_jsonl(p, outs)
    with p.open("a", encoding="utf-8") as f:
        f.write('{"claim_id": "half-written')                # dòng ghi dở khi máy tắt
    again = ScriptedLLM(responder=responder)
    assert fn(cfg, argparse.Namespace(), llm=again, count=wc, load_image=lambda p: b"") == 0
    assert len(again.requests) == 1                         # chỉ chạy lại mẫu lỗi
    final = read_jsonl(p, VerifierOutput)
    assert len(final) == 6 and all(o.status == "ok" for o in final)
