"""Lần chạy GPU X1 (B0 / IMG / G1) bằng model giả: tập con B0, nhập kết quả, báo cáo."""

import argparse
import datetime as dt
import json

from sciverx import cli
from sciverx.llm import ScriptedLLM
from sciverx.utils import manifest
from sciverx.utils.io import read_jsonl
from sciverx.verifier.b0 import import_outputs, make_subset, raw_json_name
from test_verify_stage import _cfg                     # thư mục tests/ nằm trên sys.path


def test_find_or_make_run_id_reuses_dir_from_another_day(tmp_path):
    cfg = {"a": 1}
    first = manifest.find_or_make_run_id(tmp_path, "x", "validation", cfg, dt.date(2026, 11, 26))
    manifest.open_run_dir(tmp_path, manifest.build_manifest(first, cfg, ["c"]))
    again = manifest.find_or_make_run_id(tmp_path, "x", "validation", cfg, dt.date(2026, 11, 27))
    assert again == first == "20261126_x_validation_" + manifest.config_hash(cfg)[:8]
    other = manifest.find_or_make_run_id(tmp_path, "x", "validation", {"a": 2}, dt.date(2026, 11, 27))
    assert other.startswith("20261127_") and other != first


def test_make_subset_picks_rows_by_claim_id_and_absolute_paths(mini_sciver):
    rows = json.loads((mini_sciver / "valset.json").read_text(encoding="utf-8"))
    sub = make_subset(rows, ["validation-2", "validation-0"], "validation", mini_sciver)
    assert [q["claim_id"] for q in sub] == ["validation-2", "validation-0"]
    assert sub[1]["claim"] == rows[0]["claim"]
    assert sub[1]["paper_path"].endswith("papers/2409.00001v1.json") and "./SciVer" not in sub[1]["paper_path"]
    assert rows[0]["paper_path"].startswith("./SciVer/")              # dữ liệu gốc không bị sửa


def test_import_outputs_scores_like_original():
    raw = [{"claim_id": "validation-0", "claim_type": "direct", "image_path": "/a.png",
            "response": "Initially yes... Therefore, the final answer is: Answer: no"},
           {"claim_id": "validation-2", "claim_type": "parallel", "item1_path": "/a", "item2_path": "/b",
            "response": ""}]
    outs = import_outputs(raw, "run")
    assert outs[0].pred_old is True and outs[0].pred_new is False and outs[0].status == "ok"
    assert outs[1].pred_old is None and outs[1].status == "empty" and outs[1].n_images == 2
    assert all(o.system == "B0" and o.sample_name == "b0" and o.seed == -1 for o in outs)
    assert raw_json_name() == "subset_cot/Qwen2.5-VL-7B-Instruct.json"


def _yes(req):
    return "Therefore, the final answer is: Answer: yes"


def test_x1_end_to_end_with_mock_model(mini_sciver, tmp_path):
    cfg = _cfg(mini_sciver, tmp_path, "S-oracle", ["IMG", "G1"])
    cfg["experiment"] = "x1_gpu_img_g1"
    cfg["eval"] = {"reference": "IMG", "bootstrap_n": 50}
    ns = argparse.Namespace()
    # B0: tập con → (giả lập main.py của SciVer) → nhập
    assert cli.resolve_stage("b0-subset")(cfg, ns) == 0
    sub = json.loads((tmp_path / "w" / "validation" / "b0" / "subset.json").read_text(encoding="utf-8"))
    paths = (tmp_path / "w" / "validation" / "b0" / "paths.sh").read_text(encoding="utf-8")
    raw_json = paths.split('RAW_JSON="')[1].split('"')[0]
    for q in sub:
        q["response"] = "Answer: no"
    from pathlib import Path
    Path(raw_json).parent.mkdir(parents=True, exist_ok=True)
    Path(raw_json).write_text(json.dumps(sub), encoding="utf-8")
    assert cli.resolve_stage("b0-import")(cfg, ns) == 0
    # IMG và G1 bằng pipeline; dùng CHUNG thư mục lượt chạy mà b0-subset đã mở
    from sciverx.m1_prepare.units import whitespace_counter as wc
    assert cli.resolve_stage("verify")(cfg, ns, llm=ScriptedLLM(responder=_yes), count=wc,
                                       load_image=lambda p: b"") == 0
    runs = list((tmp_path / "o").iterdir())
    assert len(runs) == 1
    assert {p.name for p in runs[0].iterdir() if p.is_dir()} >= {"B0", "IMG", "G1"}
    assert cli.resolve_stage("report")(cfg, ns) == 0
    rep = json.loads((runs[0] / "eval" / "verdict_report.json").read_text(encoding="utf-8"))
    assert list(rep["conditions"]) == ["B0", "IMG", "G1"]
    img = rep["conditions"]["IMG"]["pred_new"]
    assert img["main_view"] == "t0" and set(img["samples"]) == {"t1_s1", "t0"}
    b0_vs = rep["compare"]["B0_vs_IMG/pred_new"]
    assert set(b0_vs) == {"t1_s1~b0", "t0~b0"} and b0_vs["t0~b0"]["n"] == 3
    g1_vs = rep["compare"]["G1_vs_IMG/pred_new"]
    assert g1_vs["t0"]["diff"] == 0.0 and g1_vs["t0"]["mcnemar_p"] == 1.0 and "ci_low" in g1_vs["t0"]
    md = (runs[0] / "eval" / "verdict_report.md").read_text(encoding="utf-8")
    assert "| B0 |" in md and "So với IMG" in md


def test_report_without_outputs_returns_2(mini_sciver, tmp_path):
    cfg = _cfg(mini_sciver, tmp_path, "S-oracle", ["IMG"])
    assert cli.resolve_stage("report")(cfg, argparse.Namespace()) == 2
