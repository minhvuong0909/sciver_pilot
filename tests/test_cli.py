from pathlib import Path

import pytest

from sciverx import cli
from sciverx.config import deep_merge, load_config

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "base.yaml"


def test_base_config_matches_spec():
    cfg = load_config(BASE)
    assert cfg["split"] == "validation"
    assert cfg["m1"]["units"]["max_wordpieces"] == 384
    assert cfg["m3"]["rrf_constant"] == 60
    assert [s["name"] for s in cfg["verifier"]["samples"]] == ["t1_s1", "t1_s2", "t1_s3", "t0"]


def test_config_inherits_base(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text(f"base: {BASE.as_posix()}\nm3: {{rrf_constant: 10}}\n", encoding="utf-8")
    cfg = load_config(exp)
    assert cfg["m3"]["rrf_constant"] == 10
    assert cfg["m3"]["k_units"] == 50  # giữ giá trị từ base


def test_deep_merge_does_not_mutate():
    a = {"x": {"y": 1, "z": 2}}
    b = {"x": {"y": 3}}
    out = deep_merge(a, b)
    assert out == {"x": {"y": 3, "z": 2}}
    assert a == {"x": {"y": 1, "z": 2}}


def test_every_stage_in_spec_is_registered():
    spec_stages = {"audit", "select", "sources", "gold", "extract", "normalize", "quality",
                   "cellmap", "units", "index", "retrieve", "align", "numeric", "package",
                   "verify", "evaluate", "report", "annotate-export", "annotate-import"}
    assert set(cli.STAGES) == spec_stages


def test_unimplemented_stage_returns_3(monkeypatch):
    monkeypatch.setitem(cli.STAGES, "audit", "sciverx.khong_ton_tai:run")
    assert cli.main(["audit", "--config", str(BASE)]) == 3


def test_test_split_requires_frozen_config():
    assert cli.main(["retrieve", "--config", str(BASE), "--split", "test"]) == 2


def test_limit_capped_at_20():
    assert cli.main(["retrieve", "--config", str(BASE), "--limit", "21"]) == 2


def test_stage_receives_config(monkeypatch):
    seen = {}

    def fake_run(cfg, args):
        seen["cfg"] = cfg
        return 0

    import types, sys
    mod = types.ModuleType("sciverx_fake_stage")
    mod.run = fake_run
    monkeypatch.setitem(sys.modules, "sciverx_fake_stage", mod)
    monkeypatch.setitem(cli.STAGES, "audit", "sciverx_fake_stage:run")
    assert cli.main(["audit", "--config", str(BASE), "--limit", "5"]) == 0
    assert seen["cfg"]["limit"] == 5


def test_unknown_stage_rejected():
    with pytest.raises(SystemExit):
        cli.main(["khong-co", "--config", str(BASE)])
