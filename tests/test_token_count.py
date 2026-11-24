"""Stage count-tokens (không GPU) và ước lượng visual token của Qwen2.5-VL."""

import argparse

import pytest

from sciverx.m1_prepare.units import whitespace_counter as wc
from sciverx.utils import manifest
from sciverx.utils.io import read_jsonl
from sciverx.verifier import token_count as T


def test_smart_resize_rounds_to_patch_and_respects_limits():
    assert T.smart_resize(224, 224) == (224, 224)              # đã chia hết cho 28
    assert T.smart_resize(230, 500) == (224, 504)               # làm tròn tới bội số 28
    h, w = T.smart_resize(10, 10)                               # quá nhỏ → phóng tới min_pixels
    assert h * w >= T.MIN_PIXELS and h % 28 == 0 and w % 28 == 0
    h, w = T.smart_resize(20000, 20000)                         # quá lớn → thu về ≤ max_pixels
    assert h * w <= T.MAX_PIXELS and h % 28 == 0
    assert T.visual_tokens(224, 224) == 8 * 8 + 2


def test_summarize_counts_over_limit():
    rows = [{"claim_id": f"c{i}", "total_tokens_est": t} for i, t in enumerate([100, 200, 23000])]
    s = T.summarize(rows, max_model_len=32768, max_tokens=10240)
    assert s["input_limit"] == 22528 and s["n_over_limit"] == 1 and s["over_limit_claims"] == ["c2"]
    assert s["max"] == 23000 and s["median"] == 200


def test_count_tokens_stage_oracle(mini_sciver, tmp_path):
    from test_verify_stage import _cfg       # thư mục tests/ nằm trên sys.path (không có __init__.py)
    cfg = _cfg(mini_sciver, tmp_path, "S-oracle", ["G1"])
    sizes = {}

    def fake_image(p):
        sizes[str(p)] = True
        return b""

    assert T.run(cfg, argparse.Namespace(), count=wc, load_image=fake_image) == 0
    rows = read_jsonl(tmp_path / "w" / "validation" / "token_counts" / "t_S-oracle_G1.jsonl")
    assert len(rows) == 3
    assert all(r["text_tokens"] > 0 and r["total_tokens_est"] == r["text_tokens"] + T.TEMPLATE_MARGIN
               for r in rows)                                  # ảnh giả → 0 visual token


def test_count_tokens_refuses_align(mini_sciver, tmp_path):
    cfg = {"verifier": {"setting": "S-align4", "table_conditions": ["G1"]}}
    assert T.run(cfg, argparse.Namespace(), count=wc) == 2


def test_update_manifest_keeps_frozen_fields(tmp_path):
    d = manifest.open_run_dir(tmp_path, manifest.build_manifest("r1", {"a": 1}, ["x"]))
    m = manifest.update_manifest(d, duration_s=1.5, error_counts={"ok": 2}, runtime={"vllm_use_v1": True})
    assert m.duration_s == 1.5 and m.runtime["vllm_use_v1"] is True and m.claim_ids == ["x"]
    with pytest.raises(ValueError):
        manifest.update_manifest(d, config_hash="khac")
