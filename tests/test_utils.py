import datetime as dt

import pytest

from sciverx.schemas import ClaimInput
from sciverx.utils import hashing, io, manifest


def test_jsonl_roundtrip_models_and_unicode(tmp_path):
    p = tmp_path / "a" / "claims.jsonl"
    recs = [ClaimInput(claim_id=f"validation-{i}", split="validation", paper_id="p",
                       text="Độ chính xác 93.2%") for i in range(3)]
    assert io.write_jsonl(p, recs) == 3
    assert "Độ chính xác" in p.read_text(encoding="utf-8")
    assert io.read_jsonl(p, ClaimInput) == recs


def test_read_jsonl_reports_bad_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"a": 1}\n{bad\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        io.read_jsonl(p)


def test_hash_is_order_independent_for_keys():
    assert hashing.sha256_obj({"a": 1, "b": 2}) == hashing.sha256_obj({"b": 2, "a": 1})
    assert hashing.ids_hash(["a", "b"]) != hashing.ids_hash(["b", "a"])


def test_file_hash(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert hashing.sha256_file(p) == hashing.sha256_bytes(b"abc")


def test_run_id_format():
    cfg = {"m3": {"rrf_constant": 60}}
    rid = manifest.make_run_id("x2_retrieval", "validation", cfg, date=dt.date(2026, 11, 13))
    assert rid.startswith("20261113_x2_retrieval_validation_")
    assert len(rid.rsplit("_", 1)[1]) == 8
    other = manifest.make_run_id("x2_retrieval", "validation", {"m3": {"rrf_constant": 10}},
                                 date=dt.date(2026, 11, 13))
    assert rid != other


def test_manifest_records_ids_and_versions():
    m = manifest.build_manifest("r1", {"a": 1}, ["validation-1", "validation-2"])
    assert m.claim_ids_hash == hashing.ids_hash(["validation-1", "validation-2"])
    assert m.versions["python"]
    assert "pydantic" in m.versions


def test_run_dir_never_overwritten(tmp_path):
    m = manifest.build_manifest("r1", {"a": 1}, ["x"])
    d = manifest.open_run_dir(tmp_path, m)
    assert (d / "manifest.json").exists()
    # cùng hash → chạy tiếp được
    assert manifest.open_run_dir(tmp_path, manifest.build_manifest("r1", {"a": 1}, ["x"])) == d
    # khác hash cùng run_id → từ chối
    with pytest.raises(manifest.RunExistsError):
        manifest.open_run_dir(tmp_path, manifest.build_manifest("r1", {"a": 2}, ["x"]))
    with pytest.raises(manifest.RunExistsError):
        manifest.open_run_dir(tmp_path, m, resume=False)
