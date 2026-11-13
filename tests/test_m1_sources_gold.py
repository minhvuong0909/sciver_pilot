import argparse
import csv

from sciverx import cli
from sciverx.m1_prepare import sources_path, tables_path, work_dir
from sciverx.m1_prepare.gold_tables import audit_rows, build_gold_table
from sciverx.schemas import TableRecord
from sciverx.utils.io import read_jsonl

NS = argparse.Namespace()


def _cfg(mini_sciver, tmp_path):
    return {"split": "validation", "seed_global": 2026,
            "paths": {"data_root": str(mini_sciver), "work": str(tmp_path / "w")},
            "m1": {"gold": {"audit_random_n": 2}}}


def _stage(name, cfg):
    return cli.resolve_stage(name)(cfg, NS)


def test_sources_list_every_item_and_image_presence(mini_sciver, tmp_path):
    cfg = _cfg(mini_sciver, tmp_path)
    assert _stage("select", cfg) == 0
    assert _stage("sources", cfg) == 0
    srcs = {s["paper_id"]: s for s in read_jsonl(sources_path(cfg))}
    a = srcs["2409.00001v1"]
    assert [t["table_id"] for t in a["tables"]] == ["1", "2"]
    assert a["tables"][0]["image_path"] == "images/2409.00001v1-Table1-1.png"
    assert a["tables"][1]["image_path"] is None          # bảng không có ảnh trong dataset
    assert [s["section_id"] for s in a["sections"]] == ["1", "4", "4.1"]
    assert a["charts"][0]["chart_id"] == "3"
    # không có trường cấp claim hay trường gán nhãn của paper
    text = sources_path(cfg).read_text(encoding="utf-8")
    for leak in ("label", "extract_table_info", "related_sentences", "LEAK"):
        assert leak not in text


def test_gold_g0_then_normalize(mini_sciver, tmp_path):
    cfg = _cfg(mini_sciver, tmp_path)
    for st in ("select", "sources", "gold", "normalize"):
        assert _stage(st, cfg) == 0
    g0 = read_jsonl(tables_path(cfg, "G0"), TableRecord)
    assert len(g0) == 4 and all(r.status == "ok" and r.extractor == "gold_json" for r in g0)
    assert g0[0].n_header_rows == 1 and g0[0].caption == "Table 1: Results."
    assert g0[0].source_version == "1"
    g1 = read_jsonl(tables_path(cfg, "G1"), TableRecord)
    assert [r.condition for r in g1] == ["G1"] * 4
    rows = list(csv.DictReader((work_dir(cfg) / "annotation" / "gold_audit.csv").open(encoding="utf-8")))
    assert rows and {r["reason"] for r in rows} <= {"random_small", "random_medium", "random_large"}


def test_build_gold_table_statuses():
    rec, info = build_gold_table("p1v2", "1", {"table_html": "<figure><img src='x.png'/></figure>",
                                               "capture": "Table 1"})
    assert rec.status == "gold_unavailable" and rec.cells == []
    bad = "<table><tr><td colspan='2' rowspan='2'>a</td></tr><tr><td>b</td><td>c</td></tr></table>"
    rec, info = build_gold_table("p1v2", "2", {"table_html": bad, "capture": ""})
    assert rec.status == "ok"  # b, c nằm sau vùng bị phủ → không chồng
    rec, info = build_gold_table("p1v2", "3", {"table_html": "<table><tr><td>x</td></tr><tr><td>y</td>"
                                               "</tr><table><tr><td>z</td></tr></table></table>"})
    assert rec.status == "ok" and rec.source_version == "2"


def test_audit_rows_include_errors_and_multi_part():
    ok, info_ok = build_gold_table("p", "1", {"table_html": "<table><tr><td>a</td></tr></table>"})
    multi, info_m = build_gold_table("p", "2", {"table_html": "<figure><table><tr><td>a</td></tr></table>"
                                               "<table><tr><td>b</td></tr></table></figure>"})
    miss, info_x = build_gold_table("p", "3", {"table_html": ""})
    rows = audit_rows([ok, multi, miss], [info_ok, info_m, info_x], {}, n_random=5)
    reasons = {r["table_id"]: r["reason"] for r in rows}
    assert reasons["2"] == "multi_part" and reasons["3"] == "status"
    assert reasons["1"].startswith("random_")


def test_quality_and_cellmap_stages_with_fake_e0(mini_sciver, tmp_path):
    cfg = _cfg(mini_sciver, tmp_path)
    for st in ("select", "sources", "gold"):
        _stage(st, cfg)
    g0 = read_jsonl(tables_path(cfg, "G0"), TableRecord)
    # E0 giả: bản sao G0 đổi điều kiện, một ô bị đọc sai
    e0 = []
    for r in g0:
        cells = [c.model_copy(update={"text_raw": "92.1"}) if c.text_raw == "92.4" else c for c in r.cells]
        e0.append(r.model_copy(update={"condition": "E0", "extractor": "fake",
                                       "table_record_id": TableRecord.make_id(r.paper_id, r.table_id, "E0"),
                                       "cells": cells}))
    from sciverx.utils.io import write_jsonl
    write_jsonl(tables_path(cfg, "E0"), e0)
    for st in ("normalize", "quality", "cellmap"):
        assert _stage(st, cfg) == 0
    q = list(csv.DictReader((work_dir(cfg) / "table_quality.csv").open(encoding="utf-8")))
    assert len(q) == 8 and all(float(r["teds_struct"]) == 1.0 for r in q)
    assert all(float(r["teds"]) < 1.0 for r in q)
    maps = read_jsonl(work_dir(cfg) / "cell_mapping.jsonl")
    assert len(maps) == 4 and all(m["unmatched_g"] == [] for m in maps)
