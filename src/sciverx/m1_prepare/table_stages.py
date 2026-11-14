"""Các stage trên bảng: normalize (M1.4), quality (M1.5), cellmap (M1.5)."""

from __future__ import annotations

import csv

from sciverx.m1_prepare import tables_path, work_dir
from sciverx.schemas import TableRecord
from sciverx.tables.cell_mapping import map_cells
from sciverx.tables.normalize import normalize_table
from sciverx.tables.numeric import numeric_precision_recall
from sciverx.tables.teds import teds
from sciverx.utils.io import read_jsonl, write_jsonl


def _load(cfg: dict, cond: str) -> list[TableRecord]:
    path = tables_path(cfg, cond)
    return read_jsonl(path, TableRecord) if path.exists() else []


def run_normalize(cfg: dict, args) -> int:
    log: list[dict] = []
    for src, dst in (("G0", "G1"), ("E0", "E1")):
        recs = _load(cfg, src)
        if not recs:
            print(f"[normalize] bỏ qua {src}: chưa có file")
            continue
        out = []
        for r in recs:
            res = normalize_table(r)
            out.append(res.record)
            log.extend(res.log)
        write_jsonl(tables_path(cfg, dst), out)
        print(f"[normalize] {src} → {dst}: {len(out)} bảng")
    write_jsonl(work_dir(cfg) / "normalize_log.jsonl", log)
    rules: dict[str, int] = {}
    for e in log:
        for rule in e["rules"]:
            rules[rule] = rules.get(rule, 0) + 1
    print(f"[normalize] {len(log)} ô đổi; phân bố luật: {rules}")
    return 0


def _pairs(cfg: dict, e_cond: str, g_cond: str):
    g = {(r.paper_id, r.table_id): r for r in _load(cfg, g_cond)}
    for e in _load(cfg, e_cond):
        gold = g.get((e.paper_id, e.table_id))
        if gold is not None and gold.status == "ok":
            yield e, gold


def run_quality(cfg: dict, args) -> int:
    rows = []
    for e_cond, g_cond in (("E0", "G0"), ("E1", "G1")):
        for e, g in _pairs(cfg, e_cond, g_cond):
            row = {"paper_id": e.paper_id, "table_id": e.table_id, "e_condition": e_cond,
                   "g_condition": g_cond, "e_status": e.status}
            if e.status == "ok":
                prec, rec = numeric_precision_recall(e, g)
                row.update(teds=round(teds(e, g), 4), teds_struct=round(teds(e, g, True), 4),
                           num_prec=round(prec, 4), num_rec=round(rec, 4))
            rows.append(row)
    out = work_dir(cfg) / "table_quality.csv"
    cols = ["paper_id", "table_id", "e_condition", "g_condition", "e_status", "teds",
            "teds_struct", "num_prec", "num_rec"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[quality] {len(rows)} cặp bảng → {out}")
    return 0


def run_cellmap(cfg: dict, args) -> int:
    maps = [map_cells(e, g) for e, g in _pairs(cfg, "E1", "G1") if e.status == "ok"]
    write_jsonl(work_dir(cfg) / "cell_mapping.jsonl", maps)
    n_g = sum(len(m.unmatched_g) + sum(len(p["g_cell_ids"]) for p in m.pairs) for m in maps)
    n_um = sum(len(m.unmatched_g) for m in maps)
    print(f"[cellmap] {len(maps)} bảng; ô G không ánh xạ được: {n_um}/{n_g}")
    return 0
