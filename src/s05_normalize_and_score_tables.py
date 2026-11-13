"""Bước 4: tạo 4 phiên bản bảng cho mỗi bằng chứng và đo chất lượng trích bảng.

  G0 = bảng chuẩn thô        G1 = bảng chuẩn sau chuẩn hóa
  E0 = bảng Docling thô      E1 = bảng Docling sau chuẩn hóa
Cùng một bộ chuẩn hóa (tables.normalize_table) cho cả G và E.

Đầu ra:
  work/tables/<G0|G1|E0|E1>/<key>.json
  work/normalize_log.jsonl        (mọi ô bị đổi, trước/sau, luật)
  work/table_quality.csv          (TEDS, TEDS-Struct, độ khớp số giữa E và G)

  python src/s05_normalize_and_score_tables.py --out_dir work
"""
import argparse
import csv
import glob
import os
from collections import Counter

from common import load_json, save_json, save_jsonl
from tables import normalize_table, numeric_tokens
from teds import teds


def num_match(pred, gold):
    """Precision/recall của multiset số (bảng trích so với bảng chuẩn)."""
    p, g = Counter(numeric_tokens(pred)), Counter(numeric_tokens(gold))
    inter = sum((p & g).values())
    prec = inter / sum(p.values()) if p else 0.0
    rec = inter / sum(g.values()) if g else (1.0 if not p else 0.0)
    return round(prec, 4), round(rec, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="work")
    ap.add_argument("--extractor", default="docling")
    args = ap.parse_args()

    gold_files = sorted(glob.glob(os.path.join(args.out_dir, "gold", "*.json")))
    if not gold_files:
        raise SystemExit("Chưa có bảng chuẩn: chạy s03 trước.")
    logs, quality = [], []
    for gf in gold_files:
        g = load_json(gf)
        key = g["key"]
        ef = os.path.join(args.out_dir, f"extracted_{args.extractor}", key + ".json")
        e = load_json(ef) if os.path.exists(ef) else {"table": {"grid": [], "n_header_rows": 0,
                                                                  "source": args.extractor},
                                                        "status": "missing_file"}
        g0, e0 = g["table"], e["table"]
        g1, glog = normalize_table(g0)
        e1, elog = normalize_table(e0)
        for cond, tab in (("G0", g0), ("G1", g1), ("E0", e0), ("E1", e1)):
            save_json({"key": key, "caption": g["caption"], "table": tab},
                      os.path.join(args.out_dir, "tables", cond, key + ".json"))
        logs += [{"key": key, "side": "G", **x} for x in glog]
        logs += [{"key": key, "side": "E", **x} for x in elog]

        row = {"key": key, "extract_status": e.get("status", ""),
               "gold_ok_manual": "", "gold_rows": len(g0["grid"]), "ext_rows": len(e0["grid"]),
               "gold_cols": len(g0["grid"][0]) if g0["grid"] else 0,
               "ext_cols": len(e0["grid"][0]) if e0["grid"] else 0}
        for tag, a, b in (("raw", e0, g0), ("norm", e1, g1)):
            row[f"teds_{tag}"] = round(teds(a["grid"], b["grid"]), 4)
            row[f"teds_struct_{tag}"] = round(teds(a["grid"], b["grid"], structure_only=True), 4)
            row[f"num_prec_{tag}"], row[f"num_rec_{tag}"] = num_match(a, b)
        row["n_norm_changes_G"], row["n_norm_changes_E"] = len(glog), len(elog)
        quality.append(row)
        print(f"{key}: TEDS raw={row['teds_raw']} norm={row['teds_norm']} | num_rec raw={row['num_rec_raw']}")

    save_jsonl(logs, os.path.join(args.out_dir, "normalize_log.jsonl"))
    with open(os.path.join(args.out_dir, "table_quality.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(quality[0].keys()))
        w.writeheader(); w.writerows(quality)
    n = len(quality)
    mean = lambda k: sum(r[k] for r in quality) / n  # noqa: E731
    print(f"\n{n} bảng | TEDS raw {mean('teds_raw'):.3f} | TEDS norm {mean('teds_norm'):.3f} | "
          f"TEDS-Struct {mean('teds_struct_raw'):.3f} | num_rec {mean('num_rec_raw'):.3f}")
    print("Luật chuẩn hóa được dùng:", dict(Counter(r for x in logs for r in x["rules"])))


if __name__ == "__main__":
    main()
