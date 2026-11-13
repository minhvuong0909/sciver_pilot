"""Bước 1: chọn mẫu pilot (mặc định 40 = 4 tập con x 10, mỗi tập con 5 Entailed / 5 Refuted).

Chỉ lấy claim có bằng chứng là bảng:
  - direct / analytical: type == "table"
  - parallel / sequential: cả hai item là table (ưu tiên); nếu thiếu thì cho phép 1 bảng + 1 biểu đồ
    (có cờ mixed=True để ghi chú trong báo cáo).

Ví dụ:
  python src/s01_select_samples.py --data_root data/SciVer --data_file data/SciVer/validation.json \
      --per_subset 10 --seed 2026 --out work/samples.json
"""
import argparse
import os
import random
from collections import Counter

from common import (PAIR_TYPES, evidence_items, label_bool, load_json, qid_of, save_json)


def table_status(q):
    types = [t for t, _, _ in evidence_items(q)]
    n_tab = sum(t == "table" for t in types)
    if n_tab == len(types):
        return "all_table"
    if n_tab > 0:
        return "mixed"
    return "no_table"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--data_file", required=True, help="validation (dev) hoặc test (báo cáo cuối) của SciVer")
    ap.add_argument("--split", default=None, help="tiền tố ID, vd val/test (mặc định lấy theo tên file)")
    ap.add_argument("--all", action="store_true",
                    help="lấy TOÀN BỘ claim thỏa tiêu chí (chạy trên cả tập của paper), bỏ qua --per_subset")
    ap.add_argument("--per_subset", type=int, default=10, help="chỉ dùng khi không có --all")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--allow_mixed", action="store_true",
                    help="cho phép claim cần cả bảng và biểu đồ (mặc định: KHÔNG, chỉ bảng + đoạn văn)")
    ap.add_argument("--out", default="work/samples.json")
    args = ap.parse_args()

    data = load_json(args.data_file)
    split = args.split or os.path.splitext(os.path.basename(args.data_file))[0]
    rng = random.Random(args.seed)
    for i, q in enumerate(data):
        q["_qid"] = f"{split}-{qid_of(q, i)}"
        q["_split"] = split
        q["_label"] = label_bool(q["label"])
        q["_table_status"] = table_status(q)

    print("Tổng số claim:", len(data))
    print("Theo (claim_type, table_status):",
          dict(Counter((q["claim_type"], q["_table_status"]) for q in data)))

    picked = []
    if args.all:
        ok = ("all_table", "mixed") if args.allow_mixed else ("all_table",)
        picked = [q for q in data if q["_table_status"] in ok]
        for q in picked:
            q["_mixed"] = q["_table_status"] == "mixed"
        save_json(picked, args.out)
        print(f"[--all] {len(picked)}/{len(data)} claim thỏa tiêu chí -> {args.out}")
        print("Phân bố (claim_type, nhãn):", dict(Counter((q["claim_type"], q["_label"]) for q in picked)))
        return
    half = args.per_subset // 2
    for ct in ("direct", "parallel", "sequential", "analytical"):
        for lab in (True, False):
            need = half if lab else args.per_subset - half
            pool = [q for q in data if q["claim_type"] == ct and q["_label"] == lab
                    and q["_table_status"] == "all_table"]
            rng.shuffle(pool)
            chosen = pool[:need]
            if len(chosen) < need and ct in PAIR_TYPES and args.allow_mixed:
                extra = [q for q in data if q["claim_type"] == ct and q["_label"] == lab
                         and q["_table_status"] == "mixed"]
                rng.shuffle(extra)
                chosen += extra[:need - len(chosen)]
            if len(chosen) < need:
                print(f"[CẢNH BÁO] {ct}/{'Entailed' if lab else 'Refuted'}: chỉ có {len(chosen)}/{need} mẫu")
            picked += chosen

    for q in picked:
        q["_mixed"] = q["_table_status"] == "mixed"
    save_json(picked, args.out)
    print(f"Đã chọn {len(picked)} mẫu -> {args.out}")
    print("Phân bố:", dict(Counter((q["claim_type"], q["_label"]) for q in picked)))
    print("Số mẫu mixed (1 bảng + 1 biểu đồ):", sum(q["_mixed"] for q in picked))


if __name__ == "__main__":
    main()
