"""Bước 2a: xem cấu trúc dữ liệu thật trước khi dựng bảng chuẩn.

In ra các trường của 1 query, các khóa của paper JSON và các khóa của tables[item].
Nếu tables[item] đã chứa nội dung bảng (html/text/cells...), s03 sẽ dùng luôn
thay vì tải HTML arXiv.

  python src/s02_inspect_tables.py --data_root data/SciVer --samples work/samples.json
"""
import argparse
import json

from common import PathResolver, evidence_items, load_json, load_paper, table_entry


def short(v, n=300):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s[:n] + ("..." if len(s) > n else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--samples", default="work/samples.json")
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()

    res = PathResolver(args.data_root)
    samples = load_json(args.samples)
    q = samples[0]
    print("== Các trường của query:")
    for k, v in q.items():
        print(f"  {k}: {short(v, 120)}")

    for q in samples[: args.n]:
        paper = load_paper(res, q)
        print("\n== paper_path:", q["paper_path"])
        print("   khóa paper:", list(paper.keys()))
        tabs = paper.get("tables")
        print("   kiểu tables:", type(tabs).__name__,
              "| khóa ví dụ:", list(tabs.keys())[:5] if isinstance(tabs, dict) else len(tabs))
        for t, item, img in evidence_items(q):
            if t != "table":
                continue
            entry = table_entry(paper, item)
            print(f"   -- tables[{item!r}] có các trường:", list(entry.keys()))
            for k, v in entry.items():
                print(f"      {k}: {short(v)}")
            print("      image_path:", img)


if __name__ == "__main__":
    main()
