"""Bước 2b: dựng bảng chuẩn (G0) cho mọi bảng bằng chứng trong các mẫu pilot.

Thứ tự ưu tiên:
  1. Nếu tables[item] trong paper JSON đã có HTML bảng (trường chứa "<table") -> dùng luôn.
  2. Nếu không: tải bản HTML của bài trên arXiv (https://arxiv.org/html/<id>),
     tìm bảng theo id LaTeXML (vd "S4.T2") hoặc theo caption giống nhất.
Kết quả: work/gold/<qid>__<item>.json  + work/gold_review.csv để NHÓM KIỂM TAY
(mở ảnh bảng và bảng chuẩn cạnh nhau, điền cột ok = 1/0).

  python src/s03_build_gold.py --data_root data/SciVer --samples work/samples.json --out_dir work
"""
import argparse
import csv
import difflib
import os
import re
import time

import requests
from bs4 import BeautifulSoup

from common import (PathResolver, arxiv_id_from, evidence_items, load_json, load_paper,
                    save_json, table_entry, caption_of)
from tables import make_table

HTML_KEYS = ("table_html", "html", "content", "table", "text", "latex_html", "body")


def html_in_entry(entry):
    for k in HTML_KEYS:
        v = entry.get(k)
        if isinstance(v, str) and "<table" in v.lower():
            return v, k
    for k, v in entry.items():
        if isinstance(v, str) and "<table" in v.lower():
            return v, k
    return None, None


def paper_arxiv_id(paper, query):
    for k in ("arxiv_id", "paper_id", "id", "url", "link"):
        aid = arxiv_id_from(str(paper.get(k, "")))
        if aid:
            return aid
    return arxiv_id_from(query.get("paper_path", ""))


_cache = {}


def fetch_arxiv_html(aid, cache_dir, sleep=3.0):
    path = os.path.join(cache_dir, f"{aid}.html")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    url = f"https://arxiv.org/html/{aid}"
    r = requests.get(url, timeout=60, headers={"User-Agent": "sciver-pilot/0.1 (research)"})
    time.sleep(sleep)  # lịch sự với arXiv
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} cho {url}")
    os.makedirs(cache_dir, exist_ok=True)
    open(path, "w", encoding="utf-8").write(r.text)
    return r.text


def find_table_in_arxiv(html, item_id, caption):
    soup = BeautifulSoup(html, "lxml")
    # 1) theo id LaTeXML
    if isinstance(item_id, str) and item_id:
        el = soup.find(id=item_id)
        if el is not None and el.find("table") is not None:
            return str(el), "arxiv_id_match", 1.0
    # 2) theo caption
    figs = soup.select("figure.ltx_table") or [t.find_parent("figure") or t for t in soup.find_all("table")]
    best, best_s = None, 0.0
    cap = re.sub(r"\s+", " ", caption or "").strip().lower()
    for f in figs:
        fc = f.find("figcaption")
        txt = re.sub(r"\s+", " ", fc.get_text(" ", strip=True) if fc else "").lower()
        s = difflib.SequenceMatcher(None, cap[:300], txt[:300]).ratio() if cap else 0.0
        if s > best_s and f.find("table") is not None:
            best, best_s = f, s
    if best is not None:
        return str(best), "arxiv_caption_match", round(best_s, 3)
    return None, "not_found", 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--samples", default="work/samples.json")
    ap.add_argument("--out_dir", default="work")
    ap.add_argument("--no_arxiv", action="store_true", help="không tải arXiv, chỉ dùng HTML trong JSON")
    args = ap.parse_args()

    res = PathResolver(args.data_root)
    samples = load_json(args.samples)
    gold_dir = os.path.join(args.out_dir, "gold")
    cache_dir = os.path.join(args.out_dir, "arxiv_html_cache")
    review = []
    for q in samples:
        paper = load_paper(res, q)
        for t, item, img in evidence_items(q):
            if t != "table":
                continue
            key = f"{q['_qid']}__{item}"
            cap = caption_of(paper, t, item)
            entry = table_entry(paper, item)
            html, method, score = None, "", 0.0
            src_html, k = html_in_entry(entry)
            if src_html:
                html, method, score = src_html, f"json_field:{k}", 1.0
            elif not args.no_arxiv:
                aid = paper_arxiv_id(paper, q)
                if aid:
                    try:
                        page = fetch_arxiv_html(aid, cache_dir)
                        html, method, score = find_table_in_arxiv(page, item, cap)
                    except Exception as e:  # noqa: BLE001
                        method = f"arxiv_error:{e}"
                else:
                    method = "no_arxiv_id"
            tab = make_table(html, "gold") if html else {"grid": [], "n_header_rows": 0, "source": "gold"}
            save_json({"key": key, "caption": cap, "method": method, "match_score": score,
                       "image_path": img, "table": tab, "raw_html": html}, os.path.join(gold_dir, key + ".json"))
            review.append({"key": key, "qid": q["_qid"], "item": item, "image_path": img,
                           "method": method, "match_score": score,
                           "n_rows": len(tab["grid"]), "n_cols": len(tab["grid"][0]) if tab["grid"] else 0,
                           "ok": "", "note": ""})
            print(f"{key}: {method} (score={score}) {len(tab['grid'])} dòng")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "gold_review.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(review[0].keys()) if review else ["key"])
        w.writeheader(); w.writerows(review)
    bad = [r for r in review if not r["n_rows"] or r["match_score"] < 0.6]
    print(f"\nTổng {len(review)} bảng; {len(bad)} bảng cần xem kỹ (không tìm thấy hoặc caption khớp < 0.6).")
    print("Mở work/gold_review.csv, so ảnh với bảng chuẩn, điền ok=1/0. Sửa tay file JSON nếu cần.")


if __name__ == "__main__":
    main()
