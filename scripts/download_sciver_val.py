"""Tải valset.json và các paper JSON mà validation dùng từ HuggingFace (chengyewang/SciVer).

KHÔNG tải testset.json: test bị khóa tới GĐ B4 (SPEC mục 2.1, 5.4).
Ảnh chỉ tải khi có --images (ảnh bằng chứng và ảnh bảng có sẵn trong dataset).

    python scripts/download_sciver_val.py --root data/raw/SciVer [--images]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/chengyewang/SciVer/resolve/main/"


def fetch(rel: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(BASE + rel, timeout=60) as r:
                data = r.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            time.sleep(2 ** attempt)
        except OSError:
            time.sleep(2 ** attempt)
    return False


def rel_from_dataset_path(p: str) -> str:
    """'./SciVer/papers/x.json' -> 'papers/x.json'."""
    p = p.replace("\\", "/")
    for prefix in ("./SciVer/", "SciVer/", "./"):
        if p.startswith(prefix):
            return p[len(prefix):]
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/SciVer")
    ap.add_argument("--images", action="store_true", help="ảnh bằng chứng của các claim validation")
    ap.add_argument("--table-images", action="store_true",
                    help="ảnh của mọi bảng trong các bài validation, nếu dataset có")
    args = ap.parse_args()
    root = Path(args.root)
    if not fetch("valset.json", root / "valset.json"):
        raise SystemExit("Không tải được valset.json")
    rows = json.loads((root / "valset.json").read_text(encoding="utf-8"))
    papers = sorted({rel_from_dataset_path(r["paper_path"]) for r in rows})
    missing = [p for p in papers if not fetch(p, root / p)]
    print(f"papers: {len(papers) - len(missing)}/{len(papers)} (thiếu {len(missing)})")
    if args.images:
        imgs = set()
        for r in rows:
            for k in ("image_path", "item1_path", "item2_path"):
                if r.get(k):
                    imgs.add(rel_from_dataset_path(r[k]))
        miss_img = [p for p in sorted(imgs) if not fetch(p, root / p)]
        print(f"images: {len(imgs) - len(miss_img)}/{len(imgs)} (thiếu {len(miss_img)})")
    if args.table_images:
        # Ảnh của MỌI bảng trong các bài validation mà dataset có (không chỉ ảnh bằng chứng),
        # tên theo mẫu images/{paperid}-Table{N}-1.png; 404 nghĩa là dataset không có ảnh đó.
        got = tried = 0
        for p in papers:
            paper = json.loads((root / p).read_text(encoding="utf-8"))
            pid = Path(p).stem
            for tid in paper.get("tables", {}):
                tried += 1
                got += fetch(f"images/{pid}-Table{tid}-1.png", root / "images" / f"{pid}-Table{tid}-1.png")
        print(f"table images: {got}/{tried}")


if __name__ == "__main__":
    main()
