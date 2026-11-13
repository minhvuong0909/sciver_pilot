"""Tiện ích dùng chung: đọc dataset SciVer, tìm đường dẫn file, lấy context/caption.

Tên trường dựa theo code gốc của SciVer (utils/input_processing.py):
  query: claim, claim_type, label, paper_path, section,
         type, item, image_path                        (direct / analytical)
         item1_type, item1, item1_path, item2_type, item2, item2_path   (parallel / sequential)
  paper JSON: sections[{section_id, section_name, text}],
              image_paths[id]{caption}, tables[id]{capture}
Nếu dataset thật khác tên trường, chỉ cần sửa ở file này.
"""
import json
import os
import re
from functools import lru_cache

PAIR_TYPES = ("parallel", "sequential")
SINGLE_TYPES = ("direct", "analytical")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class PathResolver:
    """Tìm file trong thư mục dataset dù đường dẫn trong JSON là tương đối theo gốc nào."""

    def __init__(self, data_root):
        self.data_root = os.path.abspath(data_root)
        self._index = None

    def _build_index(self):
        idx = {}
        for d, _, files in os.walk(self.data_root):
            for fn in files:
                idx.setdefault(fn, []).append(os.path.join(d, fn))
        self._index = idx

    def resolve(self, p):
        if not p:
            return None
        cands = [p,
                 os.path.join(self.data_root, p),
                 os.path.join(os.path.dirname(self.data_root), p)]
        for c in cands:
            if os.path.exists(c):
                return os.path.abspath(c)
        if self._index is None:
            self._build_index()
        hits = self._index.get(os.path.basename(p), [])
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:  # chọn file có phần đuôi đường dẫn khớp nhiều nhất
            tail = p.replace("\\", "/").split("/")
            best = max(hits, key=lambda h: sum(a == b for a, b in zip(reversed(h.split(os.sep)), reversed(tail))))
            return best
        raise FileNotFoundError(f"Không tìm thấy file: {p} (data_root={self.data_root})")


@lru_cache(maxsize=512)
def _load_paper_cached(path):
    return load_json(path)


def load_paper(resolver, query):
    return _load_paper_cached(resolver.resolve(query["paper_path"]))


def evidence_items(query):
    """Trả về danh sách bằng chứng [(type, item_id, image_path)] theo thứ tự như code gốc."""
    if query["claim_type"] in SINGLE_TYPES:
        return [(query["type"], query["item"], query["image_path"])]
    return [(query["item1_type"], query["item1"], query["item1_path"]),
            (query["item2_type"], query["item2"], query["item2_path"])]


def _get_item(container, item_id):
    if isinstance(container, dict):
        if item_id in container:
            return container[item_id]
        if str(item_id) in container:
            return container[str(item_id)]
    if isinstance(container, list):
        try:
            return container[int(item_id)]
        except (ValueError, IndexError, TypeError):
            pass
    raise KeyError(f"Không có item {item_id!r}")


def table_entry(paper, item_id):
    return _get_item(paper["tables"], item_id)


def caption_of(paper, item_type, item_id):
    if item_type == "chart":
        return _get_item(paper["image_paths"], item_id).get("caption", "")
    t = table_entry(paper, item_id)
    return t.get("capture", t.get("caption", ""))


def prepare_context(paper, section_list):
    """Giống hệt code gốc: lấy toàn bộ section cấp 1 chứa bằng chứng."""
    top = []
    for s in section_list:
        k = str(s).split(".")[0]
        if k not in top:
            top.append(k)
    ctx = ""
    for sec in paper["sections"]:
        if str(sec["section_id"]).split(".")[0] in top:
            ctx += sec["section_name"] + ":\n" + sec["text"] + "\n"
    return ctx


def qid_of(query, idx=None):
    for k in ("id", "qid", "example_id", "uid"):
        if k in query:
            return str(query[k])
    return f"q{idx:05d}" if idx is not None else None


def arxiv_id_from(text):
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", text or "")
    return m.group(1) if m else None


def label_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "entailed", "yes", "1", "supported"):
        return True
    if s in ("false", "refuted", "no", "0"):
        return False
    raise ValueError(f"Nhãn không rõ: {v!r}")
