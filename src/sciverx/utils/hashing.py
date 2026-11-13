"""Hash ổn định cho cấu hình, danh sách ID, nội dung và file."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    """JSON chuẩn tắc: khóa sắp xếp, không khoảng trắng thừa, giữ Unicode."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def ids_hash(ids: list[str]) -> str:
    """Hash danh sách ID theo thứ tự đã cho (thứ tự là một phần của tập chạy)."""
    return sha256_text("\n".join(ids))
