"""Đọc/ghi JSONL cho dict hoặc model Pydantic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def write_jsonl(path: str | Path, records: Iterable[BaseModel | dict]) -> int:
    """Ghi từng bản ghi một dòng (UTF-8, không escape tiếng Việt). Trả số dòng."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            if isinstance(rec, BaseModel):
                line = rec.model_dump_json()
            else:
                line = json.dumps(rec, ensure_ascii=False)
            f.write(line + "\n")
            n += 1
    return n


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: JSON không hợp lệ: {exc}") from exc


def read_jsonl(path: str | Path, model: type[M] | None = None) -> list:
    """Đọc toàn bộ file; nếu có `model` thì kiểm định từng dòng."""
    rows = list(iter_jsonl(path))
    if model is None:
        return rows
    return [model.model_validate(r) for r in rows]


def write_json(path: str | Path, obj: BaseModel | dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = obj.model_dump_json(indent=2) if isinstance(obj, BaseModel) else json.dumps(
        obj, ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
