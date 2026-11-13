"""M1 — chuẩn bị nguồn và lập chỉ mục. Đường dẫn trung gian dùng chung cho các stage."""

from __future__ import annotations

from pathlib import Path


def work_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["work"]) / cfg["split"]


def tables_path(cfg: dict, condition: str) -> Path:
    return work_dir(cfg) / "tables" / f"{condition}.jsonl"


def units_path(cfg: dict, condition: str) -> Path:
    return work_dir(cfg) / "units" / f"{condition}.jsonl"


def sources_path(cfg: dict) -> Path:
    return work_dir(cfg) / "sources_manifest.jsonl"
