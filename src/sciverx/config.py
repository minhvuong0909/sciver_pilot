"""Đọc cấu hình YAML, hỗ trợ kế thừa qua khóa `base` (SPEC mục 13)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Trộn đệ quy: giá trị ở `override` thắng; dict con được trộn tiếp."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Đọc file YAML. Nếu có khóa `base`, đọc file đó trước (đường dẫn tương đối
    theo file hiện tại) rồi trộn đè lên."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Cấu hình {path} phải là một mapping YAML")
    base_ref = cfg.pop("base", None)
    if base_ref:
        base_cfg = load_config((path.parent / base_ref).resolve())
        cfg = deep_merge(base_cfg, cfg)
    return cfg
