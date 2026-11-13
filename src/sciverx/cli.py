"""CLI chung: `python -m sciverx <stage> --config <file.yaml>` (SPEC mục 13).

Mỗi stage trỏ tới một hàm `run(cfg, args)` ở module tương ứng. Module được
import khi stage chạy, để các stage nhẹ không phải cài vLLM, Docling hay torch.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any, Callable

from sciverx.config import load_config

# stage -> "module:hàm"
STAGES: dict[str, str] = {
    "audit": "sciverx.data.audit:run",
    "select": "sciverx.data.select:run",
    "sources": "sciverx.m1_prepare.sources:run",
    "gold": "sciverx.m1_prepare.gold_tables:run",
    "extract": "sciverx.m1_prepare.extract_docling:run",
    "normalize": "sciverx.m1_prepare.table_stages:run_normalize",
    "quality": "sciverx.m1_prepare.table_stages:run_quality",
    "cellmap": "sciverx.m1_prepare.table_stages:run_cellmap",
    "units": "sciverx.m1_prepare.units:run",
    "index": "sciverx.m1_prepare.index:run",
    "retrieve": "sciverx.m3_retrieve.retrieve:run",
    "align": "sciverx.m4_align.align:run",
    "numeric": "sciverx.m5_numeric.calculator:run",
    "package": "sciverx.m6_package.package:run",
    "verify": "sciverx.verifier.run_vllm:run",
    "evaluate": "sciverx.evaluation.report:run_evaluate",
    "report": "sciverx.evaluation.report:run_report",
    "annotate-export": "sciverx.annotation.export_tasks:run",
    "annotate-import": "sciverx.annotation.import_labels:run",
}


class StageNotImplemented(RuntimeError):
    pass


def resolve_stage(name: str) -> Callable[[dict[str, Any], argparse.Namespace], Any]:
    target = STAGES[name]
    module_name, func_name = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and module_name.startswith(exc.name):
            raise StageNotImplemented(f"Stage '{name}' chưa được triển khai ({module_name})") from exc
        raise
    func = getattr(module, func_name, None)
    if func is None:
        raise StageNotImplemented(f"Stage '{name}' chưa có hàm {func_name} trong {module_name}")
    return func


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sciverx", description="Pipeline SciVer-TableVerify")
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("--config", required=True, help="File YAML cấu hình")
    parser.add_argument("--split", choices=["validation", "test"], default=None,
                        help="Ghi đè split trong cấu hình")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test: chỉ chạy N claim đầu (≤ 20, không dùng để báo cáo)")
    parser.add_argument("--structure-only", action="store_true",
                        help="Chỉ cho stage audit: kiểm tên trường và kiểu dữ liệu, không in giá trị")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.split:
        cfg["split"] = args.split
    if cfg.get("split") == "test" and args.stage != "audit":
        frozen = str(args.config).replace("\\", "/")
        if "configs/frozen/" not in frozen:
            print("Từ chối chạy test: chỉ dùng cấu hình trong configs/frozen/ (SPEC mục 5.4)",
                  file=sys.stderr)
            return 2
    if args.limit is not None:
        if args.limit > 20:
            print("--limit chỉ dùng cho smoke test, tối đa 20 claim (SPEC mục 1.2)", file=sys.stderr)
            return 2
        cfg["limit"] = args.limit
    try:
        func = resolve_stage(args.stage)
    except StageNotImplemented as exc:
        print(str(exc), file=sys.stderr)
        return 3
    result = func(cfg, args)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
