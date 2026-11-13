"""run_id, thư mục lượt chạy và manifest.json — SPEC mục 13.

run_id = {date}_{experiment}_{split}_{confighash[:8]}. Không bao giờ ghi đè:
cùng run_id thì chỉ được chạy tiếp (resume) khi hash cấu hình trùng.
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import platform
import subprocess
from pathlib import Path

from sciverx.schemas import RunManifest
from sciverx.utils.hashing import ids_hash, sha256_obj
from sciverx.utils.io import read_json, write_json

TRACKED_PACKAGES = ["sciverx", "pydantic", "numpy", "rank_bm25", "sentence-transformers",
                    "transformers", "torch", "vllm", "docling"]


class RunExistsError(RuntimeError):
    pass


def config_hash(cfg: dict) -> str:
    """Hash toàn bộ cấu hình, kể cả `limit`, để smoke test có run_id riêng."""
    return sha256_obj(cfg)


def make_run_id(experiment: str, split: str, cfg: dict, date: dt.date | None = None) -> str:
    date = date or dt.date.today()
    return f"{date:%Y%m%d}_{experiment}_{split}_{config_hash(cfg)[:8]}"


def git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def package_versions(names: list[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_info() -> dict | None:
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    if not torch.cuda.is_available():  # pragma: no cover
        return {"cuda": False}
    props = torch.cuda.get_device_properties(0)  # pragma: no cover
    return {"cuda": True, "name": props.name, "total_memory_gb": round(props.total_memory / 2**30, 1),
            "count": torch.cuda.device_count(), "cuda_version": torch.version.cuda}


def build_manifest(run_id: str, cfg: dict, claim_ids: list[str], **extra) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        git_commit=git_commit(),
        config=cfg,
        config_hash=config_hash(cfg),
        claim_ids=list(claim_ids),
        claim_ids_hash=ids_hash(list(claim_ids)),
        versions=package_versions(),
        gpu=gpu_info(),
        **extra,
    )


def open_run_dir(outputs_root: str | Path, manifest: RunManifest, resume: bool = True) -> Path:
    """Tạo outputs/runs/<run_id>/ và ghi manifest. Nếu thư mục đã có:
    - cùng config_hash và resume=True → dùng lại (chạy tiếp);
    - khác hash hoặc resume=False → lỗi, không ghi đè."""
    run_dir = Path(outputs_root) / manifest.run_id
    mpath = run_dir / "manifest.json"
    if mpath.exists():
        old = read_json(mpath)
        if not resume:
            raise RunExistsError(f"{run_dir} đã tồn tại; không ghi đè lượt chạy cũ")
        if old.get("config_hash") != manifest.config_hash:
            raise RunExistsError(f"{run_dir} đã tồn tại với cấu hình khác hash; không ghi đè")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(mpath, manifest)
    return run_dir
