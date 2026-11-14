"""Logger dùng chung; ghi ra console và (tùy chọn) file trong thư mục lượt chạy."""

from __future__ import annotations

import logging
from pathlib import Path

_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "sciverx", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    if log_file is not None:
        path = str(Path(log_file).resolve())
        if not any(getattr(h, "baseFilename", None) == path for h in logger.handlers):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FMT))
            logger.addHandler(fh)
    return logger
