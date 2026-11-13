"""Đặt seed toàn cục cho random, numpy và (nếu có) torch."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch  # type: ignore
    except ImportError:
        return
    torch.manual_seed(seed)  # pragma: no cover
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)
