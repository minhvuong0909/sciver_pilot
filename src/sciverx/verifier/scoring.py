"""Chấm câu trả lời của verifier — SPEC 8.4.

- pred_old: 'yes' xuất hiện ở bất kỳ đâu (như acc_evaluation.py của SciVer).
- pred_new: lần xuất hiện CUỐI của "Answer: yes|no" (cho phép **, ngoặc, khoảng
  trắng); dự phòng "final answer is yes|no"; không đọc được → None (tính là sai).
"""

from __future__ import annotations

import re

_ANSWER = re.compile(r"answer\s*[:：]\s*[\*\s]*[\(\[\"'`]*\s*(yes|no)\b", re.I)
_FALLBACK = re.compile(r"final answer is\s*[:：]?\s*[\*\s]*[\(\[\"'`]*\s*(yes|no)\b", re.I)


def pred_old(response: str) -> bool:
    return "yes" in response.lower()


def pred_new(response: str) -> bool | None:
    hits = _ANSWER.findall(response)
    if not hits:
        hits = _FALLBACK.findall(response)
    if not hits:
        return None
    return hits[-1].lower() == "yes"


def is_correct(pred: bool | None, label: bool) -> bool:
    return pred is not None and pred == label
