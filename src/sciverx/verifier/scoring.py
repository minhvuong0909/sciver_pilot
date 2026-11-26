"""Chấm câu trả lời của verifier — SPEC 8.4.

- pred_old: 'yes' xuất hiện ở bất kỳ đâu (như acc_evaluation.py của SciVer).
  Đầu ra rỗng → None (tính là sai), vì code gốc bỏ qua mẫu có `response` rỗng
  (`if item["response"]`) nên mẫu đó sai ở cả hai nhãn.
- pred_new: lần xuất hiện CUỐI của "Answer: yes|no" (cho phép **, ngoặc, khoảng
  trắng); dự phòng "final answer is yes|no"; không đọc được → None (tính là sai).
- verifier_status: phân loại mỗi đầu ra để báo riêng tỷ lệ lỗi kỹ thuật, đầu ra
  rỗng và parse_error (hyper_parameter.md mục 5.1, 9.1).
"""

from __future__ import annotations

import re
from typing import Literal

_ANSWER = re.compile(r"answer\s*[:：]\s*[\*\s]*[\(\[\"'`]*\s*(yes|no)\b", re.I)
_FALLBACK = re.compile(r"final answer is\s*[:：]?\s*[\*\s]*[\(\[\"'`]*\s*(yes|no)\b", re.I)

VerifierStatus = Literal["ok", "parse_error", "empty", "error"]


def pred_old(response: str) -> bool | None:
    if not response:
        return None
    return "yes" in response.lower()


def pred_new(response: str) -> bool | None:
    hits = _ANSWER.findall(response)
    if not hits:
        hits = _FALLBACK.findall(response)
    if not hits:
        return None
    return hits[-1].lower() == "yes"


def verifier_status(response: str, error: str | None) -> VerifierStatus:
    """error: lỗi kỹ thuật (OOM, timeout, vượt context…); empty: đầu ra chỉ có
    khoảng trắng; parse_error: có chữ nhưng không đọc được kết luận; ok: còn lại.
    Đầu ra bị cắt (finish_reason = length) vẫn có thể là ok hoặc parse_error;
    tỷ lệ bị cắt được báo riêng theo finish_reason."""
    if error:
        return "error"
    if not response.strip():
        return "empty"
    if pred_new(response) is None:
        return "parse_error"
    return "ok"


def is_correct(pred: bool | None, label: bool) -> bool:
    return pred is not None and pred == label
