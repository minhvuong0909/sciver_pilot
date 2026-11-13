"""Dung sai theo câu chữ, độ chính xác của số và quy ước làm tròn (SPEC 7-M5).

Không có ngưỡng cứng chung: quy tắc ghi dưới dạng văn bản trong
`tolerance_rule`, và `within_tolerance` chỉ dùng cho cấu hình calculator.
"""

from __future__ import annotations

from decimal import Decimal


def half_unit(precision_decimals: int | None) -> Decimal:
    """Nửa đơn vị của chữ số cuối claim nêu: 12.5 → 0.05; 12 → 0.5."""
    p = precision_decimals or 0
    return Decimal(5) / (Decimal(10) ** (p + 1))


def tolerance_rule(qualifier: str, precision_decimals: int | None, relation: str) -> str | None:
    if relation in ("extremum", "universal", "comparison"):
        return "so sánh thứ tự trên tập so sánh; không dùng dung sai số"
    h = half_unit(precision_decimals)
    if qualifier == "exact":
        return f"làm tròn giá trị bằng chứng tới {precision_decimals or 0} chữ số thập phân như claim (±{h})"
    if qualifier == "approximately":
        return f"'approximately': chấp nhận sai khác do làm tròn tới chữ số có nghĩa của claim (±{h * 10})"
    if qualifier in ("at_least", "at_most", "more_than", "less_than"):
        return f"'{qualifier}': kiểm bất đẳng thức, so sánh sau khi làm tròn tới {precision_decimals or 0} chữ số"
    return None


def within_tolerance(claim_value: Decimal, evidence_value: Decimal, qualifier: str,
                     precision_decimals: int | None) -> bool:
    h = half_unit(precision_decimals)
    if qualifier == "exact":
        return abs(evidence_value - claim_value) <= h
    if qualifier == "approximately":
        return abs(evidence_value - claim_value) <= h * 10
    if qualifier == "at_least":
        return evidence_value >= claim_value - h
    if qualifier == "at_most":
        return evidence_value <= claim_value + h
    if qualifier == "more_than":
        return evidence_value > claim_value
    if qualifier == "less_than":
        return evidence_value < claim_value
    return abs(evidence_value - claim_value) <= h
