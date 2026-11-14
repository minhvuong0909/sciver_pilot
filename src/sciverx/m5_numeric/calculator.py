"""Bộ tính tùy chọn (m5.calculator = true, thí nghiệm X4c riêng) — SPEC 7-M5.

Chỉ tính từ evidence_operands; ghi `computed = {value, formula, config_id}`.
Không đưa kết quả này vào verifier ở X4b (chỉ mô tả phép tính).
"""

from __future__ import annotations

from decimal import Decimal, getcontext

from sciverx.schemas import NumericCheck

getcontext().prec = 28


def _vals(chk: NumericCheck) -> list[Decimal]:
    return [Decimal(o.value) for o in chk.evidence_operands if o.value is not None]


def compute(chk: NumericCheck, config_id: str = "calc_v1") -> dict | None:
    v = _vals(chk)
    if chk.relation == "percentage_point_diff" and len(v) >= 2:
        return {"value": str(v[1] - v[0]), "formula": f"{v[1]} - {v[0]}", "config_id": config_id}
    if chk.relation == "relative_change" and len(v) >= 2 and v[0] != 0:
        val = (v[1] - v[0]) / v[0] * 100
        return {"value": str(val.normalize()), "formula": f"({v[1]} - {v[0]}) / {v[0]} * 100",
                "config_id": config_id}
    if chk.relation == "ratio_multiple" and len(v) >= 2 and min(v[0], v[1]) != 0:
        if chk.quantity_kind == "time":  # nhanh hơn N lần = thời gian cũ / thời gian mới
            hi, lo = max(v[0], v[1]), min(v[0], v[1])
            return {"value": str((hi / lo).normalize()), "formula": f"{hi} / {lo} (time: inverse)",
                    "config_id": config_id}
        if chk.quantity_kind in ("throughput", "speed"):
            hi, lo = max(v[0], v[1]), min(v[0], v[1])
            return {"value": str((hi / lo).normalize()), "formula": f"{hi} / {lo}", "config_id": config_id}
        return None
    if chk.relation == "average" and v:
        return {"value": str((sum(v) / len(v)).normalize()), "formula": f"mean of {len(v)} values",
                "config_id": config_id}
    if chk.relation == "sum" and v:
        return {"value": str(sum(v)), "formula": f"sum of {len(v)} values", "config_id": config_id}
    return None


def with_computed(checks: list[NumericCheck], config_id: str = "calc_v1") -> list[NumericCheck]:
    return [c.model_copy(update={"computed": compute(c, config_id)}) for c in checks]


def describe(checks: list[NumericCheck]) -> str:
    """Mô tả cho verifier (unified_v1_numeric): toán hạng có nguồn và phép tính,
    KHÔNG có kết quả hay kết luận."""
    lines = []
    for c in checks:
        claim = ", ".join(o.value_text for o in c.claim_operands) or "-"
        ev = "; ".join(
            f"{o.value_text} [{o.ref.row_header} / {o.ref.col_header}]" if o.ref is not None and hasattr(o.ref, "row_header")
            else o.value_text for o in c.evidence_operands) or "-"
        extra = []
        if c.qualifier not in ("exact", "none"):
            extra.append(f"qualifier: {c.qualifier}")
        if c.direction and c.direction != "unknown":
            extra.append(f"direction: {c.direction}")
        if c.quantity_kind and c.quantity_kind != "unknown":
            extra.append(f"quantity: {c.quantity_kind}")
        if c.comparison_set:
            extra.append(f"comparison set: {len(c.comparison_set)} cells ({c.comparison_set_status})")
        if c.time:
            extra.append(f"time: {c.time.text}")
        tail = f" ({'; '.join(extra)})" if extra else ""
        lines.append(f"- {c.relation}: claim value(s) {claim}; evidence value(s) {ev}{tail}")
    return "\n".join(lines)
