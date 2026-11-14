"""Hash request gửi tới model (SPEC 8.2): SHA-256 của (prompt_id, văn bản, sha256 từng ảnh).

Dùng để kiểm chống rò rỉ: xáo trộn GoldInfo không được làm đổi request_hash ở
các thiết lập truy xuất (SPEC 12.4).
"""

from __future__ import annotations

from pathlib import Path

from sciverx.schemas import VerifierRequest
from sciverx.utils.hashing import sha256_file, sha256_obj


def request_hash(prompt_id: str, text: str, image_sha256: list[str]) -> str:
    return sha256_obj({"prompt_id": prompt_id, "text": text, "images": list(image_sha256)})


def make_request(claim_id: str, prompt_id: str, text: str, images: list[str | Path]) -> VerifierRequest:
    shas = [sha256_file(p) for p in images]
    return VerifierRequest(claim_id=claim_id, prompt_id=prompt_id, text=text, image_sha256=shas,
                           request_hash=request_hash(prompt_id, text, shas))
