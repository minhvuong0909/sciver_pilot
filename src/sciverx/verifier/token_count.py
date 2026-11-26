"""Stage `count-tokens`: đếm độ dài đầu vào của verifier mà không cần GPU.

Mục đích: chốt `verifier.max_model_len` trước khi chạy thật (hyper_parameter.md
mục 5.5 #1). Một request chạy được khi
    prompt_tokens + max_tokens <= max_model_len.

- Token chữ: đếm bằng tokenizer của model verifier (cần `transformers`).
- Token ảnh: ƯỚC LƯỢNG theo cách Qwen2.5-VL đổi kích thước ảnh (`smart_resize`,
  mỗi visual token = ô 28×28 pixel, giới hạn pixel mặc định của processor), cộng
  2 token đánh dấu đầu/cuối ảnh. Chưa gồm chat template (vài chục token), nên
  cộng thêm `TEMPLATE_MARGIN`. Con số thật xem `prompt_tokens` sau khi chạy.

Chỉ hỗ trợ S-oracle và S-retr (S-align cần gọi M4 trên GPU).
Ghi `<work>/token_counts/<experiment>_<cond>.jsonl` và in tóm tắt.
"""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Callable

from sciverx.m1_prepare import work_dir
from sciverx.schemas import ClaimInput
from sciverx.utils.io import read_jsonl, write_jsonl

PATCH = 28                     # 14×14 patch, gộp 2×2 → mỗi visual token ứng với 28×28 pixel
MIN_PIXELS = 4 * 28 * 28       # mặc định của processor Qwen2.5-VL [CẦN KIỂM với revision đang dùng]
MAX_PIXELS = 16384 * 28 * 28
TEMPLATE_MARGIN = 64


def smart_resize(height: int, width: int, factor: int = PATCH, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """Kích thước ảnh sau khi processor Qwen2/2.5-VL đổi cỡ (chia hết cho factor)."""
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def visual_tokens(width: int, height: int) -> int:
    h, w = smart_resize(height, width)
    return (h // PATCH) * (w // PATCH) + 2


def _pct(values: list[int], q: float) -> int:
    s = sorted(values)
    return s[min(len(s) - 1, math.ceil(q * len(s)) - 1)] if s else 0


def summarize(rows: list[dict], max_model_len: int, max_tokens: int) -> dict:
    totals = [r["total_tokens_est"] for r in rows]
    limit = max_model_len - max_tokens
    over = [r["claim_id"] for r in rows if r["total_tokens_est"] > limit]
    return {"n": len(rows), "input_limit": limit, "mean": round(mean(totals), 1) if totals else 0,
            "median": median(totals) if totals else 0, "p95": _pct(totals, 0.95),
            "max": max(totals) if totals else 0, "n_over_limit": len(over), "over_limit_claims": over}


def count_inputs(inputs, count: Callable[[str], int], load_image, image_size) -> list[dict]:
    rows = []
    for b in inputs:
        text = count(b.request.text)
        vis = 0
        sizes = []
        for p in b.images:
            sz = image_size(load_image(p))
            if sz is not None:
                sizes.append(sz)
                vis += visual_tokens(*sz)
        rows.append({"claim_id": b.request.claim_id, "prompt_id": b.request.prompt_id,
                     "text_tokens": text, "visual_tokens_est": vis, "image_sizes": sizes,
                     "total_tokens_est": text + vis + TEMPLATE_MARGIN})
    return rows


def run(cfg: dict, args, count: Callable[[str], int] | None = None, embedder=None, load_image=None) -> int:
    from sciverx.verifier.run_vllm import image_size, preprocess_image
    from sciverx.verifier.stage import build_all_inputs

    vcfg = cfg["verifier"]
    if vcfg["setting"] not in ("S-oracle", "S-retr"):
        print("count-tokens chỉ hỗ trợ S-oracle và S-retr (S-align cần M4 trên GPU)")
        return 2
    claims = read_jsonl(work_dir(cfg) / "claims_input.jsonl", ClaimInput)
    if cfg.get("limit"):
        claims = claims[: cfg["limit"]]
    if count is None:  # pragma: no cover - cần transformers
        from sciverx.m3_retrieve.stage import qwen_token_counter
        count = qwen_token_counter(vcfg["model"])
    if embedder is None and vcfg["setting"] == "S-retr" and vcfg.get("method", "hybrid") != "bm25":  # pragma: no cover
        from sciverx.m3_retrieve.rankers import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder(cfg["m1"]["units"]["embed_model"])
    load_image = load_image or preprocess_image
    max_model_len = vcfg.get("max_model_len", 32768)
    max_tokens = vcfg.get("max_tokens", 10240)
    exp = cfg.get("experiment", "verify")
    out_dir = work_dir(cfg) / "token_counts"
    for cond in vcfg["table_conditions"]:
        inputs, _ = build_all_inputs(cfg, claims, cond, None, count, embedder)
        rows = count_inputs(inputs, count, load_image, image_size)
        write_jsonl(out_dir / f"{exp}_{cond}.jsonl", rows)
        s = summarize(rows, max_model_len, max_tokens)
        print(f"[count-tokens] {vcfg['setting']}/{cond}: n={s['n']} trung bình={s['mean']} "
              f"trung vị={s['median']} p95={s['p95']} max={s['max']} | giới hạn đầu vào={s['input_limit']} "
              f"(max_model_len {max_model_len} − max_tokens {max_tokens}) → vượt: {s['n_over_limit']}")
    print(f"[count-tokens] → {out_dir}")
    return 0
