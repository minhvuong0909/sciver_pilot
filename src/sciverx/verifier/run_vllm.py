"""Chạy verifier Qwen2.5-VL-7B qua vLLM — SPEC 8.1.

Bốn lượt sinh: t1_s1, t1_s2, t1_s3 (temperature 1.0, seed 1/2/3, seed được
truyền THẬT vào SamplingParams — khác code gốc) và t0 (temperature 0).
Ảnh tiền xử lý như code gốc: RGB, phóng ảnh có cạnh < 224 (giữ tỉ lệ), JPEG.
[CẦN KIỂM] max_model_len và bộ nhớ trên GPU thật (ghi vào VERIFICATION_LOG).
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Sequence

from sciverx.llm import LLMRequest, LLMResponse
from sciverx.schemas import VerifierOutput
from sciverx.verifier.build_inputs import BuiltInput
from sciverx.verifier.scoring import pred_new, pred_old


def preprocess_image(path: str | Path, min_size: int = 224) -> bytes:
    """Giống SciVer với Qwen2.5-VL: RGB; cạnh nào < min_size thì phóng giữ tỉ lệ; JPEG."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if img.width < min_size or img.height < min_size:
        r = max(min_size / img.width, min_size / img.height)
        img = img.resize((int(img.width * r), int(img.height * r)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class VllmClient:  # pragma: no cover - cần GPU và vllm
    def __init__(self, model: str, max_model_len: int = 32768, limit_mm_per_prompt: dict | None = None,
                 tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
        from vllm import LLM
        self.model_name = model
        self.llm = LLM(model=model, max_model_len=max_model_len,
                       limit_mm_per_prompt=limit_mm_per_prompt or {"image": 2},
                       tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=gpu_memory_utilization)

    def generate(self, requests: Sequence[LLMRequest]) -> list[LLMResponse]:
        from vllm import SamplingParams
        out = []
        for req in requests:  # mỗi request có thể khác temperature/seed
            content = [{"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(img).decode()}} for img in req.images]
            content.append({"type": "text", "text": req.prompt})
            sp = SamplingParams(temperature=req.temperature, max_tokens=req.max_tokens, seed=req.seed)
            o = self.llm.chat([{"role": "user", "content": content}], sp, use_tqdm=False)[0]
            out.append(LLMResponse(text=o.outputs[0].text, finish_reason=o.outputs[0].finish_reason,
                                   prompt_tokens=len(o.prompt_token_ids or []),
                                   output_tokens=len(o.outputs[0].token_ids)))
        return out


def run_samples(llm, inputs: list[BuiltInput], samples: list[dict], max_tokens: int, *, run_id: str,
                system: str, setting: str, table_condition: str, load_image=preprocess_image
                ) -> list[VerifierOutput]:
    """Gửi mọi input cho từng lượt sinh; một mẫu lỗi không làm dừng cả lượt."""
    outs: list[VerifierOutput] = []
    for s in samples:
        reqs = [LLMRequest(prompt=b.request.text, images=[load_image(p) for p in b.images],
                           temperature=s["temperature"], seed=s["seed"], max_tokens=max_tokens)
                for b in inputs]
        try:
            resps = llm.generate(reqs)
            errors: list[str | None] = [None] * len(resps)
        except Exception:
            resps, errors = [], []
            for r in reqs:
                try:
                    resps.append(llm.generate([r])[0])
                    errors.append(None)
                except Exception as exc:  # noqa: BLE001
                    resps.append(LLMResponse(text="", finish_reason="error"))
                    errors.append(f"{type(exc).__name__}: {exc}")
        for b, r, err in zip(inputs, resps, errors):
            outs.append(VerifierOutput(
                claim_id=b.request.claim_id, run_id=run_id, system=system, evidence_setting=setting,
                table_condition=table_condition, prompt_id=b.request.prompt_id,
                request_hash=b.request.request_hash, sample_name=s["name"],
                temperature=s["temperature"], seed=s["seed"], response=r.text,
                finish_reason=r.finish_reason, prompt_tokens=r.prompt_tokens,
                output_tokens=r.output_tokens, n_images=len(b.images),
                pred_old=pred_old(r.text) if not err else None,
                pred_new=pred_new(r.text) if not err else None, error=err))
    return outs
