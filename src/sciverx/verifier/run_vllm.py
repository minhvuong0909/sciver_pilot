"""Chạy verifier Qwen2.5-VL-7B qua vLLM — SPEC 8.1.

Bốn lượt sinh: t1_s1, t1_s2, t1_s3 (temperature 1.0, seed 1/2/3, seed được
truyền THẬT vào SamplingParams — khác code gốc) và t0 (temperature 0).
Ảnh tiền xử lý như code gốc: RGB, phóng ảnh có cạnh < 224 (giữ tỉ lệ), JPEG.
[CẦN KIỂM] max_model_len và bộ nhớ trên GPU thật (ghi vào VERIFICATION_LOG).

Retry (hyper_parameter.md mục 9.1): mỗi lượt gửi cả lô một lần; nếu lô lỗi thì
gửi lại TỪNG request, mỗi request được thử thêm tối đa `max_retries` lần, cùng
cấu hình. Số lần gọi của mỗi request ghi vào `VerifierOutput.attempts`.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Sequence

from sciverx.llm import LLMRequest, LLMResponse
from sciverx.schemas import VerifierOutput
from sciverx.verifier.build_inputs import BuiltInput
from sciverx.verifier.scoring import pred_new, pred_old, verifier_status


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


def image_size(data: bytes) -> list[int] | None:
    """[rộng, cao] của ảnh đã tiền xử lý; None nếu không đọc được (vd ảnh giả trong test)."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as img:
            return [img.width, img.height]
    except Exception:  # noqa: BLE001 - thiếu PIL hoặc bytes không phải ảnh
        return None


class VllmClient:  # pragma: no cover - cần GPU và vllm
    def __init__(self, model: str, max_model_len: int = 32768, limit_mm_per_prompt: dict | None = None,
                 tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9,
                 revision: str | None = None):
        from vllm import LLM
        self.model_name = model
        self.llm = LLM(model=model, max_model_len=max_model_len,
                       limit_mm_per_prompt=limit_mm_per_prompt or {"image": 2},
                       tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=gpu_memory_utilization,
                       revision=revision)
        self.runtime = {"max_model_len": max_model_len, "tensor_parallel_size": tensor_parallel_size,
                        "gpu_memory_utilization": gpu_memory_utilization,
                        "limit_mm_per_prompt": limit_mm_per_prompt or {"image": 2},
                        "model_revision": revision, **_engine_info(self.llm)}

    def generate(self, requests: Sequence[LLMRequest]) -> list[LLMResponse]:
        """Gửi CẢ LÔ trong một lần llm.chat để vLLM xử lý song song (nhanh hơn nhiều so với
        từng câu). Mỗi request có SamplingParams riêng (seed riêng), nên kết quả của một
        request không phụ thuộc request khác trong lô. vLLM trả đúng thứ tự đầu vào.
        [CẦN KIỂM trên GPU] llm.chat nhận danh sách SamplingParams ở vLLM 0.9.0.1; so vài
        claim với cách gửi từng câu trước khi chạy thật."""
        from vllm import SamplingParams
        if not requests:
            return []
        convs, sps = [], []
        for req in requests:
            content = [{"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(img).decode()}} for img in req.images]
            content.append({"type": "text", "text": req.prompt})
            convs.append([{"role": "user", "content": content}])
            sps.append(SamplingParams(temperature=req.temperature, max_tokens=req.max_tokens, seed=req.seed))
        outs = self.llm.chat(convs, sps, use_tqdm=True)
        return [LLMResponse(text=o.outputs[0].text, finish_reason=o.outputs[0].finish_reason,
                            prompt_tokens=len(o.prompt_token_ids or []),
                            output_tokens=len(o.outputs[0].token_ids)) for o in outs]


def _engine_info(llm) -> dict:  # pragma: no cover - cần vllm
    """Engine V0/V1 và dtype thực tế. Đọc được gì ghi nấy; không đọc được thì None."""
    info: dict = {"env_VLLM_USE_V1": os.environ.get("VLLM_USE_V1")}
    try:
        import vllm.envs as envs
        info["vllm_use_v1"] = bool(envs.VLLM_USE_V1)
    except Exception:  # noqa: BLE001
        info["vllm_use_v1"] = None
    try:
        info["dtype"] = str(llm.llm_engine.model_config.dtype)
    except Exception:  # noqa: BLE001
        info["dtype"] = None
    try:
        info["engine_class"] = type(llm.llm_engine).__module__ + "." + type(llm.llm_engine).__name__
    except Exception:  # noqa: BLE001
        info["engine_class"] = None
    return info


def _generate_with_retry(llm, reqs: list[LLMRequest], max_retries: int
                         ) -> tuple[list[LLMResponse], list[str | None], list[int]]:
    """Trả (responses, errors, attempts) cùng thứ tự với reqs."""
    try:
        resps = llm.generate(reqs)
        return list(resps), [None] * len(resps), [1] * len(resps)
    except Exception:  # noqa: BLE001 - lô lỗi → gửi lại từng request
        pass
    resps, errors, attempts = [], [], []
    for r in reqs:
        last_err: str | None = None
        resp = LLMResponse(text="", finish_reason="error")
        n = 1                                   # lần gửi cả lô đã tính là lần 1
        for _ in range(max_retries):
            n += 1
            try:
                resp = llm.generate([r])[0]
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
        else:
            if max_retries == 0:
                last_err = "batch_failed_no_retry"
        resps.append(resp if last_err is None else LLMResponse(text="", finish_reason="error"))
        errors.append(last_err)
        attempts.append(n)
    return resps, errors, attempts


def run_samples(llm, inputs: list[BuiltInput], samples: list[dict], max_tokens: int, *, run_id: str,
                system: str, setting: str, table_condition: str, load_image=preprocess_image,
                max_retries: int = 1, done: dict[tuple[str, str], str] | None = None,
                chunk_size: int = 0, on_chunk=None) -> list[VerifierOutput]:
    """Gửi input cho từng lượt sinh theo lô; một mẫu lỗi không làm dừng cả lượt.

    Chạy tiếp được (resume): `done` = {(claim_id, sample_name): request_hash} của các đầu ra
    đã có và không lỗi; request nào khớp cả ba thì bỏ qua. `chunk_size` > 0 chia mỗi lượt
    thành lô nhỏ; sau mỗi lô gọi `on_chunk(outputs_của_lô)` để ghi ngay ra đĩa, nên GPU
    ngắt giữa chừng chỉ mất tối đa một lô. Trả về các đầu ra MỚI sinh trong lần gọi này."""
    done = done or {}
    outs: list[VerifierOutput] = []
    img_cache: dict[int, tuple[list[bytes], list[list[int]]]] = {}

    def imgs_of(i: int):
        if i not in img_cache:
            imgs = [load_image(p) for p in inputs[i].images]
            img_cache[i] = (imgs, [s for s in (image_size(x) for x in imgs) if s is not None])
        return img_cache[i]

    for s in samples:
        todo = [i for i, b in enumerate(inputs)
                if done.get((b.request.claim_id, s["name"])) != b.request.request_hash]
        step = chunk_size if chunk_size > 0 else max(1, len(todo))
        for c0 in range(0, len(todo), step):
            idx = todo[c0:c0 + step]
            chunk_outs = _run_chunk(llm, [inputs[i] for i in idx], [imgs_of(i) for i in idx], s, max_tokens,
                                    max_retries, run_id=run_id, system=system, setting=setting,
                                    table_condition=table_condition)
            outs.extend(chunk_outs)
            if on_chunk is not None:
                on_chunk(chunk_outs)
    return outs


def _run_chunk(llm, inputs: list[BuiltInput], imgs: list[tuple[list[bytes], list[list[int]]]], s: dict,
               max_tokens: int, max_retries: int, *, run_id: str, system: str, setting: str,
               table_condition: str) -> list[VerifierOutput]:
    outs: list[VerifierOutput] = []
    reqs = [LLMRequest(prompt=b.request.text, images=im[0], temperature=s["temperature"],
                       seed=s["seed"], max_tokens=max_tokens) for b, im in zip(inputs, imgs)]
    resps, errors, attempts = _generate_with_retry(llm, reqs, max_retries)
    for b, r, err, n, (_, sz) in zip(inputs, resps, errors, attempts, imgs):
        outs.append(VerifierOutput(
            claim_id=b.request.claim_id, run_id=run_id, system=system, evidence_setting=setting,
            table_condition=table_condition, prompt_id=b.request.prompt_id,
            request_hash=b.request.request_hash, sample_name=s["name"],
            temperature=s["temperature"], seed=s["seed"], response=r.text,
            finish_reason=r.finish_reason, prompt_tokens=r.prompt_tokens,
            output_tokens=r.output_tokens, n_images=len(b.images),
            pred_old=pred_old(r.text) if not err else None,
            pred_new=pred_new(r.text) if not err else None, error=err,
            status=verifier_status(r.text, err), attempts=n, image_sizes=sz))
    return outs
