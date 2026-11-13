"""Bước 5: chạy bộ kiểm chứng Qwen2.5-VL-7B qua vLLM cho các điều kiện và các lần chạy.

Điều kiện:
  IMG  : như paper - bảng là ảnh (tiền xử lý ảnh giống code gốc)
  G0/G1/E0/E1 : bảng thành văn bản (linearize dạng "|"), biểu đồ (nếu có) vẫn là ảnh
Lần chạy (mặc định): t1_s1, t1_s2, t1_s3 (temperature 1.0 như paper, seed 1/2/3) và t0 (temperature 0).

Chạy trong môi trường "sciver" (vLLM). Ví dụ:
  python src/s06_run_verifier.py --data_root data/SciVer --samples work/samples.json --out_dir work \
      --conditions IMG G0 G1 E0 E1 --model Qwen/Qwen2.5-VL-7B-Instruct
Thêm --dry_run để chỉ in prompt mẫu, không cần GPU.
"""
import argparse
import base64
import io
import os
import time

from common import (PathResolver, caption_of, evidence_items, load_json, load_paper,
                    prepare_context, save_json, save_jsonl)
from prompts import COT_PROMPT, apply_text_mode
from tables import linearize

DEFAULT_RUNS = [("t1_s1", 1.0, 1), ("t1_s2", 1.0, 2), ("t1_s3", 1.0, 3), ("t0", 0.0, 0)]


def image_data_url(path, model_name):
    """Tiền xử lý giống SciVer: RGB, phóng cạnh < 224 lên 224 (giữ tỉ lệ), nén lại JPEG."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    min_size = 224
    if model_name in ("Qwen/Qwen2-VL-7B-Instruct", "microsoft/Phi-4-multimodal-instruct"):
        s = min_size / min(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.BICUBIC)
    elif img.width < min_size or img.height < min_size:
        r = max(min_size / img.width, min_size / img.height)
        img = img.resize((int(img.width * r), int(img.height * r)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_message(q, cond, res, out_dir, model_name):
    paper = load_paper(res, q)
    ct = q["claim_type"]
    ctx = prepare_context(paper, q["section"])
    items = evidence_items(q)
    caps = [caption_of(paper, t, it) for t, it, _ in items]
    tpl = COT_PROMPT[ct]
    if len(items) == 1:
        text = tpl.substitute(claim=q["claim"], context=ctx, caption=caps[0])
    else:
        text = tpl.substitute(claim=q["claim"], context=ctx, caption1=caps[0], caption2=caps[1])

    content, table_blocks, meta = [], [], []
    for k, ((t, it, img), cap) in enumerate(zip(items, caps), start=1):
        if t == "table" and cond != "IMG":
            key = f"{q['_qid']}__{it}"
            tab = load_json(os.path.join(out_dir, "tables", cond, key + ".json"))["table"]
            name = "Table" if len(items) == 1 else f"item{k} Table"
            table_blocks.append(linearize(tab, cap, name=name))
            meta.append({"item": it, "as": "text", "rows": len(tab["grid"])})
        else:
            content.append({"type": "image_url", "image_url": {"url": image_data_url(res.resolve(img), model_name)}})
            meta.append({"item": it, "as": "image"})
    if table_blocks:
        if len(items) == 1:
            text = apply_text_mode(text)
        text = "Table content (as text):\n" + "\n\n".join(table_blocks) + "\n" + text
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}], meta, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--samples", default="work/samples.json")
    ap.add_argument("--out_dir", default="work")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--conditions", nargs="+", default=["IMG", "G0", "G1", "E0", "E1"])
    ap.add_argument("--runs", nargs="+", default=None, help="vd t1_s1 t0 (mặc định: cả 4)")
    ap.add_argument("--max_tokens", type=int, default=10240, help="paper dùng 10240")
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel (số GPU)")
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    res = PathResolver(args.data_root)
    samples = load_json(args.samples)
    runs = [r for r in DEFAULT_RUNS if args.runs is None or r[0] in args.runs]

    built = {}
    for cond in args.conditions:
        built[cond] = [build_message(q, cond, res, args.out_dir, args.model) for q in samples]
    if args.dry_run:
        for cond in args.conditions:
            msg, meta, text = built[cond][0]
            n_img = sum(c["type"] == "image_url" for c in msg[0]["content"])
            print(f"===== {cond} | qid={samples[0]['_qid']} | {n_img} ảnh | meta={meta}\n{text[:1500]}\n")
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              limit_mm_per_prompt={"image": 2}, gpu_memory_utilization=args.gpu_mem)
    save_json({"model": args.model, "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
               "runs": runs, "conditions": args.conditions},
              os.path.join(args.out_dir, "runs", "run_config.json"))

    for cond in args.conditions:
        msgs = [m for m, _, _ in built[cond]]
        for run_name, temp, seed in runs:
            out_path = os.path.join(args.out_dir, "runs", cond, f"{run_name}.jsonl")
            if os.path.exists(out_path):
                print(f"Bỏ qua {out_path} (đã có)")
                continue
            sp = SamplingParams(temperature=temp, max_tokens=args.max_tokens, seed=seed)
            t0 = time.time()
            try:
                outs = llm.chat(msgs, sp, use_tqdm=True)
                errors = [None] * len(msgs)
            except Exception as e:  # noqa: BLE001  - chạy từng mẫu để biết mẫu nào lỗi
                print(f"[{cond}/{run_name}] lỗi theo lô ({e}); chạy từng mẫu...")
                outs, errors = [], []
                for m in msgs:
                    try:
                        outs.append(llm.chat([m], sp, use_tqdm=False)[0]); errors.append(None)
                    except Exception as e2:  # noqa: BLE001
                        outs.append(None); errors.append(f"{type(e2).__name__}: {e2}")
            dt = time.time() - t0
            rows = []
            for q, (_, meta, _), o, err in zip(samples, built[cond], outs, errors):
                rows.append({
                    "qid": q["_qid"], "claim_type": q["claim_type"], "label": q["_label"],
                    "mixed": q.get("_mixed", False), "condition": cond, "run": run_name,
                    "temperature": temp, "seed": seed, "evidence": meta,
                    "response": o.outputs[0].text if o else "",
                    "finish_reason": o.outputs[0].finish_reason if o else "error",
                    "prompt_tokens": len(o.prompt_token_ids) if o and o.prompt_token_ids else None,
                    "output_tokens": len(o.outputs[0].token_ids) if o else None,
                    "error": err,
                })
            save_jsonl(rows, out_path)
            print(f"[{cond}/{run_name}] {len(rows)} mẫu, {dt:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
