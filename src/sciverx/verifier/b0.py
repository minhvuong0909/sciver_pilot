"""B0: chạy code SciVer GỐC (third_party/SciVer, commit 0a97282) trên đúng tập claim của nhóm.

Hai stage CPU bao quanh bước chạy GPU bằng main.py của SciVer:
- `b0-subset`: lấy đúng các claim trong claims_input.jsonl (vd 291 claim bảng) từ
  valset.json, đổi đường dẫn bài/ảnh thành đường dẫn tuyệt đối (main.py chạy trong
  thư mục SciVer) và thêm khóa `claim_id`. Ghi `<work>/b0/subset.json` và
  `<work>/b0/paths.sh` (SUBSET, RAW_DIR, RAW_JSON cho script GPU).
- `b0-import`: đọc file kết quả main.py, chấm bằng pred_old (đúng như acc_evaluation.py)
  và pred_new, ghi `<run_dir>/B0/verifier_outputs.jsonl` để `report` so B0 với IMG, G1.

B0 không truyền seed (đúng code gốc) nên mỗi lần chạy có thể khác: ghi seed = -1.
main.py không lưu finish_reason nên ghi "unknown".
"""

from __future__ import annotations

import json
from pathlib import Path

from sciverx.data.loader import load_split, resolve
from sciverx.m1_prepare import work_dir
from sciverx.schemas import ClaimInput, VerifierOutput
from sciverx.utils.io import read_jsonl, write_jsonl
from sciverx.verifier.prompts import CLAIM_TYPE_TO_ID
from sciverx.verifier.scoring import pred_new, pred_old, verifier_status

PATH_KEYS = ("paper_path", "image_path", "item1_path", "item2_path")
B0_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def b0_dir(cfg: dict) -> Path:
    return work_dir(cfg) / "b0"


def raw_json_name(model: str = B0_MODEL, subset_name: str = "subset", prompt: str = "cot") -> str:
    """main.py ghi vào {output_dir}/{tên file dữ liệu}_{prompt}/{tên model}.json."""
    return f"{subset_name}_{prompt}/{model.split('/')[-1]}.json"


def make_subset(rows: list[dict], claim_ids: list[str], split: str, data_root: str | Path) -> list[dict]:
    """claim_id = "{split}-{vị trí}" (data/select.py) → lấy đúng dòng đó, giữ thứ tự claim_ids."""
    out = []
    for cid in claim_ids:
        sp, pos = cid.rsplit("-", 1)
        if sp != split:
            raise ValueError(f"claim {cid} không thuộc split {split}")
        q = dict(rows[int(pos)])
        for k in PATH_KEYS:
            if q.get(k):
                q[k] = resolve(data_root, q[k]).resolve().as_posix()
        q["claim_id"] = cid
        out.append(q)
    return out


def run_subset(cfg: dict, args) -> int:
    claims = read_jsonl(work_dir(cfg) / "claims_input.jsonl", ClaimInput)
    if cfg.get("limit"):
        claims = claims[: cfg["limit"]]
    root = cfg["paths"]["data_root"]
    subset = make_subset(load_split(root, cfg["split"]), [c.claim_id for c in claims], cfg["split"], root)
    missing = [q["claim_id"] for q in subset
               for k in PATH_KEYS if q.get(k) and not Path(q[k]).exists()]
    d = b0_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    (d / "subset.json").write_text(json.dumps(subset, ensure_ascii=False, indent=1), encoding="utf-8")
    from sciverx.evaluation.report import run_dir_of
    from sciverx.utils.manifest import build_manifest, open_run_dir
    rd = run_dir_of(cfg)
    open_run_dir(rd.parent, build_manifest(rd.name, cfg, [c.claim_id for c in claims]))  # để verify/report dùng chung
    raw_dir = (rd / "B0" / "sciver_raw").resolve()
    model = cfg.get("verifier", {}).get("model", B0_MODEL)
    (d / "paths.sh").write_text(
        f'SUBSET="{(d / "subset.json").resolve().as_posix()}"\n'
        f'RAW_DIR="{raw_dir.as_posix()}"\n'
        f'RAW_JSON="{(raw_dir / raw_json_name(model)).as_posix()}"\n'
        f'B0_MODEL="{model}"\n', encoding="utf-8")
    print(f"[b0-subset] {len(subset)} claim → {d / 'subset.json'}; thiếu file: {len(set(missing))} claim")
    return 1 if missing else 0


def import_outputs(raw: list[dict], run_id: str) -> list[VerifierOutput]:
    outs = []
    for q in raw:
        resp = q.get("response") or ""
        outs.append(VerifierOutput(
            claim_id=q["claim_id"], run_id=run_id, system="B0", evidence_setting="S-oracle",
            table_condition="IMG", prompt_id=CLAIM_TYPE_TO_ID[q["claim_type"]], request_hash="",
            sample_name="b0", temperature=1.0, seed=-1, response=resp, finish_reason="unknown",
            n_images=sum(bool(q.get(k)) for k in PATH_KEYS[1:]),
            pred_old=pred_old(resp), pred_new=pred_new(resp), status=verifier_status(resp, None)))
    return outs


def run_import(cfg: dict, args) -> int:
    from sciverx.evaluation.report import run_dir_of
    rd = run_dir_of(cfg)
    model = cfg.get("verifier", {}).get("model", B0_MODEL)
    raw_path = rd / "B0" / "sciver_raw" / raw_json_name(model)
    if not raw_path.exists():
        print(f"[b0-import] chưa có kết quả main.py: {raw_path}")
        return 2
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    want = {c.claim_id for c in read_jsonl(work_dir(cfg) / "claims_input.jsonl", ClaimInput)}
    if cfg.get("limit"):
        want = set(sorted(want, key=lambda c: int(c.rsplit("-", 1)[1]))[: cfg["limit"]])
    got = {q["claim_id"] for q in raw}
    outs = import_outputs(raw, rd.name)
    write_jsonl(rd / "B0" / "verifier_outputs.jsonl", outs)
    print(f"[b0-import] {len(outs)} claim (thiếu {len(want - got)}) → {rd / 'B0' / 'verifier_outputs.jsonl'}")
    return 0 if want <= got else 1
