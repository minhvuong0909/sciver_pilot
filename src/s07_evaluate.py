"""Bước 6: chấm điểm, đo độ ổn định, liệt kê đổi kết luận, viết báo cáo tóm tắt.

Hai cách chấm:
  old : giống acc_evaluation.py của SciVer - có chữ "yes" ở BẤT KỲ đâu => Entailed
  new : đọc kết luận cuối "Answer: yes/no" (lần xuất hiện cuối cùng); không đọc được => unparsed (tính sai)

Đầu ra trong work/eval/:
  per_run.csv       accuracy từng (điều kiện, lần chạy, cách chấm), theo tập con
  stability.csv     độ ổn định giữa các seed temperature 1.0
  flips.csv         các claim đổi đúng/sai giữa 2 điều kiện (theo lần t0 và theo đa số)
  per_claim.csv     bảng claim x điều kiện để xem tay
  summary.md        tóm tắt cho báo cáo pilot

  python src/s07_evaluate.py --out_dir work
"""
import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict
from statistics import mean, pstdev

from common import load_jsonl

ANS_RE = re.compile(r"answer\s*[:：]\s*[\*\"'`\[\(\s]*\s*(yes|no)\b", re.I)
FINAL_RE = re.compile(r"final answer\s*(?:is)?\s*[:：]?\s*[\*\"'`\[\(\s]*(yes|no)\b", re.I)


def score_old(resp):
    if not resp:
        return None
    return "yes" in resp.lower()


def parse_new(resp):
    if not resp:
        return None
    m = ANS_RE.findall(resp) or FINAL_RE.findall(resp)
    if m:
        return m[-1].lower() == "yes"
    return None


def correct(pred, label):
    return pred is not None and pred == label


def mcnemar_exact(b, c):
    """p-value hai phía của kiểm định McNemar chính xác (binomial) với b, c là số cặp lệch."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="work")
    ap.add_argument("--pairs", nargs="+",
                    default=["IMG:G1", "G1:E1", "IMG:E1", "G0:G1", "E0:E1", "G0:E0"])
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.out_dir, "runs", "*", "*.jsonl")))
    if not files:
        raise SystemExit("Chưa có kết quả chạy: chạy s06 trước.")
    R = defaultdict(dict)  # R[(cond, run)][qid] = row
    for fp in files:
        for r in load_jsonl(fp):
            r["pred_old"], r["pred_new"] = score_old(r["response"]), parse_new(r["response"])
            R[(r["condition"], r["run"])][r["qid"]] = r
    conds = sorted({c for c, _ in R}, key=lambda c: ["IMG", "G0", "G1", "E0", "E1"].index(c)
                   if c in ["IMG", "G0", "G1", "E0", "E1"] else 99)
    runs = sorted({r for _, r in R})
    t1_runs = [r for r in runs if r.startswith("t1")]
    qids = sorted({q for d in R.values() for q in d})
    eval_dir = os.path.join(args.out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # ---- accuracy từng lần chạy
    per_run = []
    for (cond, run), d in sorted(R.items()):
        rows = list(d.values())
        for sc in ("old", "new"):
            rec = {"condition": cond, "run": run, "scorer": sc, "n": len(rows),
                   "correct": sum(correct(x[f"pred_{sc}"], x["label"]) for x in rows)}
            rec["acc"] = round(rec["correct"] / len(rows), 4)
            for ct in ("direct", "parallel", "sequential", "analytical"):
                sub = [x for x in rows if x["claim_type"] == ct]
                rec[f"acc_{ct}"] = round(sum(correct(x[f"pred_{sc}"], x["label"]) for x in sub) / len(sub), 4) if sub else ""
            for lab, nm in ((True, "entailed"), (False, "refuted")):
                sub = [x for x in rows if x["label"] == lab]
                rec[f"acc_{nm}"] = round(sum(correct(x[f"pred_{sc}"], x["label"]) for x in sub) / len(sub), 4) if sub else ""
            rec["unparsed"] = sum(x[f"pred_{sc}"] is None for x in rows)
            rec["old_new_disagree"] = sum((x["pred_old"] != x["pred_new"]) for x in rows)
            rec["truncated"] = sum(x.get("finish_reason") == "length" for x in rows)
            rec["errors"] = sum(bool(x.get("error")) for x in rows)
            tok = [x["output_tokens"] for x in rows if x.get("output_tokens")]
            rec["mean_output_tokens"] = round(mean(tok), 1) if tok else ""
            ptok = [x["prompt_tokens"] for x in rows if x.get("prompt_tokens")]
            rec["mean_prompt_tokens"] = round(mean(ptok), 1) if ptok else ""
            per_run.append(rec)
    write_csv(per_run, os.path.join(eval_dir, "per_run.csv"))

    # ---- độ ổn định (cách chấm new) giữa các seed temperature 1.0
    stab, majority = [], {}
    for cond in conds:
        avail = [r for r in t1_runs if (cond, r) in R]
        if not avail:
            continue
        accs = [sum(correct(R[(cond, r)][q]["pred_new"], R[(cond, r)][q]["label"]) for q in qids
                    if q in R[(cond, r)]) / len(qids) for r in avail]
        unanimous, maj = 0, {}
        for q in qids:
            preds = [R[(cond, r)][q]["pred_new"] for r in avail if q in R[(cond, r)]]
            if len(set(preds)) == 1:
                unanimous += 1
            yes = sum(p is True for p in preds); no = sum(p is False for p in preds)
            maj[q] = True if yes > no else False if no > yes else None
        majority[cond] = maj
        lab = {q: next(R[(cond, r)][q]["label"] for r in avail if q in R[(cond, r)]) for q in qids}
        maj_acc = sum(correct(maj[q], lab[q]) for q in qids) / len(qids)
        t0_acc = ""
        if (cond, "t0") in R:
            t0_acc = round(sum(correct(x["pred_new"], x["label"]) for x in R[(cond, "t0")].values()) / len(qids), 4)
        stab.append({"condition": cond, "n_seeds": len(avail),
                     "acc_mean_t1": round(mean(accs), 4), "acc_std_t1": round(pstdev(accs), 4),
                     "acc_min_t1": round(min(accs), 4), "acc_max_t1": round(max(accs), 4),
                     "unanimous_rate": round(unanimous / len(qids), 4),
                     "majority_acc": round(maj_acc, 4), "acc_t0": t0_acc})
    write_csv(stab, os.path.join(eval_dir, "stability.csv"))

    # ---- đổi kết luận giữa các cặp điều kiện
    flips, flip_summary = [], []
    for pair in args.pairs:
        a, b = pair.split(":")
        for basis in ("t0", "majority"):
            def ok(cond, q):
                if basis == "t0":
                    x = R.get((cond, "t0"), {}).get(q)
                    return None if x is None else correct(x["pred_new"], x["label"])
                lab = next((R[(cond, r)][q]["label"] for r in t1_runs if (cond, r) in R and q in R[(cond, r)]), None)
                p = majority.get(cond, {}).get(q, "NA")
                return None if p == "NA" or lab is None else correct(p, lab)
            b_cnt = c_cnt = 0
            for q in qids:
                oa, ob = ok(a, q), ok(b, q)
                if oa is None or ob is None or oa == ob:
                    continue
                if oa and not ob:
                    b_cnt += 1; kind = f"{a} đúng -> {b} sai"
                else:
                    c_cnt += 1; kind = f"{a} sai -> {b} đúng"
                ref = next(d[q] for d in R.values() if q in d)
                flips.append({"pair": pair, "basis": basis, "qid": q, "claim_type": ref["claim_type"],
                              "label": ref["label"], "flip": kind, "cause_manual": "", "note": ""})
            has_a = (a, "t0") in R if basis == "t0" else a in majority
            has_b = (b, "t0") in R if basis == "t0" else b in majority
            if has_a and has_b:
                flip_summary.append({"pair": pair, "basis": basis, f"{a}_right_{b}_wrong": b_cnt,
                                     f"{a}_wrong_{b}_right": c_cnt, "mcnemar_p": round(mcnemar_exact(b_cnt, c_cnt), 4)})
    write_csv(flips, os.path.join(eval_dir, "flips.csv"))

    # ---- bảng claim x điều kiện (lần t0, cách chấm new) để xem tay
    per_claim = []
    for q in qids:
        ref = next(d[q] for d in R.values() if q in d)
        row = {"qid": q, "claim_type": ref["claim_type"], "label": ref["label"], "mixed": ref["mixed"]}
        for cond in conds:
            x = R.get((cond, "t0"), {}).get(q)
            row[f"{cond}_t0"] = "" if x is None else ("✓" if correct(x["pred_new"], x["label"]) else "✗")
            row[f"{cond}_maj"] = "" if cond not in majority else (
                "✓" if correct(majority[cond][q], ref["label"]) else "✗")
        per_claim.append(row)
    write_csv(per_claim, os.path.join(eval_dir, "per_claim.csv"))

    # ---- tóm tắt
    n = len(qids)
    lines = [f"# Báo cáo pilot SciVer ({n} claim)\n",
             f"Mỗi claim = {100 / n:.1f} điểm %. Khoảng tin cậy 95% của accuracy ở n={n} rộng khoảng "
             f"±{196 * math.sqrt(0.25 / n):.0f} điểm: đây là pilot, KHÔNG kết luận cải thiện.\n",
             "## Accuracy theo điều kiện (cách chấm new)\n",
             "| Điều kiện | TB 3 seed (t=1) | Độ lệch | Đa số | t=0 | Nhất quán 3 seed |", "|---|---|---|---|---|---|"]
    for s in stab:
        lines.append(f"| {s['condition']} | {s['acc_mean_t1']:.3f} | {s['acc_std_t1']:.3f} | "
                     f"{s['majority_acc']:.3f} | {s['acc_t0']} | {s['unanimous_rate']:.2f} |")
    lines += ["\n## Cách chấm cũ (SciVer) so với mới\n", "| Điều kiện | Lần chạy | Acc old | Acc new | Lệch kết luận | Không đọc được (new) |",
              "|---|---|---|---|---|---|"]
    by = {(r["condition"], r["run"], r["scorer"]): r for r in per_run}
    for (cond, run) in sorted(R):
        o, w = by[(cond, run, "old")], by[(cond, run, "new")]
        lines.append(f"| {cond} | {run} | {o['acc']:.3f} | {w['acc']:.3f} | {w['old_new_disagree']} | {w['unparsed']} |")
    lines += ["\n## Đổi kết luận giữa các điều kiện\n", "| Cặp | Cơ sở | A đúng→B sai | A sai→B đúng | McNemar p |", "|---|---|---|---|---|"]
    for s in flip_summary:
        v = list(s.values())
        lines.append(f"| {v[0]} | {v[1]} | {v[2]} | {v[3]} | {v[4]} |")
    lines += ["\nGhi chú: điền cột `cause_manual` trong flips.csv (đọc ảnh / trích bảng / chuẩn hóa / suy luận / nhiễu seed)."]
    qpath = os.path.join(args.out_dir, "table_quality.csv")
    if os.path.exists(qpath):
        rows = list(csv.DictReader(open(qpath, encoding="utf-8")))
        if rows:
            m = lambda k: mean(float(r[k]) for r in rows)  # noqa: E731
            lines += [f"\n## Chất lượng trích bảng (Docling, {len(rows)} bảng)\n",
                      f"- TEDS thô: {m('teds_raw'):.3f}; sau chuẩn hóa: {m('teds_norm'):.3f}",
                      f"- TEDS-Struct: {m('teds_struct_raw'):.3f}",
                      f"- Độ khớp số (recall): {m('num_rec_raw'):.3f}; precision: {m('num_prec_raw'):.3f}",
                      f"- Docling không tìm thấy bảng: {sum(r['extract_status'] != 'ok' and not r['extract_status'].startswith('ok') for r in rows)}"]
    open(os.path.join(eval_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nĐã ghi kết quả vào {eval_dir}/")


if __name__ == "__main__":
    main()
