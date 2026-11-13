#!/usr/bin/env bash
# Chạy pilot theo 2 giai đoạn:
#   bash scripts/run_pilot.sh            -> chọn mẫu + dựng bảng chuẩn, rồi DỪNG để kiểm tay
#   bash scripts/run_pilot.sh --after-gold -> Docling, chuẩn hóa, chạy Qwen, đánh giá
# Sửa DATA_FILE theo tên file validation thật trong data/SciVer.
set -euo pipefail
DATA_ROOT=data/SciVer
DATA_FILE=${DATA_FILE:-data/SciVer/validation.json}
WORK=work
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ "${1:-}" != "--after-gold" ]; then
  conda activate sciver
  python src/s01_select_samples.py --data_root $DATA_ROOT --data_file $DATA_FILE --per_subset 10 --seed 2026 --out $WORK/samples.json
  python src/s02_inspect_tables.py --data_root $DATA_ROOT --samples $WORK/samples.json
  python src/s03_build_gold.py     --data_root $DATA_ROOT --samples $WORK/samples.json --out_dir $WORK
  echo ">>> Kiểm tay work/gold_review.csv (và sửa work/gold/*.json nếu sai), rồi chạy: bash scripts/run_pilot.sh --after-gold"
  exit 0
fi

conda activate docling
python src/s04_extract_docling.py --data_root $DATA_ROOT --samples $WORK/samples.json --out_dir $WORK
python src/s05_normalize_and_score_tables.py --out_dir $WORK

conda activate sciver
python src/s06_run_verifier.py --data_root $DATA_ROOT --samples $WORK/samples.json --out_dir $WORK --dry_run
python src/s06_run_verifier.py --data_root $DATA_ROOT --samples $WORK/samples.json --out_dir $WORK \
       --conditions IMG G0 G1 E0 E1 --model Qwen/Qwen2.5-VL-7B-Instruct
python src/s07_evaluate.py --out_dir $WORK
