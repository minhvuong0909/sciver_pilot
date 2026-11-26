#!/usr/bin/env bash
# Lần chạy GPU đầu tiên: so B0 (code SciVer gốc), IMG và G1 trên 291 claim bảng validation.
#
# Cần: Linux, GPU NVIDIA ≥ 24 GB, môi trường "sciver" (bash scripts/setup_gpu.sh), dữ liệu
# validation có ảnh (python scripts/download_sciver_val.py --root data/raw/SciVer --images).
#
# Chạy:   bash scripts/run_gpu_x1.sh
# Biến môi trường (tùy chọn):
#   CFG=configs/experiments/x1_gpu_img_g1.yaml   cấu hình
#   SMOKE_ONLY=1     chỉ chạy thử 5 claim (đo bộ nhớ, thời gian) rồi dừng
#   SKIP_SMOKE=1     bỏ bước chạy thử
#   SKIP_B0=1        không chạy B0
#   SCIVER_DIR=third_party/SciVer                 repo SciVer gốc
#
# Chạy lại đúng lệnh này sau khi bị ngắt: verify tiếp tục từ chỗ dừng; B0 bỏ qua nếu đã xong.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=${CFG:-configs/experiments/x1_gpu_img_g1.yaml}
SCIVER_DIR=${SCIVER_DIR:-third_party/SciVer}
SCIVER_COMMIT=0a97282
mkdir -p logs
LOG="logs/run_gpu_x1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "=== [$(date '+%F %T')] $*"; }
run() { echo "    \$ $*"; local t0=$SECONDS; "$@"; echo "    xong sau $((SECONDS - t0))s"; }

step "0. Kiểm tra môi trường"
echo "nhánh: $(git rev-parse --abbrev-ref HEAD) | commit: $(git rev-parse --short HEAD) | thay đổi chưa commit: $(git status --porcelain | wc -l) file"
echo "cấu hình: $CFG | log: $LOG"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python - <<'PY'
import torch, vllm, transformers
assert torch.cuda.is_available(), "torch không thấy GPU"
mem = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"vllm {vllm.__version__} | transformers {transformers.__version__} | torch {torch.__version__} | "
      f"GPU {torch.cuda.get_device_name(0)} {mem:.0f} GB x {torch.cuda.device_count()}")
if vllm.__version__ != "0.9.0.1":
    print("CẢNH BÁO: vLLM khác 0.9.0.1 của SciVer; ghi lại trong VERIFICATION_LOG")
if mem < 23:
    print("CẢNH BÁO: GPU < 24 GB; có thể OOM với max_model_len 32768")
PY
pip install -q -e . >/dev/null
DATA_ROOT=$(python -c "from sciverx.config import load_config; print(load_config('$CFG')['paths']['data_root'])")
if [ ! -s "$DATA_ROOT/valset.json" ]; then
  echo "Thiếu $DATA_ROOT/valset.json. Chạy: python scripts/download_sciver_val.py --root $DATA_ROOT --images"
  exit 1
fi

step "1. Chuẩn bị dữ liệu (CPU): chọn 291 claim, bảng chuẩn G0 → G1"
for s in select sources gold normalize; do run python -m sciverx "$s" --config "$CFG"; done

step "2. Đếm độ dài prompt (không cần GPU)"
run python -m sciverx count-tokens --config "$CFG"

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  step "3. Chạy thử 5 claim (đo bộ nhớ và thời gian)"
  run python -m sciverx verify --config "$CFG" --limit 5
  nvidia-smi --query-gpu=memory.used --format=csv
  if [ "${SMOKE_ONLY:-0}" = "1" ]; then
    echo "SMOKE_ONLY=1: dừng ở đây. Xem outputs/runs/*_x1_gpu_img_g1_* (thư mục có --limit) để kiểm đầu ra."
    exit 0
  fi
fi

step "4. Verify đầy đủ: IMG và G1 × 4 lượt (chạy tiếp được nếu bị ngắt)"
run python -m sciverx verify --config "$CFG"

if [ "${SKIP_B0:-0}" != "1" ]; then
  step "5. B0: code SciVer gốc trên đúng 291 claim"
  [ -d "$SCIVER_DIR" ] || git clone https://github.com/QDRhhhh/SciVer "$SCIVER_DIR"
  (cd "$SCIVER_DIR" && git fetch -q --all && git checkout -q "$SCIVER_COMMIT" && echo "SciVer @ $(git rev-parse --short HEAD)")
  run python -m sciverx b0-subset --config "$CFG"
  # shellcheck disable=SC1091
  source data/processed/validation/b0/paths.sh
  if [ -s "$RAW_JSON" ]; then
    echo "B0 đã có kết quả ($RAW_JSON), bỏ qua. Xóa file này nếu muốn chạy lại."
  else
    mkdir -p "$RAW_DIR"
    # Không dùng --overwrite (cờ này hoạt động ngược tên trong main.py gốc); --max_num -1 = mọi claim
    (cd "$SCIVER_DIR" && run python main.py --model "$B0_MODEL" --prompt cot \
        --data_path "$SUBSET" --output_dir "$RAW_DIR" --max_num -1)
  fi
  run python -m sciverx b0-import --config "$CFG"
fi

step "6. Báo cáo: B0 / IMG / G1"
run python -m sciverx report --config "$CFG"
echo
echo "Log: $LOG"
