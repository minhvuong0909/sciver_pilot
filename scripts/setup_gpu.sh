#!/usr/bin/env bash
# Cài môi trường GPU (vLLM + Qwen2.5-VL) và tải dữ liệu, model.
set -euo pipefail
conda create -y -n sciver python=3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sciver
pip install -r requirements-gpu.txt
pip install -U "huggingface_hub[cli]"

# Dataset SciVer (có ảnh -> cần git-lfs)
git lfs install
mkdir -p data
[ -d data/SciVer ] || git clone https://huggingface.co/datasets/chengyewang/SciVer data/SciVer

# Repo gốc SciVer (để chạy baseline đúng code của tác giả nếu muốn so sánh)
[ -d third_party/SciVer ] || git clone https://github.com/QDRhhhh/SciVer third_party/SciVer

# Tải trước model (~16 GB)
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct

python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
ls data/SciVer
