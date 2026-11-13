#!/usr/bin/env bash
# Môi trường riêng cho Docling (tránh xung đột phiên bản torch với vLLM). Chạy được trên CPU.
set -euo pipefail
conda create -y -n docling python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate docling
pip install -r requirements-docling.txt
python -c "from importlib.metadata import version; print('docling', version('docling'))"
