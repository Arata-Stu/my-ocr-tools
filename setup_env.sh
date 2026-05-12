#!/bin/bash

set -e

echo "[Info] CUDA 12.6 を有効化"

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PIP_CACHE_DIR="$(pwd)/.pip-cache"

mkdir -p "$PIP_CACHE_DIR"

echo "[Info] nvcc version"
nvcc -V

echo "[Info] venv 作成"

python3 -m venv env

VENV_PYTHON="$(pwd)/env/bin/python"
VENV_PIP="$VENV_PYTHON -m pip"

echo "[Info] pip 更新"

$VENV_PIP install --upgrade pip wheel "setuptools<81"

echo "[Info] setuptools version"
$VENV_PYTHON - <<'PY'
import setuptools
print(setuptools.__version__)
PY

echo "[Info] PyTorch install"

$VENV_PIP install \
  torch==2.1.2 \
  torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[Info] requirements install"

$VENV_PIP install -r requirements.txt

echo "[Info] venv python"
$VENV_PYTHON - <<'PY'
import sys
print(sys.executable)
print(sys.prefix)
PY

echo "[Info] 使い方:"
echo "[Info]   直接実行: ./run.sh IMAGE_PATH"
echo "[Info]   手動activate: . ./env.sh"

echo "[Info] 完了"
