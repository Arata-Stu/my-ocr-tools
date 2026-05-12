#!/bin/bash

set -e

echo "[Info] CUDA 12.6 を有効化"

export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "[Info] nvcc version"
nvcc -V

echo "[Info] venv 作成"

python3 -m venv env

source env/bin/activate

echo "[Info] pip 更新"

pip install --upgrade pip setuptools wheel

echo "[Info] PyTorch install"

pip install \
  torch==2.1.2 \
  torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[Info] requirements install"

pip install -r requirements.txt

echo "[Info] flash-attn install"

MAX_JOBS=1 pip install flash-attn==2.5.8 --no-build-isolation

echo "[Info] 完了"