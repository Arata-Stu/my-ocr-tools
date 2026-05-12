#!/bin/sh

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

VENV_PYTHON="$(pwd)/env/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "[Error] $VENV_PYTHON が見つかりません。先に bash setup_env.sh を実行してください。"
  exit 1
fi

exec "$VENV_PYTHON" test-vlm.py "$@"
