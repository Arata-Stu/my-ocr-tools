#!/bin/sh

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

if [ ! -f "env/bin/activate" ]; then
  echo "[Error] env/bin/activate が見つかりません。先に bash setup_env.sh を実行してください。"
  return 1 2>/dev/null || exit 1
fi

. env/bin/activate

echo "[Info] venv を有効化しました"
python - <<'PY'
import sys
print(sys.executable)
print(sys.prefix)
PY
