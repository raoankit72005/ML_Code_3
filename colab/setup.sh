#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MATCHER="${1:-cuda}"
case "$MATCHER" in cuda|cpu) ;; *) echo 'Use cuda or cpu'; exit 1;; esac
command -v nvidia-smi >/dev/null
nvidia-smi
# Dedicated interpreter: leave Colab's notebook packages and existing Torch untouched.
COLAB_ER_ENV=/content/ml_er_env
python3 -m venv "$COLAB_ER_ENV"
COLAB_ER_PY="$COLAB_ER_ENV/bin/python"
"$COLAB_ER_PY" -m pip install --upgrade pip
"$COLAB_ER_PY" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
"$COLAB_ER_PY" -m pip install -r requirements-hybrid.txt
"$COLAB_ER_PY" -c 'import torch; assert torch.cuda.is_available(), "Select a Colab GPU runtime first"; print(torch.cuda.get_device_name(0))'
if [ "$MATCHER" = cuda ]; then
  command -v nvcc >/dev/null || { echo 'CUDA toolkit missing; choose cpu matcher explicitly or use a CUDA development runtime.'; exit 1; }
  command -v g++ >/dev/null
  "$COLAB_ER_PY" -m pip install 'cmake>=3.28,<4' ninja
  COLAB_ER_ARCH=$("$COLAB_ER_PY" -c 'import torch; a,b=torch.cuda.get_device_capability();print(f"{a}{b}")')
  export CMAKE_BUILD_PARALLEL_LEVEL=2
  "$COLAB_ER_PY" -m pip install --force-reinstall --no-deps --no-cache-dir --no-binary=lightgbm \
    --config-settings=cmake.define.USE_CUDA=ON \
    --config-settings="cmake.define.CMAKE_CUDA_ARCHITECTURES=$COLAB_ER_ARCH" lightgbm==4.6.0
fi
# Test the actual requested backend before any dataset processing.
"$COLAB_ER_PY" - "$MATCHER" <<'PY'
import sys,numpy as np,lightgbm as lgb
rng=np.random.default_rng(7);x=rng.normal(size=(256,4))
lgb.train(dict(objective='binary',device_type=sys.argv[1],verbosity=-1,num_threads=2,max_bin=63,min_data_in_leaf=5),lgb.Dataset(x,label=(x[:,0]>0)),num_boost_round=2)
print('LightGBM backend passed:',sys.argv[1])
PY
"$COLAB_ER_PY" -m pip freeze > /content/ml_er_environment.txt
