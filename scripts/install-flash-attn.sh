#!/usr/bin/env bash
# Build flash-attn for current torch/CUDA (4070 Ti promote → seq 2048).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== flash-attn install (Unsloth 2048 path) ==="
uv run --package llm-train --extra unsloth python3 -c "
import torch
print(f'torch={torch.__version__} cuda={torch.version.cuda}')
"

export MAX_JOBS="${MAX_JOBS:-4}"
export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-TRUE}"

echo "Building flash-attn (10–20 min). Requires nvcc + ~8 GiB free RAM."
uv pip install flash-attn --no-build-isolation

uv run --package llm-train --extra unsloth python3 -c "
from llm_train.flash_attn import flash_attn_available
assert flash_attn_available(), 'flash_attn import failed'
print('FA2: OK')
"

echo "Re-run: uv run --package llm-train train-preflight --promote"
echo "Expect: FA2=True padding-free seq<=2048"
