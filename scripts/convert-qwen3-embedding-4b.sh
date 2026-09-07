#!/usr/bin/env bash
# convert-qwen3-embedding-4b.sh — Reproduce the production MLX embedding model.
#
# Why this exists: the official Qwen/Qwen3-Embedding-4B checkpoint ships
# backbone-only weights (no `model.` prefix, no LM head) under a
# Qwen3ForCausalLM config, so stock `mlx_lm convert` fails with
# "Received N parameters not in model". mlx-community worked around it with
# a custom DWQ conversion — but our standard affine 4-bit build of the same
# weights measures BOTH better recall (84% vs 80%) and faster kernels
# (22.1 vs 18.7 texts/s batch-32). See docs/benchmarks/all-mlx-results.md.
#
#   scripts/convert-qwen3-embedding-4b.sh
#
# Output: ~/.cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine (idempotent)
set -euo pipefail

OUT="$HOME/.cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine"
FIXED="$HOME/.cache/qmd/models/qwen3-embedding-4b-hf-fixed"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -d "$OUT" ]; then
  echo "exists, skipping: $OUT"
  exit 0
fi

# 1. Full snapshot (mlx_lm refuses incomplete snapshots at save time).
"$REPO_DIR/.venv/bin/python" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-Embedding-4B'))
"

# 2. Remap backbone-only names -> model.-prefixed (strict loader requirement).
"$REPO_DIR/.venv/bin/python" - <<'EOF'
import json, glob, os
from safetensors import safe_open
from safetensors.torch import save_file

snap = glob.glob(os.path.expanduser(
    '~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/*'))[0]
out = os.path.expanduser('~/.cache/qmd/models/qwen3-embedding-4b-hf-fixed')
os.makedirs(out, exist_ok=True)

idx = json.load(open(snap + '/model.safetensors.index.json'))
shards = sorted(set(idx['weight_map'].values()))
new_idx = {"metadata": idx.get("metadata", {}), "weight_map": {}}
for old, shard in idx['weight_map'].items():
    new_idx['weight_map'][old if old.startswith('model.') else 'model.' + old] = shard

for shard in shards:
    tensors = {}
    with safe_open(snap + '/' + shard, framework='pt') as sf:
        for k in sf.keys():
            tensors[k if k.startswith('model.') else 'model.' + k] = sf.get_tensor(k).contiguous()
    save_file(tensors, out + '/' + shard, metadata={"format": "pt"})
    del tensors

json.dump(new_idx, open(out + '/model.safetensors.index.json', 'w'), indent=1)
for f in os.listdir(snap):
    if f.endswith(('.json', '.txt', '.jinja')) and 'safetensors' not in f:
        import shutil
        shutil.copy(snap + '/' + f, out + '/' + f)
print('remapped checkpoint at', out)
EOF

# 3. Standard affine 4-bit quant (same recipe as the reranker).
"$REPO_DIR/.venv/bin/python" -m mlx_lm convert \
  --hf-path "$FIXED" \
  --mlx-path "$OUT" \
  -q --q-bits 4 --q-group-size 64 --dtype bfloat16
echo "done: $OUT"
