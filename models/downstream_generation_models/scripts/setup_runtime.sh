#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_PY="/home/avis/pslg-nilm/.venv-generation-torch/bin/python"
TARGET="$ROOT/../../runtime/generation/torch_site"

mkdir -p "$TARGET" "$ROOT/results" "$ROOT/data" "$ROOT/.envs"

"$TORCH_PY" -m pip install --upgrade --target "$TARGET" --no-deps \
  einops==0.8.2 loguru==0.7.3 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 \
  linear-attention-transformer==0.19.1 local-attention==1.11.2 \
  axial-positional-embedding==0.3.12 product-key-memory==0.2.11 einops-exts==0.0.4 \
  hyper-connections==0.1.15 linformer==0.2.3 colt5-attention==0.11.1 tqdm==4.67.1

# ImagenTime imports torchaudio even when delay embedding rather than STFT is used.
"$TORCH_PY" -m pip install --upgrade --target "$TARGET" --no-deps \
  --index-url https://download.pytorch.org/whl/cu118 torchaudio==2.5.1+cu118

echo "Torch overlay installed at $TARGET"
echo "TimeGAN legacy env is optional and isolated. Create it with:"
echo "  conda env create -p $ROOT/../../runtime/envs/timegan -f $ROOT/environment/timegan-legacy.yml"
