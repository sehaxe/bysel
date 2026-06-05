#!/usr/bin/env bash
# 🛸 busel auto-setup — detects GPU and runs `uv sync --extra <match>` + maturin.
set -euo pipefail

EXTRA="${1:-}"

detect() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "cu130"
        return
    fi
    if [ "$(uname -m)" = "Darwin" ] || [ "$(uname -s)" = "MINGW"* ] || [ "$(uname -s)" = "CYGWIN"* ]; then
        echo "cpu"
        return
    fi
    echo "cpu"
}

if [ -z "$EXTRA" ]; then
    EXTRA=$(detect)
fi

case "$EXTRA" in
    cpu|cu118|cu126|cu128|cu130) ;;
    *)
        echo "Unknown extra: $EXTRA"
        echo "Valid extras: cpu cu118 cu126 cu128 cu130"
        echo "AMD ROCm is currently broken upstream (pytorch-triton-rocm 3.x dep). Use cpu on AMD for now."
        exit 1
        ;;
esac

echo "🛸 busel setup → using extra: $EXTRA"
uv sync --extra "$EXTRA"
uv run maturin develop --release
echo "✅ done. Run: uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"

