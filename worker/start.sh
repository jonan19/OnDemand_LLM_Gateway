#!/usr/bin/env bash
# Entrypoint for the llama.cpp worker container (CPU and CUDA variants share this script).
# Translates env vars into a llama-server invocation. The model itself is NOT baked into
# the image -- llama-server's built-in `-hf` downloader fetches it from Hugging Face into
# $LLAMA_CACHE on first run and reuses it on every later run (mount a volume there).
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-TheBloke/Mistral-7B-Instruct-v0.2-GGUF}"
MODEL_QUANT="${MODEL_QUANT:-Q4_K_M}"
CTX_SIZE="${CTX_SIZE:-4096}"
N_GPU_LAYERS="${N_GPU_LAYERS:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

echo "[worker] model=${MODEL_REPO}:${MODEL_QUANT} ctx=${CTX_SIZE} ngl=${N_GPU_LAYERS} cache=${LLAMA_CACHE:-unset} host=${HOST} port=${PORT}"

exec llama-server \
    -hf "${MODEL_REPO}:${MODEL_QUANT}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --ctx-size "${CTX_SIZE}" \
    --n-gpu-layers "${N_GPU_LAYERS}"
