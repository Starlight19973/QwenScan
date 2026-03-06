#!/bin/bash
# Qwen3.5-27B Q5_K_M via llama-server b8185 (native binary + CUDA 12.8)
set -euo pipefail

BIN_DIR=/home/testusr4/QwenScan/bin/llama-b8185
MODEL=/home/testusr4/QwenScan/models/Qwen3.5-27B-GGUF/Qwen_Qwen3.5-27B-Q5_K_M.gguf
MMPROJ=/home/testusr4/QwenScan/models/Qwen3.5-27B-GGUF/mmproj-Qwen_Qwen3.5-27B-bf16.gguf

export LD_LIBRARY_PATH="${BIN_DIR}:/usr/local/lib/ollama/cuda_v12${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${BIN_DIR}/llama-server"   --model "${MODEL}"   --mmproj "${MMPROJ}"   --host 0.0.0.0   --port 8001   --n-gpu-layers 999   --ctx-size 8192   --batch-size 2048   --ubatch-size 512   --flash-attn on   --alias document-parser   --jinja
