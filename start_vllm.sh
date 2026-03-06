#!/bin/bash
cd /home/testusr4/QwenScan
. venv/bin/activate
exec python3 -m vllm.entrypoints.openai.api_server   --model models/Qwen3-VL-8B-Instruct-AWQ-8bit   --served-model-name document-parser   --port 8001   --host 0.0.0.0   --max-model-len 32768   --gpu-memory-utilization 0.85   --limit-mm-per-prompt '{"image": 20}'   --dtype float16
