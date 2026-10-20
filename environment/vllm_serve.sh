#!/usr/bin/env bash
# Exact launch command of the inference server (run as a systemd service).
# Harmony formatting (analysis/final channels) is handled by vLLM.
exec vllm serve openai/gpt-oss-20b \
  --served-model-name gpt-oss-20b \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --host 0.0.0.0 --port 8000
