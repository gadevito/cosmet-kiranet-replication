# Inference machine (Sec. V-A of the paper)

A desktop workstation the company ALREADY operated for local inference of the
same model before the experience (Sec. IV: gpt-oss-20B was part of Kiranet's
stack; Sec. VIII-E: hardware acquisition is therefore excluded from the
economics). For the study it was reserved to the measurement service during
evaluation runs (single request at a time). It sits on the company LAN with no
inbound access from outside; only the inference port is exposed to the LAN.

| Component | Specification |
|-----------|---------------|
| GPU       | NVIDIA GeForce RTX 5090, 32 GB GDDR7 (Blackwell, sm_120) |
| CPU       | AMD Ryzen 9 9950X (16 cores) |
| RAM       | 128 GB DDR5 |
| Storage   | 2 TB NVMe SSD, PCIe 5.0 |
| PSU       | 1200 W |
| OS        | Ubuntu 24.04 LTS |
| Driver / CUDA | NVIDIA 580.x / CUDA 12.8 |
| Price     | approx. EUR 6,000-8,000 excl. VAT (autumn 2025) -- what an organization buying it for measurement would pay; not charged to this adoption |

Availability check: RTX 5090 launched 30 Jan 2025; gpt-oss released 5 Aug 2025;
vLLM support for gpt-oss/Harmony from 0.10.1 (Aug 2025). The whole stack
therefore pre-dates the start of the experience (Nov 2025).

## Deployment facts (Sec. V-A)

* Endpoint: vLLM's OpenAI-compatible `POST /v1/chat/completions`; Harmony formatting (analysis and
  final channels) handled by vLLM; `reasoning_effort` sent through the client's `extra_body`.
* No fine-tuning: the released `openai/gpt-oss-20b` weights (revision 6cee5e8, native MXFP4, ≈13 GB
  on disk) are used as published.
* Network: the server listens on the LAN only; no inbound access from outside; `vllm_serve.sh` and
  `client_example.py` are the launch command and a sanitized client (no credentials, `localhost`
  in place of the internal address).
* During the evaluation runs the workstation was reserved to the measurement service and requests were
  issued one at a time (`concurrency_during_evaluation: 1`), although the server allows 4 concurrent
  sequences.
* Timing figures are measured client-side on an already running (warm) server with the few-shot
  prefix cached; automated time only, human review excluded.

## Throughput and latency (how the paper's figures were measured)

* **~200 tok/s** = single-stream *decode* throughput:
  `usage.completion_tokens` (vLLM counts reasoning tokens here) divided by the
  generation time, averaged over Analyzer requests. Not prefill, not batched
  throughput. Measured at the prompt lengths actually used (8-12k tokens for the
  Analyzer: instructions + 12 worked examples + the split use case; ~3k for the Splitter). Independent measurements on the
  same GPU/model report 320-420 tok/s at 8k and ~200 tok/s at 16k context, so
  200 is conservative.
* **2-5 s per functional process** = end-to-end wall-clock measured client-side,
  from the first request to the parsed CFP: Splitter call (amortised over the
  use case) + Analyzer call, including prefill, generation and parsing. Warm
  server, few-shot prefix cached (prefill < 0.3 s). At reasoning effort
  `medium` a functional process yields 400-900 generated tokens: 2-4.5 s at
  200 tok/s, plus prefill. Human review time is NOT included.
* `--max-model-len 131072` is the server's capacity, not the working length;
  our prompts never approach it (generation speed on a consumer GPU degrades
  sharply near the 131k limit).
