# GLM-5.3-Flash-FP4 on 2x RTX PRO 6000 — TP2 + DCP2, MTP3, b12x inline-CSF v5

The vLLM deployment used on the 2x RTX PRO 6000 Blackwell Workstation workstation, captured from the
running container (SparkDeck deployment *GLM-5.3-Flash-FP4 TP2 (b12x inline-CSF v5, RTX PRO 6000)*).

## Recipe

| | |
|---|---|
| Model | `nvidia/GLM-5.3-Flash-FP4-Local` @ `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` (`--quantization modelopt_mixed`) |
| Image | `ghcr.io/local-inference-lab/vllm:release-10152026` (`sha256:745feb91…dac0fb29`) |
| Parallelism | TP2, decode-context-parallel 2 (`--dcp-kv-cache-interleave-size 4`) |
| Speculative | MTP, 3 draft tokens, probabilistic draft sampling, standard rejection sampling |
| Backends | b12x attention / MoE / linear, PCIe one-shot all-reduce (b12x) |
| KV cache | FP8, 7.65 GB per GPU (`--kv-cache-memory-bytes 7650410496`), block size 256, prefix caching |
| Limits | `--max-model-len 524288`, `--max-num-seqs 8`, `--max-num-batched-tokens 4096`, `--max-parallel-prefills 1` |
| Overlay | 7 b12x source files in [`overlay/`](overlay) (inline compressed-scale-factor decode in the W4A16 MoE kernel, fused CSF expansion + pack, FC2 prefetch) |
| Host | Linux 7.0, NVIDIA driver 595.99.02, **`iommu=pt` required** for GPU P2P (TP2 hangs at NCCL init otherwise) |

Launch: [`serve.sh`](serve.sh) (`HF_HOME_HOST` defaults to `~/.cache/huggingface`; listens on :8000, model name `glm-5.3-flash`).

## Speeds at 300 W per GPU

Measured 2026-10-31 with both GPUs at a 300 W enforced power limit, server otherwise idle
(verified: no competing requests; benchmark request counts match backend counters).

### Decode — `bench/coding.py`

Coding prompts (~240 prompt tokens), 1536 output tokens per request (all length-capped), temperature 0,
thinking on (template default), unique cache salt per request (0 cached tokens). C1: 5 rounds; C4/C8: 3 rounds.

| Concurrency | Aggregate output tok/s (median) | Rounds | Per-request decode tok/s | Median TTFT | MTP accepted / step |
|---|---|---|---|---|---|
| C1 | **170.7** | 169.4 – 176.0 | 174.0 | 175 ms | 1.59 (52.9%) |
| C4 | **317.6** | 314.2 – 329.1 | 83.2 | 440 ms | 1.58 (52.5%) |
| C8 | **435.6** | 428.7 – 438.2 | 57.4 | 529 ms | 1.57 (52.3%) |

Aggregate = total completion tokens / batch wall time (includes prefill and client overhead).

### Prefill — `bench/prefill.py`

Unique random-token prompts sent as token IDs, `max_tokens=1`, zero cached tokens verified. Median of 3 reps.

| Workload | TTFT | Prefill tok/s |
|---|---|---|
| 1 × 1K tokens | 0.163 s | 6,294 |
| 1 × 4K tokens | 0.482 s | 8,504 |
| 1 × 16K tokens | 1.960 s | 8,358 |
| 1 × 64K tokens | 8.010 s | 8,182 |
| C1 × 8K tokens | 0.970 s | **8,448** |
| C4 × 8K tokens | 3.871 s (last) | **8,464** (aggregate) |
| C8 × 8K tokens | 7.756 s (last) | **8,450** (aggregate) |

Prefill is serialized (`--max-parallel-prefills 1`), so aggregate prefill throughput is flat across C1/C4/C8
and the last request's TTFT scales linearly with concurrency.

Raw outputs: [`results/300w-2026-10-31/`](results/300w-2026-10-31).

## Reproduce

```bash
sudo nvidia-smi -i 0,1 -pl 300
./serve.sh
python3 bench/coding.py --base-url http://127.0.0.1:8000 --model glm-5.3-flash --metrics-port 8000 --concurrency 1 --rounds 5 --output out/decode-c1
python3 bench/coding.py --base-url http://127.0.0.1:8000 --model glm-5.3-flash --metrics-port 8000 --concurrency 4 8 --rounds 3 --output out/decode-c4c8
python3 bench/prefill.py --base-url http://127.0.0.1:8000 --label w300 --lengths 1024 4096 16384 65536 --reps 3 --concurrent 1 4 8 --concurrent-length 8192 --output out/prefill
```
