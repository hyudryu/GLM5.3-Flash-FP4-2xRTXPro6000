# GLM5.3-Flash-FP4-2xRTXPro6000

GLM-5.3-Flash (NVFP4) served with vLLM on 2x RTX PRO 6000 Blackwell Workstation: TP2 + DCP2, MTP with 3 draft
tokens, and b12x kernels with inline compressed-scale-factor decode. The full recipe, launch script, overlay
and raw results are in [`recipes/sparkdeck-v5/`](recipes/sparkdeck-v5).

## Decode speeds

Measured with `bench/coding.py` on 2026-10-31. Each request has a ~240-token coding prompt and generates 1,536
tokens (temperature 0, thinking on, no prefix-cache hits). Aggregate = total completion tokens / batch wall time,
median across rounds (C1: 5 rounds, C4/C8: 3 rounds).

### 450 W per GPU

| Concurrency | Aggregate tok/s | Per-request tok/s | Median TTFT | MTP accepted / step |
|---|---|---|---|---|
| C1 | **188.8** | 192.9 | 180 ms | 1.64 (54.5%) |
| C2 | not measured yet | | | |
| C4 | **390.0** | 102.4 | 442 ms | 1.61 (53.6%) |
| C8 | **546.8** | 72.7 | 499 ms | 1.58 (52.6%) |

### 300 W per GPU

| Concurrency | Aggregate tok/s | Per-request tok/s | Median TTFT | MTP accepted / step |
|---|---|---|---|---|
| C1 | **170.7** | 174.0 | 175 ms | 1.59 (52.9%) |
| C2 | not measured yet | | | |
| C4 | **317.6** | 83.2 | 440 ms | 1.58 (52.5%) |
| C8 | **435.6** | 57.4 | 529 ms | 1.57 (52.3%) |

## Prefill speeds

Prefill runs one request at a time (`--max-parallel-prefills 1`), so aggregate throughput is the same at every
concurrency.

| Workload | 450 W tok/s | 300 W tok/s |
|---|---|---|
| 1 × 64K tokens | 11,156 | 8,182 |
| C1 / C4 / C8 × 8K tokens (aggregate) | 11,483 / 11,446 / 11,427 | 8,448 / 8,464 / 8,450 |

Per-length prefill results, round ranges and the reproduction commands are in the
[recipe README](recipes/sparkdeck-v5/README.md).
