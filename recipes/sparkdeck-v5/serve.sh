#!/usr/bin/env bash
# GLM-5.3-Flash-FP4 on 2x RTX PRO 6000 Blackwell: TP2 + DCP2, MTP3, b12x inline-CSF v5 overlay.
# Reproduces the SparkDeck deployment "GLM-5.3-Flash-FP4 TP2 (b12x inline-CSF v5, RTX PRO 6000)" exactly as benchmarked.
# Requires iommu=pt (GPU P2P), image ghcr.io/local-inference-lab/vllm:release-10152026
#   (sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HF_HOME_HOST="${HF_HOME_HOST:-$HOME/.cache/huggingface}"
SNAPSHOT=/root/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5
SP=/opt/venv/lib/python3.12/site-packages

docker run -d --name glm53-flash-v5 --gpus all --ipc=host --shm-size=16g -p 8000:8000 \
  -v "$HF_HOME_HOST":/root/.cache/huggingface \
  -v "$HERE/overlay/b12x/moe_w4a16_kernel.py":$SP/flashinfer/experimental/b12x/moe/_shared/kernels/w4a16/kernel.py:ro \
  -v "$HERE/overlay/b12x/route_pack.py":$SP/flashinfer/experimental/b12x/moe/_shared/kernels/w4a16/route_pack.py:ro \
  -v "$HERE/overlay/b12x/fused_moe_impl.py":$SP/flashinfer/experimental/b12x/moe/fused_moe/_impl.py:ro \
  -v "$HERE/overlay/b12x/fused_moe_preparation.py":$SP/flashinfer/experimental/b12x/moe/fused_moe/_preparation.py:ro \
  -v "$HERE/overlay/b12x/quant/nvfp4_csf.py":$SP/flashinfer/experimental/b12x/_lib/quant/nvfp4_csf.py:ro \
  -v "$HERE/overlay/b12x/quant/csf_fast.py":$SP/flashinfer/experimental/b12x/_lib/quant/csf_fast.py:ro \
  -v "$HERE/overlay/b12x/quant/csf_blocks_tuning.py":$SP/flashinfer/experimental/b12x/_lib/quant/csf_blocks_tuning.py:ro \
  -e B12X_DYNAMIC_DIRECT_EXPERT_SCALES=1 \
  -e B12X_DYNAMIC_SKIP_SPLIT_BARRIER_RESET=1 \
  -e B12X_DYNAMIC_SPLIT_COMPUTE_MAC=224 \
  -e B12X_DYNAMIC_SPLIT_FAST_PREPARE=1 \
  -e B12X_DYNAMIC_SPLIT_LOW_SMEM=1 \
  -e B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE=1 \
  -e B12X_DYNAMIC_WORK_SOURCE=persistent_grid \
  -e B12X_MHC_PDL=1 \
  -e B12X_PCIE_ONESHOT_BLOCK_LIMIT=4 \
  -e B12X_PCIE_ONESHOT_PDL=1 \
  -e B12X_PCIE_ONESHOT_THREADS=512 \
  -e B12X_W4A16_A4_PREFILL=1 \
  -e B12X_W4A16_FP32_TOPK_WEIGHTS=1 \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUTE_DSL_ARCH=sm_120a \
  -e GLOO_SOCKET_IFNAME=lo \
  -e INSTANTTENSOR_BACKEND=BUFFERED \
  -e NCCL_BUFFSIZE=1048576 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_MAX_NCHANNELS=2 \
  -e NCCL_MIN_NCHANNELS=2 \
  -e NCCL_NET_PLUGIN=none \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e NCCL_SOCKET_IFNAME=lo \
  -e NCCL_TUNER_PLUGIN=none \
  -e OMP_NUM_THREADS=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,large_segment_size_mb:12 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e TORCH_CUDA_ARCH_LIST=12.0a \
  -e VLLM_B12X_MLA_CKV_GATHER=1 \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
  -e VLLM_CAUSAL_CONV1D_UPDATE_HOIST=1 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_GLM53_DFLASH_ATTN=1 \
  -e VLLM_GLM53_EMBED_HOST=1 \
  -e VLLM_GLM53_KDA_GATE_SIDE_STREAM=1 \
  -e VLLM_GLM53_KDA_MXFP8_ACTIVATION_MODE=prefill \
  -e VLLM_GLM53_L2_PREFETCH=1 \
  -e VLLM_GLM53_L2_PREFETCH_PERSIST_MB=0 \
  -e VLLM_GLM53_MTP_DRAFT_HEAD=bf16 \
  -e VLLM_GLM53_ONLINE_DENSE_MXFP8=0 \
  -e VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=auto \
  -e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=2048 \
  -e VLLM_LM_HEAD_A16=1 \
  -e VLLM_MTP_NVFP4_LM_HEAD=0 \
  -e VLLM_MXFP8_LM_HEAD=0 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e VLLM_PCIE_DMA_MIN_BYTES=6MB \
  -e VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE=off \
  -e VLLM_PLUGINS='' \
  -e VLLM_SHARE_PYNCCL_COMMS=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e B12X_W4A16_CSF_INLINE=1 \
  -e B12X_CSF_FAST=1 \
  -e B12X_CSF_FUSE_PACK=1 \
  -e B12X_FC2_PREFETCH_BLOCKS=4 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  ghcr.io/local-inference-lab/vllm:release-10152026 \
  vllm serve "$SNAPSHOT" \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.985 \
    --served-model-name glm-5.3-flash \
    --additional-config '{"glm53_kda_decode_backend":"auto","kda_prefill_backend":"b12x"}' \
    --attention-backend B12X \
    --block-size 256 \
    --chat-template /opt/lil/runtime/templates/glm53-flash.jinja \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --cp-kv-cache-interleave-size 4 \
    --cudagraph-capture-sizes 1 2 4 8 12 16 20 24 28 32 \
    --dcp-kv-cache-interleave-size 4 \
    --decode-context-parallel-size 2 \
    --default-chat-template-kwargs '{"reasoning_effort":"max","clear_thinking":false}' \
    --dtype bfloat16 \
    --enable-auto-tool-choice \
    --enable-chunked-prefill \
    --no-enable-flashinfer-autotune \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --gpu-memory-utilization 0.985 \
    --kv-cache-memory-bytes 7650410496 \
    --linear-backend b12x \
    --load-format instanttensor \
    --mamba-cache-mode align \
    --max-parallel-prefills 1 \
    --moe-backend b12x \
    --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
    --prefill-compute-share 0.4 \
    --prefill-schedule-interval 1 \
    --quantization modelopt_mixed \
    --reasoning-parser glm45 \
    --recurrent-checkpoint-policy request_boundaries \
    --tool-call-parser glm47 \
    --revision 76348ddb2be2e35e39ecfc00e75dc07703f4f3d5 \
    --max-model-len 524288 \
    --max-num-seqs 8 \
    --kv-cache-dtype fp8 \
    -tp 2 \
    -pp 1 \
    --max-cudagraph-capture-size 32 \
    --max-num-batched-tokens 4096 \
    --speculative-config '{"method":"mtp","draft_sample_method":"probabilistic","rejection_sample_method":"standard","moe_backend":"b12x","attention_backend":"B12X","num_speculative_tokens":3}'
