#!/bin/bash
# MiniMax-M3-MXFP4 AgentX 服务（TP4 + EAGLE3），profiler 预先挂好、由 /start_profile /stop_profile 控制。
# 参数照 ATOM recipes/MiniMax-M3-Agentic-InferenceX.md；gfx1250（MI455X）多加 gfx1250 那组环境变量。
# 在容器里运行（capture_agentx.sh 调用）: CONC=10 PROFILE_DIR=/out/traces bash m3_server.sh /out/server.log
# 环境变量:
#   CONC（必填）            并发，决定 max-num-seqs（2*CONC）、CONC>=15 开 indexer CP、CONC>=40 开 LMCache CPU 层
#   PROFILE_DIR（必填）     trace 目录，每个 rank 写到 rank_N/
#   SHAPES=0|1              1 = 录 shape（ATOM_PROFILER_RECORD_SHAPES=1），只用于单独的 shapes 窗口；计时窗口保持 0
#   MODEL / DRAFT           默认 /models/MiniMax-M3-MXFP4、/models/MiniMax-M3-EAGLE3-GQA
#   KV=fp8  GPUS=（ROCR_VISIBLE_DEVICES）  GPU_MEM=0.95  MAX_SEQS  SERVER_PORT=8890  ENGINE_PORT=8896
#   ACCEPT_RATE=0.5933      固定 EAGLE3 接受率（AgentX 性能测试的做法）；置空 = 真实接受率
#   OFFLOAD=1               CONC>=40 时的 LMCache CPU 层；0 = 不开
#   ROCTRACER_SHIFT=auto|0  镜像里有 roctracer 校准脚本时，GPU 时间戳比 host 快的节点用 shim 往回拨
#   EXTRA_ENV="A=1 B=2"     额外的环境变量（例如某些镜像需要 ROCPROFILER_REGISTER_FORCE_LOAD=0 才录得到 GPU kernel）
set -u
CONC=${CONC:?set CONC}
PROFILE_DIR=${PROFILE_DIR:?set PROFILE_DIR}
LOG=${1:?server log path}
MODEL=${MODEL:-/models/MiniMax-M3-MXFP4}
DRAFT=${DRAFT:-/models/MiniMax-M3-EAGLE3-GQA}
KV=${KV:-fp8}
SERVER_PORT=${SERVER_PORT:-8890}
ENGINE_PORT=${ENGINE_PORT:-8896}
ARCH=${ARCH:-$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')}
case "$ARCH" in gfx950|gfx1250) ;; *) echo "unsupported ARCH='$ARCH', set ARCH=gfx950|gfx1250"; exit 1;; esac
ATOM_DIR=${ATOM_DIR:-/app/ATOM}
[ -d "$ATOM_DIR" ] || ATOM_DIR=$(python3 -c 'import importlib.util as u, os; s = u.find_spec("atom"); print(os.path.dirname(os.path.dirname(s.origin)) if s else "")' 2>/dev/null)
[ -d "$ATOM_DIR" ] || { echo "ATOM not found in this container (set ATOM_DIR)"; exit 1; }
cd "$ATOM_DIR"
rm -rf /root/.cache/atom/*
mkdir -p "$PROFILE_DIR"

envs=()
[ "$ARCH" = gfx1250 ] && envs+=(
  PYTORCH_ROCM_ARCH=gfx1250 GPU_ARCHS=gfx1250 HSA_OVERRIDE_GFX_VERSION=12.5.0
  ATOM_MOE_GU_ITLV=1 AITER_ROPE_TRITON_BACKEND=1 AITER_USE_SYSTEM_TRITON=1)
[ -n "${GPUS:-}" ] && envs+=(ROCR_VISIBLE_DEVICES=$GPUS)
envs+=(
  NCCL_IB_DISABLE=1 RCCL_IB_DISABLE=1
  AITER_QUICK_REDUCE_QUANTIZATION=${QR:-INT4} AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0
  ATOM_FORCE_ATTN_TRITON=1 AITER_LOG_LEVEL=WARNING ATOM_GC_THRESHOLD=20000,50,50
  ATOM_PA_FLYDSL=${FLY:-1}
)
# profiler：每步 annotation 带 bs/tok/ctx/sqsq/sqsk/sk（分析脚本靠它推导 attention 的工作量和 GEMM 的 M）；
# 计时窗口不录 shape、不录 stack（窗口小、开销低）；导出等待上限放宽到 15 分钟（大窗口导出要几十秒）
envs+=(ATOM_ENABLE_DETAILED_ANNOTATION=1 ATOM_PROFILER_MORE=0 ATOM_TORCH_PROFILER_DIR="$PROFILE_DIR"
       ATOM_PROFILER_TIMEOUT=${PROFILER_TIMEOUT:-900})
[ "${SHAPES:-0}" = 1 ] && envs+=(ATOM_PROFILER_RECORD_SHAPES=1 ATOM_PROFILER_WITH_STACK=0 ATOM_PROFILER_PROFILE_MEMORY=0)
if [ "${ROCTRACER_SHIFT:-auto}" != 0 ] && [ -f /scripts/agentic/roctracer_calib.sh ] && [ -f /scripts/agentic/libroctracer_shift.so ]; then
  shift_ns=$(CALIB_GPUS=${GPUS:-0,1,2,3} bash /scripts/agentic/roctracer_calib.sh)
  [ "${shift_ns:-0}" != 0 ] && envs+=(LD_PRELOAD=/scripts/agentic/libroctracer_shift.so ROCTRACER_TS_SHIFT_NS=$shift_ns)
fi
[ "$CONC" -ge 15 ] && envs+=(ATOM_M3_INDEXER_CP=1)
args=(--torch-profiler-dir "$PROFILE_DIR")
[ -n "${ACCEPT_RATE-0.5933}" ] && args+=(--spec-decode-acceptance-rate "${ACCEPT_RATE-0.5933}")
if [ "$CONC" -ge 40 ] && [ "${OFFLOAD:-1}" = 1 ]; then
  envs+=(PYTHONHASHSEED=0 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=256 LMCACHE_CHUNK_SIZE=256
    ATOM_PREFIX_CACHE_POLICY=slru ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5
    LMCACHE_CACHE_POLICY=ATOM_SLRU LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3)
  args+=(--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}')
fi
# shellcheck disable=SC2206
[ -n "${EXTRA_ENV:-}" ] && envs+=(${EXTRA_ENV})

# 第一行记下实际生效的环境（分析时据此判断 indexer CP、是否录 shape）
echo "ARCH=$ARCH CONC=$CONC KV=$KV SHAPES=${SHAPES:-0} ACCEPT_RATE=${ACCEPT_RATE-0.5933} env: ${envs[*]}" > "$LOG"
nohup env "${envs[@]}" python3 -u -m atom.entrypoints.openai_server \
  --model "$MODEL" --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$ENGINE_PORT" --server-port "$SERVER_PORT" \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --kv_cache_dtype "$KV" \
  --gpu-memory-utilization "${GPU_MEM:-0.95}" \
  --block-size 128 \
  --max-num-batched-tokens 32768 \
  --attn-prefill-chunk-size 16384 \
  --max-num-seqs "${MAX_SEQS:-$((2 * CONC))}" \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
  --default-chat-template-kwargs '{"thinking_mode": "enabled"}' \
  --enable-prefix-caching \
  --method eagle3 \
  --draft-model "$DRAFT" \
  --num-speculative-tokens 3 \
  --cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,24,26,28,30,32,34,36,40,48,56,64]' \
  "${args[@]}" >> "$LOG" 2>&1 &
echo "server pid $! log $LOG"
