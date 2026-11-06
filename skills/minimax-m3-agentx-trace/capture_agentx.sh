#!/bin/bash
# MiniMax-M3 AgentX 运行中采集（宿主机上运行）。照 trace-kernel-breakdown 的做法：一次正常的 AgentX 回放里，
# aiperf profiling 阶段开始后 START_OFFSET 秒（默认 600，取 600..900 之间）开一个 WINDOW 秒（默认 25，取 20..30）
# 的 torch-profiler 窗口，ATOM_PROFILER_MORE=0（不录 shape / stack）；prefill 和 decode 都在这一个窗口里。
# SHAPES=1 时再起一次服务、另开一个短窗口只录 shape（GEMM 列用），计时仍以无 shape 的窗口为准。
#
#   bash capture_agentx.sh check   # 只做检查，不用 GPU：镜像、模型、端口、磁盘、谁在占 GPU
#   bash capture_agentx.sh run     # 等 GPU 空出来后采集；退出时停掉并删除自己的容器，GPU 随之释放
#
# 必填: IMAGE（服务镜像，内含 ATOM + aiter + aiperf + AgentX 数据集缓存）
#       MODEL_DIR（宿主机目录，内含 MiniMax-M3-MXFP4/ 和 MiniMax-M3-EAGLE3-GQA/，挂到容器 /models）
#       OUT（宿主机输出目录，挂到容器 /out）
# 可选: CONC=10  START_OFFSET=600  WINDOW=25  DURATION=1200（aiperf profiling 阶段秒数，须 > START_OFFSET+WINDOW）
#       SHAPES=0  SHAPES_OFFSET=300  SHAPES_WINDOW=10
#       NAME=m3trace_$USER  SERVER_PORT=8890  ENGINE_PORT=8896  GPUS=（ROCR_VISIBLE_DEVICES）
#       WAIT_GPUS=1（GPU 被别人占着时每 POLL 秒查一次，等到空闲；0 = 直接退出）POLL=120
#       KEEP=0（1 = 结束后保留容器）ANALYZE=1（采集完在宿主机上跑 analyze.sh，只用 CPU）
#       MIN_FREE_GB=30（OUT 所在盘的最低空闲）TICK=5（等 aiperf / 窗口时的轮询间隔，秒）
#       WORKLOAD=agentx|fixed  fixed = 不跑 aiperf，m3_fixed.py 把 decode 固定在 FIXED_BS（默认 CONC）个请求、每个
#       FIXED_CTX（192038）token，窗口里插一个 FIXED_PREFILL_CACHED（157312）+ FIXED_PREFILL_NEW（2763）的 prefill；
#       窗口前不开 profiler 量 FIXED_REF（5）秒做对照，窗口 = FIXED_PRE（2）秒 + prefill + FIXED_POST（3）秒。
#       服务仍按 CONC 的 recipe 配置启动（max-num-seqs、indexer CP），START_OFFSET / WINDOW / DURATION 不用
#       其余环境变量（ACCEPT_RATE、OFFLOAD、KV、GPU_MEM、EXTRA_ENV ...）原样传给 m3_server.sh
# 从不终止别人的进程或容器；判断 GPU 是否空闲只读 /sys/class/kfd/kfd/proc（不打开 GPU）。
# 判断是整机的：任何人的任何进程开着 KFD 都算占用。
set -u
MODE=${1:-}
case "$MODE" in check|run) ;; *) awk 'NR > 1 && /^#/ {print; next} NR > 1 {exit}' "$0"; exit 2;; esac
SKILL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${IMAGE:-}
MODEL_DIR=${MODEL_DIR:-}
OUT=${OUT:-}
CONC=${CONC:-10}
START_OFFSET=${START_OFFSET:-600}
WINDOW=${WINDOW:-25}
DURATION=${DURATION:-1200}
SHAPES=${SHAPES:-0}
SHAPES_OFFSET=${SHAPES_OFFSET:-300}
SHAPES_WINDOW=${SHAPES_WINDOW:-10}
NAME=${NAME:-m3trace_${USER:-user}}
SERVER_PORT=${SERVER_PORT:-8890}
ENGINE_PORT=${ENGINE_PORT:-8896}
WAIT_GPUS=${WAIT_GPUS:-1}
POLL=${POLL:-120}
KEEP=${KEEP:-0}
ANALYZE=${ANALYZE:-1}
PROFILER_TIMEOUT=${PROFILER_TIMEOUT:-900}
MIN_FREE_GB=${MIN_FREE_GB:-30}
TICK=${TICK:-5}
KFD=${KFD:-/sys/class/kfd/kfd}
PASS_ENV=(ACCEPT_RATE OFFLOAD KV GPU_MEM MAX_SEQS QR FLY ROCTRACER_SHIFT EXTRA_ENV GPUS)
for v in CONC START_OFFSET WINDOW DURATION SHAPES_OFFSET SHAPES_WINDOW SERVER_PORT ENGINE_PORT POLL PROFILER_TIMEOUT MIN_FREE_GB; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || { echo "$v=${!v} is not a non-negative integer"; exit 2; }
done
WORKLOAD=${WORKLOAD:-agentx}
FIXED_BS=${FIXED_BS:-$CONC}
FIXED_CTX=${FIXED_CTX:-192038}
FIXED_PREFILL_CACHED=${FIXED_PREFILL_CACHED:-157312}
FIXED_PREFILL_NEW=${FIXED_PREFILL_NEW:-2763}
FIXED_REF=${FIXED_REF:-5}
FIXED_PRE=${FIXED_PRE:-2}
FIXED_POST=${FIXED_POST:-3}
case "$WORKLOAD" in agentx|fixed) ;; *) echo "WORKLOAD=$WORKLOAD: use agentx or fixed"; exit 2;; esac
for v in FIXED_BS FIXED_CTX FIXED_PREFILL_CACHED FIXED_PREFILL_NEW FIXED_REF FIXED_PRE FIXED_POST; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || { echo "$v=${!v} is not a non-negative integer"; exit 2; }
done

log() { echo "$(date '+%F %T') $*" | { [ -n "$OUT" ] && [ -d "$OUT" ] && tee -a "$OUT/capture.log" || cat; }; }
fail=0
bad() { log "FAIL  $*"; fail=1; }
ok() { log "ok    $*"; }

# ---- 谁在占 GPU：$KFD/proc 下每个目录是一个打开了 KFD 的进程
our_cid() { docker inspect -f '{{.Id}}' "$NAME" 2>/dev/null; }
gpu_holders() {  # 输出 "pid 容器名或命令"，不含我们自己的容器
  local cid p c
  cid=$(our_cid)
  for p in $(ls "$KFD/proc" 2>/dev/null); do
    c=$(grep -oE 'docker-[0-9a-f]{64}' "/proc/$p/cgroup" 2>/dev/null | head -1 | cut -c8-)
    [ -n "$cid" ] && [ "$c" = "$cid" ] && continue
    if [ -n "$c" ]; then
      echo "$p container:$(docker inspect -f '{{.Name}}' "$c" 2>/dev/null | tr -d /)"
    else
      echo "$p $(ps -o user=,comm= -p "$p" 2>/dev/null | tr -s ' ')"
    fi
  done
}
holders_summary() { gpu_holders | awk '{print $2}' | sort | uniq -c | awk '{printf "%s(%d procs) ", $2, $1}'; }

preflight() {
  if [ "$WORKLOAD" = fixed ]; then
    log "preflight: mode=$MODE WORKLOAD=fixed CONC=$CONC (server config) FIXED_BS=$FIXED_BS FIXED_CTX=$FIXED_CTX" \
        "prefill $FIXED_PREFILL_CACHED+$FIXED_PREFILL_NEW, window ${FIXED_PRE}s+prefill+${FIXED_POST}s, reference ${FIXED_REF}s NAME=$NAME"
  else
    log "preflight: mode=$MODE CONC=$CONC START_OFFSET=$START_OFFSET WINDOW=$WINDOW DURATION=$DURATION SHAPES=$SHAPES NAME=$NAME"
  fi
  command -v docker >/dev/null && docker info >/dev/null 2>&1 && ok "docker usable" || bad "docker not usable by $(id -un)"
  [ -n "$IMAGE" ] || bad "IMAGE not set (the serving image: ATOM + aiter + aiperf + the AgentX dataset cache)"
  if [ -n "$IMAGE" ]; then
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
      ok "image $IMAGE $(docker image inspect -f '{{.Id}}' "$IMAGE" | cut -c1-19) created $(docker image inspect -f '{{.Created}}' "$IMAGE" | cut -c1-19)"
      # 不挂任何设备的临时容器：只看镜像内容，不碰 GPU
      local r
      r=$(docker run --rm --network none --entrypoint bash "$IMAGE" -c '
        A=/app/ATOM; [ -d $A ] || A=$(python3 -c "import importlib.util as u,os; s=u.find_spec(\"atom\"); print(os.path.dirname(os.path.dirname(s.origin)) if s else \"\")" 2>/dev/null)
        echo "atom_dir=${A:-MISSING}"
        echo "atom_rev=$(git -C "$A" rev-parse --short HEAD 2>/dev/null || echo unknown)"
        python3 -c "import importlib.util as u; print(\"aiter=\" + (\"ok\" if u.find_spec(\"aiter\") else \"MISSING\"))"
        command -v aiperf >/dev/null && echo "aiperf=ok" || echo "aiperf=MISSING"
        ls -d /root/.cache/huggingface/datasets/*semianalysis* /root/.cache/huggingface/hub/datasets--*semianalysis* 2>/dev/null | head -1 | sed "s/^/dataset=/"
        command -v curl >/dev/null && echo "curl=ok" || echo "curl=MISSING"
        command -v rocminfo >/dev/null && echo "rocminfo=ok" || echo "rocminfo=MISSING"
        grep -q ATOM_ENABLE_DETAILED_ANNOTATION "$A"/atom/utils/envs.py 2>/dev/null && echo "detailed_annotation=ok" || echo "detailed_annotation=MISSING"
        grep -q ATOM_PROFILER_RECORD_SHAPES "$A"/atom/utils/envs.py 2>/dev/null && echo "profiler_options=ok" || echo "profiler_options=MISSING"' 2>&1)
      echo "$r" | sed 's/^/        /' | { [ -n "$OUT" ] && [ -d "$OUT" ] && tee -a "$OUT/capture.log" || cat; }
      echo "$r" | grep -q "atom_dir=MISSING" && bad "ATOM not found in the image"
      echo "$r" | grep -q "aiter=MISSING" && bad "aiter not importable in the image"
      if [ "$WORKLOAD" = agentx ]; then
        echo "$r" | grep -q "aiperf=MISSING" && bad "aiperf not in the image"
        echo "$r" | grep -q "^dataset=" || bad "AgentX dataset cache (semianalysis_cc_traces_weka_062126) not found in the image's HF cache; aiperf would download it"
      fi
      echo "$r" | grep -qE "detailed_annotation=MISSING|profiler_options=MISSING" && bad "this ATOM lacks ATOM_ENABLE_DETAILED_ANNOTATION / per-option profiler flags (annotations would carry no bs/tok/ctx)"
    else
      bad "image $IMAGE not present locally (docker pull it first)"
    fi
  fi
  if [ -z "$MODEL_DIR" ]; then
    bad "MODEL_DIR not set"
  else
    for m in MiniMax-M3-MXFP4 MiniMax-M3-EAGLE3-GQA; do
      [ -f "$MODEL_DIR/$m/config.json" ] && ok "model $MODEL_DIR/$m" || bad "missing $MODEL_DIR/$m/config.json"
    done
  fi
  if [ -z "$OUT" ]; then
    bad "OUT not set"
  else
    mkdir -p "$OUT" 2>/dev/null && [ -w "$OUT" ] && ok "OUT $OUT writable" || bad "OUT $OUT not writable"
    if [ -d "$OUT" ]; then
      local free_gb
      free_gb=$(df -BG --output=avail "$OUT" | tail -1 | tr -dc 0-9)
      [ "${free_gb:-0}" -ge "$MIN_FREE_GB" ] && ok "free disk ${free_gb} GB at OUT" \
        || bad "only ${free_gb} GB free at OUT (want >= $MIN_FREE_GB: 4 ranks of trace + reduced copies)"
      [ -e "$OUT/traces" ] && [ "$MODE" = run ] && bad "$OUT/traces exists: use a fresh OUT per capture"
    fi
  fi
  if [ "$WORKLOAD" = fixed ]; then
    [ $((FIXED_CTX % 128)) != 0 ] && ok "FIXED_CTX=$FIXED_CTX leaves a $((FIXED_CTX % 128))-token tail to prefill when the $FIXED_BS requests start together" \
      || bad "FIXED_CTX=$FIXED_CTX is a multiple of the 128-token block: the resend would have nothing to prefill"
    [ $((FIXED_PREFILL_CACHED % 128)) = 0 ] && ok "FIXED_PREFILL_CACHED=$FIXED_PREFILL_CACHED is whole blocks" \
      || bad "FIXED_PREFILL_CACHED=$FIXED_PREFILL_CACHED is not a multiple of 128: part of it would be recomputed"
    [ "$FIXED_BS" -ge 1 ] && [ "$FIXED_BS" -le "${MAX_SEQS:-$((2 * CONC))}" ] && ok "FIXED_BS=$FIXED_BS within max-num-seqs ${MAX_SEQS:-$((2 * CONC))}" \
      || bad "FIXED_BS=$FIXED_BS must be 1..max-num-seqs (${MAX_SEQS:-$((2 * CONC))})"
    [ "$SHAPES" = 1 ] && bad "SHAPES=1 is for WORKLOAD=agentx only"
    python3 -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" "$SKILL_DIR/m3_fixed.py" || bad "m3_fixed.py has a syntax error"
  else
    [ "$DURATION" -gt $((START_OFFSET + WINDOW + 60)) ] && ok "window at ${START_OFFSET}s+${WINDOW}s inside a ${DURATION}s profiling phase" \
      || bad "DURATION=$DURATION must exceed START_OFFSET+WINDOW+60"
    { [ "$START_OFFSET" -ge 600 ] && [ "$START_OFFSET" -le 900 ]; } || log "note  START_OFFSET=$START_OFFSET is outside 600..900 s (the method's steady-state range)"
    { [ "$WINDOW" -ge 20 ] && [ "$WINDOW" -le 30 ]; } || log "note  WINDOW=$WINDOW is outside 20..30 s"
  fi
  local p
  for p in "$SERVER_PORT" "$ENGINE_PORT"; do
    ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$p\$" && bad "port $p already in use on this host (set SERVER_PORT / ENGINE_PORT)" || ok "port $p free"
  done
  docker inspect "$NAME" >/dev/null 2>&1 && bad "container name $NAME already exists (set NAME, or remove your own old container)" || ok "container name $NAME free"
  local ngpu
  ngpu=$(grep -l 'gfx_target_version [1-9]' "$KFD"/topology/nodes/*/properties 2>/dev/null | wc -l)
  [ "$ngpu" -ge 4 ] && ok "$ngpu GPUs visible to KFD (gfx_target_version $(grep -h -m1 gfx_target_version "$KFD"/topology/nodes/*/properties | awk '$2>0{print $2}' | sort -u | tr '\n' ' '))" \
    || bad "TP4 needs 4 GPUs, KFD shows $ngpu"
  local h
  h=$(holders_summary)
  [ -z "$h" ] && ok "GPUs free right now" || log "busy  GPUs held by: $h"
  command -v python3 >/dev/null && ok "host python3 $(python3 -c 'import sys; print(sys.version.split()[0])') for analysis$(python3 -c 'import orjson' 2>/dev/null && echo ' (orjson)' || echo ' (no orjson: slower json load, same result)')" \
    || bad "no host python3 for analyze.sh"
  for f in m3_server.sh m3_aiperf.sh analyze.sh; do bash -n "$SKILL_DIR/$f" || bad "$f has a syntax error"; done
  return $fail
}

wait_free() {
  local first=1 h
  while :; do
    h=$(holders_summary)
    [ -z "$h" ] && break
    [ "$WAIT_GPUS" = 1 ] || { log "GPUs busy ($h) and WAIT_GPUS=0: not starting"; exit 3; }
    [ $first = 1 ] && log "waiting for GPUs, held by: $h (checking every ${POLL}s; nothing of theirs is touched)"
    first=0
    sleep "$POLL"
  done
  [ $first = 0 ] && log "GPUs free"
  return 0
}

dexec() { docker exec "$@"; }
health() { curl -s -m 5 "localhost:$SERVER_PORT/health" -o /dev/null -w '%{http_code}'; }
server_alive() { dexec "$NAME" pgrep -f '[a]tom.entrypoints.openai_server' >/dev/null; }

stop_inside() {  # 只停我们自己容器里的服务和 aiperf；aiperf 先 SIGINT，给它时间导出已完成的记录
  dexec "$NAME" bash -c "pkill -INT -f '[a]iperf profile'
    for i in \$(seq 1 60); do pgrep -f '[a]iperf profile' >/dev/null || break; sleep 1; done
    pkill -f '[a]iperf profile'; pkill -f '[m]3_fixed.py'; pkill -f '[a]tom.entrypoints.openai_server'" 2>/dev/null
  local i n
  for i in $(seq 1 60); do
    n=$(dexec "$NAME" bash -c "ps -eo args | grep -cE '[o]penai_server|[a]iperf profile|[A]TOM::'" 2>/dev/null)
    [ "${n:-0}" = 0 ] && break
    sleep "$TICK"
  done
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if docker inspect "$NAME" >/dev/null 2>&1; then
    log "cleanup: stopping server and aiperf in $NAME"
    stop_inside
    dexec "$NAME" chown -R "$(id -u):$(id -g)" /out 2>/dev/null
    if [ "$KEEP" = 1 ]; then
      log "cleanup: KEEP=1, container $NAME kept (processes stopped)"
    else
      docker rm -f "$NAME" >/dev/null 2>&1 && log "cleanup: container $NAME removed"
    fi
  fi
  local left
  left=$(ls "$KFD/proc" 2>/dev/null | wc -l)
  log "exit rc=$rc; KFD processes on this host now: $left ($(holders_summary))"
  exit $rc
}

start_server() {  # $1 shapes(0/1) $2 profile dir (container) $3 server log (container)
  local envargs=(-e CONC="$CONC" -e SHAPES="$1" -e PROFILE_DIR="$2" -e SERVER_PORT="$SERVER_PORT" -e ENGINE_PORT="$ENGINE_PORT"
                 -e PROFILER_TIMEOUT="$PROFILER_TIMEOUT" -e MODEL=/models/MiniMax-M3-MXFP4 -e DRAFT=/models/MiniMax-M3-EAGLE3-GQA)
  local v
  for v in "${PASS_ENV[@]}"; do [ -n "${!v+x}" ] && envargs+=(-e "$v=${!v}"); done
  dexec "${envargs[@]}" "$NAME" bash /skill/m3_server.sh "$3" | tee -a "$OUT/capture.log"
  log "server starting (shapes=$1), log ${3/#\/out/$OUT}"
  local i
  for i in $(seq 1 360); do
    [ "$(health)" = 200 ] && break
    if ! server_alive; then
      log "ABORT server died: $(grep -m3 -E 'exitcode=-?[0-9]+|[A-Za-z]+Error: .{0,200}' "${3/#\/out/$OUT}" | tr '\n' ' ')"
      return 1
    fi
    sleep 10
  done
  [ "$(health)" = 200 ] || { log "ABORT server not healthy after 1 h"; return 1; }
  local ans
  ans=$(curl -s -m 600 "localhost:$SERVER_PORT/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"model":"/models/MiniMax-M3-MXFP4","messages":[{"role":"user","content":"What is 17 * 23? Answer briefly."}],"max_tokens":1024,"temperature":0}' \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'][-80:])" 2>&1 | tr '\n' ' ')
  log "server healthy; smoke: $ans"
}

window() {  # $1 artifact dir (container) $2 duration $3 offset $4 window seconds $5 tag
  local art=$1 dur=$2 off=$3 win=$4 tag=$5 host_log="$OUT/aiperf_$5.log"
  dexec -e CONC="$CONC" -e DURATION="$dur" -e SERVER_PORT="$SERVER_PORT" -e MODEL=/models/MiniMax-M3-MXFP4 \
    "$NAME" bash /skill/m3_aiperf.sh "$art" > "$host_log" 2>&1 &
  local apid=$! i t_phase=""
  log "$tag: aiperf started (profiling phase ${dur}s), log $host_log"
  for i in $(seq 1 720); do
    grep -q "Phase profiling (profiling) started" "$host_log" 2>/dev/null && { t_phase=$(date +%s.%N); break; }
    kill -0 $apid 2>/dev/null || { log "ABORT $tag: aiperf exited before its profiling phase: $(tail -n 3 "$host_log" | tr '\n' ' ')"; return 1; }
    sleep "$TICK"
  done
  [ -n "$t_phase" ] || { log "ABORT $tag: no profiling phase after $(python3 -c "print(round(720 * $TICK / 60))") min of warmup"; return 1; }
  log "$tag: aiperf profiling phase started; window opens in ${off}s"
  local until=$(python3 -c "print($t_phase + $off)")
  while python3 -c "import sys,time; sys.exit(0 if time.time() < $until else 1)"; do
    kill -0 $apid 2>/dev/null || { log "ABORT $tag: aiperf exited before the window"; return 1; }
    server_alive || { log "ABORT $tag: server died before the window"; return 1; }
    sleep "$TICK"
  done
  local t0 t1 t2 t3 c1 c2
  t0=$(date +%s.%N)
  c1=$(curl -s -m 120 -X POST "localhost:$SERVER_PORT/start_profile" -o "$OUT/start_profile_$tag.json" -w '%{http_code}')
  t1=$(date +%s.%N)
  log "$tag: start_profile -> http $c1 $(head -c 200 "$OUT/start_profile_$tag.json" 2>/dev/null)"
  [ "$c1" = 200 ] || { log "ABORT $tag: start_profile failed"; return 1; }
  sleep "$win"
  t2=$(date +%s.%N)
  c2=$(curl -s -m $((PROFILER_TIMEOUT + 120)) -X POST "localhost:$SERVER_PORT/stop_profile" -o "$OUT/stop_profile_$tag.json" -w '%{http_code}')
  t3=$(date +%s.%N)
  log "$tag: stop_profile -> http $c2 after $(python3 -c "print(round($t3 - $t2, 1))") s (export; engine paused) $(head -c 300 "$OUT/stop_profile_$tag.json" 2>/dev/null)"
  [ "$c2" = 200 ] || log "WARN $tag: stop_profile did not return 200; check the server log for 'trace exported to'"
  python3 - "$OUT/capture_meta.json" "$tag" "$t_phase" "$t0" "$t1" "$t2" "$t3" "$off" "$win" "$dur" <<'EOF'
import json, os, sys
p, tag, *v = sys.argv[1:]
m = json.load(open(p)) if os.path.exists(p) else {}
keys = ["phase_start_epoch", "start_profile_sent", "start_profile_returned", "stop_profile_sent", "stop_profile_returned",
        "offset_s", "window_s", "aiperf_duration_s"]
m.setdefault("windows", {})[tag] = dict(zip(keys, map(float, v)))
json.dump(m, open(p, "w"), indent=1)
EOF
  log "$tag: window recorded; waiting for aiperf to finish its ${dur}s phase"
  wait $apid
  log "$tag: aiperf rc=$? ; $(grep -m1 -E 'Benchmark Duration' "$host_log")"
}

fixed_window() {  # WORKLOAD=fixed：m3_fixed.py 在容器里预热、起 FIXED_BS 路 decode、量对照、开窗插 prefill、关窗
  local host_log="$OUT/fixed_client.log" rc
  log "main: fixed-shape client started (bs=$FIXED_BS x $FIXED_CTX tokens), log $host_log"
  # 上限只防挂死：预热是 FIXED_BS 个长 prompt 的整段 prefill
  timeout $((FIXED_BS * 300 + PROFILER_TIMEOUT + 1800)) docker exec "$NAME" python3 -u /skill/m3_fixed.py \
    --url "http://localhost:$SERVER_PORT" --model /models/MiniMax-M3-MXFP4 --bs "$FIXED_BS" --ctx "$FIXED_CTX" \
    --prefill-cached "$FIXED_PREFILL_CACHED" --prefill-new "$FIXED_PREFILL_NEW" \
    --ref-s "$FIXED_REF" --pre-s "$FIXED_PRE" --post-s "$FIXED_POST" \
    --server-log /out/server.log --summary /out/fixed_summary.json > "$host_log" 2>&1
  rc=$?
  log "main: fixed-shape client rc=$rc: $(grep -E 'reference|window \(|prefill shape|ABORT' "$host_log" | cut -c21- | tr '\n' ' ')"
  { [ $rc = 0 ] && [ -f "$OUT/fixed_summary.json" ]; } || { log "ABORT main: fixed-shape client failed"; return 1; }
  python3 - "$OUT/capture_meta.json" "$OUT/fixed_summary.json" <<'EOF'
import json, sys
p, fs = sys.argv[1:]
m, s = json.load(open(p)), json.load(open(fs))
w = s["window"]
m.setdefault("windows", {})["main"] = {k: w[k] for k in ("start_profile_sent", "start_profile_returned",
                                                         "stop_profile_sent", "stop_profile_returned")}
m["windows"]["main"]["window_s"] = w["stop_profile_sent"] - w["start_profile_returned"]
m["fixed"] = {"params": s["params"], "prefill_shape_ok": s["prefill_shape_ok"], "stream_errors": len(s["stream_errors"])}
json.dump(m, open(p, "w"), indent=1)
EOF
  python3 -c "import json, sys; ok = json.load(open(sys.argv[1]))['prefill_shape_ok']; sys.exit(0 if all(ok.values()) else 1)" \
    "$OUT/fixed_summary.json" || log "WARN main: the prefill shape was not found in the server log; check fixed_summary.json 'scheduled'"
  log "main: window recorded"
}

# ------------------------------------------------------------------ main
preflight
if [ "$MODE" = check ]; then
  [ $fail = 0 ] && log "preflight passed: 'bash $0 run' (same env) captures once the GPUs are free" || log "preflight FAILED"
  exit $fail
fi
[ $fail = 0 ] || { log "preflight failed, not starting"; exit 1; }
trap cleanup EXIT INT TERM
wait_free
docker run -d --name "$NAME" --network host --ipc host --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --device /dev/kfd --device /dev/dri --shm-size 64g \
  -v "$MODEL_DIR":/models:ro -v "$OUT":/out -v "$SKILL_DIR":/skill:ro \
  --entrypoint sleep "$IMAGE" infinity >/dev/null || { log "docker run failed"; exit 1; }
log "container $NAME up"
cp "$MODEL_DIR/MiniMax-M3-MXFP4/config.json" "$OUT/model_config.json"
python3 - "$OUT/capture_meta.json" <<EOF
import json, subprocess, sys
lab = json.loads(subprocess.run(["docker", "image", "inspect", "-f", "{{json .Config.Labels}}", "$IMAGE"],
                                capture_output=True, text=True).stdout or "null") or {}
iid = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", "$IMAGE"], capture_output=True, text=True).stdout.strip()
method = {"agentx": "in-run AgentX window, ATOM_PROFILER_MORE=0",
          "fixed": "fixed-shape replay (bs=$FIXED_BS x $FIXED_CTX tokens decode, one $FIXED_PREFILL_CACHED+$FIXED_PREFILL_NEW "
                   "prefill in the window), ATOM_PROFILER_MORE=0"}["$WORKLOAD"]
json.dump({"image": "$IMAGE", "image_id": iid, "image_labels": lab, "conc": $CONC, "tp": 4, "workload": "$WORKLOAD",
           "method": method, "windows": {}}, open(sys.argv[1], "w"), indent=1)
EOF
wait_free   # 建容器期间别人可能刚开始用卡
start_server 0 /out/traces /out/server.log || exit 1
if [ "$WORKLOAD" = fixed ]; then
  fixed_window || exit 1
else
  window /out/aiperf "$DURATION" "$START_OFFSET" "$WINDOW" main || exit 1
fi
if [ "$SHAPES" = 1 ]; then
  stop_inside
  wait_free
  start_server 1 /out/traces_shapes /out/server_shapes.log || exit 1
  window /out/aiperf_shapes $((SHAPES_OFFSET + SHAPES_WINDOW + 120)) "$SHAPES_OFFSET" "$SHAPES_WINDOW" shapes || exit 1
fi
stop_inside
dexec "$NAME" chown -R "$(id -u):$(id -g)" /out 2>/dev/null
log "capture done: traces under $OUT/traces/rank_*/"
if [ "$ANALYZE" = 1 ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 && log "container $NAME removed before the CPU-only analysis"
  bash "$SKILL_DIR/analyze.sh" "$OUT" 2>&1 | tee -a "$OUT/capture.log"
fi
