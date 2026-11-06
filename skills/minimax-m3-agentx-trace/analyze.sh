#!/bin/bash
# capture_agentx.sh 的输出目录 -> 全部分析结果（只用 CPU，可以在任何有 python3 的机器上跑）。
# 用法: bash analyze.sh <capture OUT dir> [rank A=0] [rank B=1]
# 环境变量: PY（解释器或项目的命令包装，默认 python3）CFG（默认 OUT/model_config.json）
#           MIN_TOK / MIN_BUSY（m3_prefill.py 的 steady 筛选，默认 0 / 0：AgentX 的 chunk 可能一半时间空闲）
#           TITLE（宽表第一行；默认由 capture_meta.json 拼出）PEAKS（peaks json，给推导出的 TFLOP/s、TB/s 做上限检查）
# 步骤: 每个 rank 精简 trace（extract_trace.py）-> 完整性检查（两个 rank）-> prefill / decode breakdown（rank A）
#       -> 窗口总账和 ITL 对账 -> 宽表 -> rank A vs B 的通信 skew；有 shapes 窗口时再用录到的 shape 核对 GEMM 的 K/N。
set -u
OUT=${1:?capture dir}
RA=${2:-0}
RB=${3:-1}
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-python3}
CFG=${CFG:-$OUT/model_config.json}
B=$OUT/breakdown
mkdir -p "$B"
[ -f "$CFG" ] || { echo "no model config at $CFG (set CFG=<MiniMax-M3-MXFP4/config.json>)"; exit 1; }
PEAKS_ARG=()
[ -n "${PEAKS:-}" ] && PEAKS_ARG=(--peaks "$PEAKS")

raw_of() { ls -t "$1"/rank_"$2"/*.pt.trace.json.gz 2>/dev/null | head -1; }
reduce() {  # $1 traces dir  $2 rank  -> prints the reduced file
  local f d
  f=$(raw_of "$1" "$2")
  [ -n "$f" ] || { echo "no trace under $1/rank_$2" >&2; return 1; }
  gzip -t "$f" || { echo "corrupt gzip: $f (copied through a shell variable? copy with a plain redirect and compare md5)" >&2; return 1; }
  d=$OUT/reduced/$(basename "$1")/rank_$2
  mkdir -p "$d"
  if [ ! -s "$d/reduced.pt.trace.json.gz" ] || [ "$f" -nt "$d/reduced.pt.trace.json.gz" ]; then
    $PY "$S/extract_trace.py" "$f" "$d" > "$d/extract.log" 2>&1 || { echo "extract_trace failed: $d/extract.log" >&2; return 1; }
  fi
  echo "$d/reduced.pt.trace.json.gz"
}

CP=()
head -1 "$OUT/server.log" 2>/dev/null | grep -q "ATOM_M3_INDEXER_CP=1" && CP=(--indexer-cp)
RED_A=$(reduce "$OUT/traces" "$RA") || exit 1
RED_B=$(reduce "$OUT/traces" "$RB") || RED_B=""
RAW_A=$(raw_of "$OUT/traces" "$RA")
echo "rank $RA: $RAW_A -> $RED_A"
[ -n "$RED_B" ] && echo "rank $RB: $(raw_of "$OUT/traces" "$RB") -> $RED_B"

{
  $PY "$S/m3_trace_completeness.py" "$RED_A" "$CFG" "rank $RA"
  [ -n "$RED_B" ] && $PY "$S/m3_trace_completeness.py" "$RED_B" "$CFG" "rank $RB"
} > "$B/completeness.txt" 2>&1
$PY "$S/m3_prefill.py" "$RED_A" "$CFG" "$B" --min-tok "${MIN_TOK:-0}" --min-busy "${MIN_BUSY:-0}" "${CP[@]}" "${PEAKS_ARG[@]}" \
  > "$B/prefill_console.txt" 2>&1
PRC=$?
$PY "$S/m3_decode.py" "$RED_A" "$CFG" "$B" "${CP[@]}" "${PEAKS_ARG[@]}" > "$B/decode_console.txt" 2>&1
DRC=$?
WIN_ARGS=(--trace-name "$(basename "$RAW_A")" --out "$B/window_summary.json")
[ -f "$OUT/server.log" ] && WIN_ARGS+=(--server-log "$OUT/server.log")
[ -d "$OUT/aiperf" ] && WIN_ARGS+=(--aiperf-dir "$OUT/aiperf")
[ -f "$OUT/fixed_summary.json" ] && WIN_ARGS+=(--fixed-summary "$OUT/fixed_summary.json")
$PY "$S/m3_window.py" "$RED_A" "${WIN_ARGS[@]}" > "$B/window.txt" 2>&1

if [ -z "${TITLE:-}" ]; then
  TITLE=$($PY - "$OUT/capture_meta.json" "$RA" "$OUT/capture.log" <<'EOF'
import json, re, sys
try:
    m = json.load(open(sys.argv[1]))
except Exception:
    m = {}
w = (m.get("windows") or {}).get("main") or {}
lab = m.get("image_labels") or {}
# 镜像标签可能是从基础镜像继承下来的旧值：优先用 check 在镜像里读到的 git rev
try:
    rev = re.findall(r"atom_rev=([0-9a-f]{7,})", open(sys.argv[3], errors="replace").read())
except OSError:
    rev = []
atom = rev[-1][:9] if rev else (lab.get("com.amd.atom.commit") or "")[:9]
img = f"image {m.get('image', '?')}{' (ATOM ' + atom + ')' if atom else ''}"
if m.get("workload") == "fixed":
    p = (m.get("fixed") or {}).get("params") or {}
    print(f"MiniMax-M3-MXFP4 TP4 EAGLE3 fixed-shape decode bs={p.get('bs', '?')} x {p.get('ctx', '?')} tokens + one "
          f"{p.get('prefill_cached', '?')}+{p.get('prefill_new', '?')} prefill, server config C{m.get('conc', '?')}, "
          f"rank {sys.argv[2]}, {img}, {w.get('window_s', 0):.0f} s window, ATOM_PROFILER_MORE=0")
else:
    print(f"MiniMax-M3-MXFP4 TP4 EAGLE3 AgentX C{m.get('conc', '?')}, rank {sys.argv[2]}, {img}, "
          f"{w.get('window_s', 0):.0f} s window at {w.get('offset_s') or 0:.0f} s into the aiperf profiling phase, "
          f"ATOM_PROFILER_MORE=0")
EOF
)
fi
$PY "$S/m3_combine_wide.py" "$B" "$B/wide.csv" "$TITLE" > "$B/combine.txt" 2>&1

if [ -n "$RED_B" ]; then
  {
    if [ -f "$B/prefill_picks.txt" ]; then
      ann=$(head -1 "$B/prefill_picks.txt" | sed 's/^prefill annotation: //')
      $PY "$S/m3_rank_skew.py" "$RED_A" "$RED_B" --kind prefill --annotation "$ann"
    fi
    echo
    $PY "$S/m3_rank_skew.py" "$RED_A" "$RED_B" --kind decode
  } > "$B/rank_skew.txt" 2>&1
fi

if [ -d "$OUT/traces_shapes" ]; then
  BS=$OUT/breakdown_shapes
  mkdir -p "$BS"
  if RED_S=$(reduce "$OUT/traces_shapes" "$RA"); then
    $PY "$S/m3_prefill.py" "$RED_S" "$CFG" "$BS" "${CP[@]}" "${PEAKS_ARG[@]}" > "$BS/prefill_console.txt" 2>&1
    $PY "$S/m3_decode.py" "$RED_S" "$CFG" "$BS" "${CP[@]}" "${PEAKS_ARG[@]}" > "$BS/decode_console.txt" 2>&1
  fi
fi

# 汇总：每一步的关键行（全文在各自的 txt 里）
{
  echo "== completeness"; grep -E "^== |RESULT" "$B/completeness.txt"
  echo "== prefill (exit $PRC)"; grep -E "representative prefill|wall=|median layer|reconcile|median-layer model|unclassified|side streams|^ *sum " "$B/prefill_console.txt"
  echo "== decode (exit $DRC)"; grep -E "chosen key|segmented steps|same-name|representative layer|reconcile:" "$B/decode_console.txt"
  echo "== window"; grep -E "== window|step period|window /|tokens per|predicted ITL|aiperf ITL|window starts|stop_profile took|fixed-shape|client step|client sees|profiler overhead" "$B/window.txt"
  [ -f "$B/rank_skew.txt" ] && { echo "== rank skew"; cat "$B/rank_skew.txt"; }
  [ -d "${BS:-/nonexistent}" ] && { echo "== shapes window"; grep -E "shapes recorded|K,N check" "$BS/prefill_console.txt" "$BS/decode_console.txt"; }
} > "$B/SUMMARY.txt"
cat "$B/SUMMARY.txt"
echo "outputs in $B (wide.csv, *_detail.csv, *_module_summary.csv, *_comm.csv, *_extras.csv, *_picks.txt, *_console.txt)"
