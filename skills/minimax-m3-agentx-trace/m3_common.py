#!/usr/bin/env python3
"""MiniMax-M3 ATOM trace 的公共部分，供各个 m3_*.py 共用。

- 读 trace（原始 .pt.trace.json.gz，或 extract_trace.py 精简后的 reduced 文件）；
- 主 stream = GPU 事件最多的 tid；Memcpy HtoD 一律丢掉：它挂在主 stream 上，实际跑在 DMA 引擎上、
  和计算并行，算进去 busy 会超过 100%；
- 切层：每层恰好一个 attention core；层边界在上一层 MLP/MoE 输出的那次 all-reduce 之后；
- kernel -> 模块：角色由它相对 core 和本层两次 all-reduce 的位置决定，名字只用来认 kernel 类型
  （通用 GEMM 名字不带角色，按名字硬映射会让后面的 GEMM 整体错位）；
- 从 config 推导 TP 下的 GEMM K/N（trace 没录 shape 时用，输出里标 derived）；
- Perfetto UI 时间（以 profiler 的 'Trace' 事件为零点）。

prefill 和 decode 共用这一份映射，两个阶段的模块集合因此一致；某个模块只在一边出现，就是映射缺口。
"""
import bisect
import collections
import gzip
import json
import re
import statistics

try:
    import orjson
    _loads = orjson.loads
except ImportError:  # 没有 orjson 时用标准库：慢一些，结果一样
    _loads = json.loads

GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")

# ---------------------------------------------------------------- kernel 类型（只认类型，不认角色）
CORE = re.compile(r"fmha_bf16_pertoken|fmha_fwd|pa_decode_tile_kernel|paged_attention_decode|pa_bf16_pertokenFp8"
                  r"|^kernel_0$")
SPARSE_CORE = re.compile(r"pa_decode_tile_kernel\w*_bs16_")
INDEXER = re.compile(r"index_block_score|index_score|topk_index")
COMM = re.compile(r"ar_gfx1250|ar_ll_gfx1250|ar_rmsnorm|ag_gfx1250|start_sync_kernel|end_sync_kernel|ncclDevKernel"
                  r"|ddaAllReduce|ddaAllGather|cross_device_reduce|allreduce|all_reduce|allgather|all_gather"
                  r"|reduce_scatter|quick_reduce|mscclKernel")
EXPERT = re.compile(r"a8w4_tdm|fused_moe|fmoe|ck_moe|moe_gemm|mfma_moe|moe_stage|gemm1_a4w4|gemm2_a4w4")
SPLITK = re.compile(r"splitk_reduce|split_k_reduce")
GEMM = re.compile(r"gemm|Cijk_|hipblaslt", re.I)
COPY = re.compile(r"^Memcpy DtoD|__amd_rocclr_copyBuffer|fused_clone")
NORM = re.compile(r"rmsnorm|rms_norm|layernorm|layer_norm")

GEMM_MODULES = {"qkv_proj", "qkv_idx_proj", "o_proj", "dense_gate_up", "dense_down", "moe_router", "lm_head",
                "gemm_unmapped"}
EXPERT_MODULES = {"moe_expert_gate_up", "moe_expert_down"}

# 模块 -> 汇总组
GROUPS = [("comm", lambda m: m.startswith("comm_")),
          ("attention_core", lambda m: m.endswith(("_attn_core", "_attn_core_reduce"))),
          ("indexer", lambda m: m.startswith(("indexer_", "sparse_attn_plan"))),
          ("attention_proj", lambda m: m.startswith(("qkv_", "o_proj", "attn_"))),
          ("moe_routed_shared_experts", lambda m: m.startswith(("moe_router", "moe_topk", "moe_route", "moe_expert",
                                                                "moe_combine"))),
          ("dense_mlp", lambda m: m.startswith("dense_")),
          ("norm", lambda m: m in ("input_norm", "post_attn_norm", "final_norm")),
          ("embedding_lm_head", lambda m: m in ("embedding", "lm_head", "lm_head_quant", "logits_row_select")),
          ("eagle3_aux", lambda m: m.startswith("aux_hidden")),
          ("other", lambda m: True)]


def group(m):
    return next(g for g, f in GROUPS if f(m))


# ---------------------------------------------------------------- 读 trace
def load_trace(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        return _loads(f.read())


def short(n):
    """可读的 kernel 名。PyTorch 的通用 launcher（elementwise_kernel_manual_unroll、vectorized_elementwise_kernel ...）
    只在模板参数里写真正做的事，这部分放进方括号：'at::native::elementwise_kernel_manual_unroll [direct_copy float]'。
    （思路来自 trace-kernel-breakdown 的 short()，见 NOTICE。）"""
    n = n.replace("(anonymous namespace)::", "")
    n = re.sub(r"^void\s+", "", n)
    base = re.sub(r"[<(].*", "", n).strip() or "(unnamed)"
    rest = n[len(base):]
    tag = ""
    if base.startswith(("at::native::", "elementwise_kernel")) and rest:
        m = (re.search(r"at::native::(?:binary_internal::)?(\w+?)(?:_kernel)?_cuda\b", rest)
             or re.search(r"at::native::(\w+?)_kernel_impl\b", rest)
             or re.search(r"at::native::(\w+?)_cuda_out\b", rest)
             or re.search(r"(Compare\w+Functor)\b", rest)
             or re.search(r"(CUDAFunctor\w*|\w*Functor)\b", rest))
        if m:
            tag = m.group(1)
            dt = re.search(r"\{lambda\((?:c10::)?([\w ]+?)\)#1\}", rest)
            if dt and tag in ("direct_copy",):
                tag += " " + dt.group(1)
            if tag.endswith("Functor") or "Functor_" in tag:
                inner = re.search(re.escape(tag) + r"<([^<>]*)>", rest)
                if inner:
                    tag += "<" + inner.group(1).replace("c10::", "")[:20] + ">"
    return (base + (f" [{tag}]" if tag else ""))[:110]


class Trace:
    """一个 rank 的 trace：主 stream、side stream、forward annotation、cpu_op（按 External id 取最内层）。"""

    def __init__(self, path, ui_zero_us=None):
        self.path = str(path)
        doc = load_trace(path)
        self.meta = {k: v for k, v in doc.items() if k != "traceEvents"}
        ev = doc["traceEvents"]
        self.ops = {}
        gpu = []
        self.annotations = []
        self.trace_events = []
        for e in ev:
            c = e.get("cat")
            if c in GPU_CATS:
                if "ts" in e and "dur" in e and not e.get("name", "").startswith("Memcpy HtoD"):
                    gpu.append(e)
            elif c == "cpu_op":
                x = (e.get("args") or {}).get("External id")
                if x is not None and (x not in self.ops or e.get("dur", 0) < self.ops[x].get("dur", 0)):
                    self.ops[x] = e  # 最内层的 op 带真实的 Input Dims
            elif c == "gpu_user_annotation":
                self.annotations.append(e)
            elif c == "Trace":
                self.trace_events.append(e)
        if not gpu:
            raise SystemExit(f"{path}: no GPU kernel events, the profiler recorded CPU activity only (some images need "
                             "ROCPROFILER_REGISTER_FORCE_LOAD=0 in the server env: EXTRA_ENV for capture_agentx.sh)")
        self.main_tid = collections.Counter(e["tid"] for e in gpu).most_common(1)[0][0]
        self.K = sorted((e for e in gpu if e["tid"] == self.main_tid), key=lambda e: e["ts"])
        self.KTS = [e["ts"] for e in self.K]
        self.SIDE = sorted((e for e in gpu if e["tid"] != self.main_tid), key=lambda e: e["ts"])
        self.STS = [e["ts"] for e in self.SIDE]
        self.main_ann = sorted((a for a in self.annotations if a.get("tid") == self.main_tid), key=lambda a: a["ts"])
        self.g0 = min(self.K[0]["ts"], self.SIDE[0]["ts"] if self.SIDE else self.K[0]["ts"])
        self.g1 = max(e["ts"] + e["dur"] for e in gpu)
        if ui_zero_us is not None:
            self.ui_zero, self.ui_src = float(ui_zero_us), "--ui-zero-us"
        elif self.trace_events:
            self.ui_zero, self.ui_src = min(e["ts"] for e in self.trace_events), "this file's 'Trace' event"
        else:
            self.ui_zero = min(e["ts"] for e in ev if "ts" in e and e.get("ph") != "M")
            self.ui_src = "this file's earliest event (the original trace's 'Trace' event is earlier: pass --ui-zero-us)"

    def forwards(self, kind):
        """主 stream 上的 prefill[...] / decode[...] annotation，时间序。"""
        return [a for a in self.main_ann if a.get("name", "").startswith(kind + "[")]

    def main_window(self, a):
        return self.K[bisect.bisect_left(self.KTS, a["ts"]): bisect.bisect_right(self.KTS, a["ts"] + a["dur"])]

    def side_window(self, lo, hi):
        return self.SIDE[bisect.bisect_left(self.STS, lo): bisect.bisect_left(self.STS, hi)]

    def op_of(self, k):
        return self.ops.get((k.get("args") or {}).get("External id"))

    def ui(self, ts):
        """trace ts（us）的 Perfetto UI 时间 HH:MM:SS.nnnnnnnnn（以 profiler 'Trace' 事件为零点）。"""
        t = max(0.0, ts - self.ui_zero) / 1e6
        return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:012.9f}"

    def epoch_ns(self, ts):
        """trace ts（us）对应的 epoch 纳秒；需要 trace 顶层的 baseTimeNanoseconds（kineto 写入）。"""
        base = self.meta.get("baseTimeNanoseconds")
        return None if base is None else int(base + ts * 1000)

    def has_shapes(self):
        return any((o.get("args") or {}).get("Input Dims") for o in self.ops.values())


# ---------------------------------------------------------------- config
class M3Config:
    """层类型从 config 读，绝不从 trace 推：sparse_attention_freq（0 = 全注意力，1 = indexer 稀疏注意力）
    和 moe_layer_freq（0 = dense MLP，1 = MoE）。"""

    def __init__(self, path, tp=4, indexer_cp=False):
        raw = json.load(open(path))
        t = raw.get("text_config", raw)
        self.nl = t["num_hidden_layers"]
        self.hidden = t["hidden_size"]
        self.heads, self.kv_heads, self.head_dim = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
        self.dense_inter = t.get("dense_intermediate_size") or t["intermediate_size"]
        self.moe_inter = t["intermediate_size"]
        self.experts = t.get("num_local_experts", 0)
        self.topk = t.get("num_experts_per_tok", 0)
        self.shared = t.get("n_shared_experts", 0) or 0
        self.vocab = t["vocab_size"]
        sp = t.get("sparse_attention_config") or {}
        self.sparse = [bool(x) for x in (sp.get("sparse_attention_freq") or [0] * self.nl)]
        self.moe = [bool(x) for x in (t.get("moe_layer_freq") or [1] * self.nl)]
        assert len(self.sparse) == self.nl and len(self.moe) == self.nl, "per-layer lists do not match num_hidden_layers"
        self.idx_heads = sp.get("sparse_num_index_heads", 0)
        self.idx_dim = sp.get("sparse_index_dim", 0)
        self.topk_blocks = sp.get("sparse_topk_blocks", 0)
        self.sparse_block = sp.get("sparse_block_size", 0)
        self.tp = tp
        self.indexer_cp = indexer_cp

    def variant(self, i):
        return ("sparse" if self.sparse[i] else "full") + "+" + ("moe" if self.moe[i] else "mlp")

    def variants(self):
        return [self.variant(i) for i in range(self.nl)]

    def variant_counts(self):
        return collections.Counter(self.variants())

    @property
    def sparse_keys(self):
        """稀疏层每个 query 最多看到的 key 数（top-k 个 block，每个 block sparse_block_size 个 token）。"""
        return self.topk_blocks * self.sparse_block

    def gemm_kn(self):
        """TP 下各 GEMM 的 (K, N, act dtype, weight dtype)，从 config 推导。
        qkv_idx_proj：稀疏层把 indexer 的 q/k 融进 qkv 投影；index_q 在 TP 下是本 rank 分到的 kv head 份额，
        开 indexer CP（ATOM_M3_INDEXER_CP=1）时是全部 index head。"""
        H, tp = self.hidden, self.tp
        q = self.heads // tp * self.head_dim
        kv = max(1, self.kv_heads // tp) * self.head_dim
        idx_q = (self.idx_heads if self.indexer_cp else max(1, self.kv_heads // tp)) * self.idx_dim
        return {"qkv_proj": (H, q + 2 * kv, "fp8", "fp8"),
                "qkv_idx_proj": (H, q + 2 * kv + idx_q + self.idx_dim, "fp8", "fp8"),
                "o_proj": (q, H, "fp8", "fp8"),
                "dense_gate_up": (H, 2 * self.dense_inter // tp, "fp8", "fp8"),
                "dense_down": (self.dense_inter // tp, H, "fp8", "fp8"),
                "moe_router": (H, self.experts, "bf16", "bf16"),
                "lm_head": (H, self.vocab // tp, "bf16", "bf16")}


# ---------------------------------------------------------------- annotation 字段
def ann_fields(name):
    """'prefill[bs=1 tok=2517 ctx=105301 nctx=261.9M sqsq=.. sqsk=.. sk=..]' -> {'bs': 1, 'tok': 2517, ...}。
    ctx 有多条序列时是列表，这里不解析；sqsq = sum nq^2，sqsk = sum nq*nkv，sk = sum nkv（ATOM scheduler 定义）。"""
    out = {}
    for k, v in re.findall(r"\b(\w+)=([0-9.]+M?)(?=[\s\]])", name):
        out[k] = float(v[:-1]) * 1e6 if v.endswith("M") else float(v)
    return out


def ann_key(name):
    """decode annotation 的形状键：去掉每步都变的 ctx/sqsq/sqsk/sk/nctx，留下 bs/tok/d/spec。"""
    kind = name.split("[", 1)[0]
    keep = [f"{k}={v}" for k, v in re.findall(r"\b(\w+)=([^\s\]]+)", name) if k in ("bs", "tok", "p", "d", "spec")]
    return f"{kind}[{' '.join(keep)}]"


def attn_pairs(name, cfg):
    """(全注意力 query-key 对数, 稀疏层 query-key 对数, 说明)；从 annotation 的 sqsq/sqsk 推导。
    全注意力（causal，每个请求的 query 是最后 nq 个位置）：sum(nq*nkv - nq*(nq-1)/2) = sqsk - (sqsq - tok)/2。
    稀疏层每个 query 最多看 sparse_keys 个 key：取 min(全注意力对数, tok*sparse_keys)，所有请求的
    已有上下文都 >= sparse_keys 时这就是精确值。"""
    f = ann_fields(name)
    if not {"tok", "sqsq", "sqsk"} <= f.keys():
        return None, None, "annotation has no sqsq/sqsk (server needs ATOM_ENABLE_DETAILED_ANNOTATION=1)"
    full = f["sqsk"] - (f["sqsq"] - f["tok"]) / 2
    sparse = min(full, f["tok"] * cfg.sparse_keys) if cfg.sparse_keys else None
    return full, sparse, (f"derived from the annotation: full pairs = sqsk - (sqsq - tok)/2 = {full:.4g}; "
                          f"sparse pairs = min(full, tok*{cfg.sparse_keys})")


# ---------------------------------------------------------------- 切层
def comm_mask(w):
    isc = [bool(COMM.search(k["name"])) for k in w]
    # 一次 aiter 集合通信 = 先一个 Memcpy DtoD（拷进注册 buffer）再接通信 kernel：两行算同一组
    for i in range(len(w) - 2, -1, -1):
        if not isc[i] and isc[i + 1] and w[i]["name"].startswith("Memcpy DtoD"):
            isc[i] = True
    return isc


def runs(mask):
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def segment(w, nl):
    """一个 forward 的主 stream kernel（时间序，无 HtoD）-> (结果, 'ok' | 失败原因)。
    结果：head（embedding + 它的 all-reduce）、layers（nl 段）、tail（final norm 之后到 logits all-gather）。
    每层恰好一个 attention core；相邻两个 core 之间恰好两组通信（attention 输出、MLP/MoE 输出）；
    第 i 层从 core i 之前最后一组通信的结束处开始。"""
    cores = [i for i, k in enumerate(w) if CORE.search(k["name"])]
    if len(cores) != nl:
        return None, f"attention core {len(cores)} != {nl}"
    groups = runs(comm_mask(w))
    starts = []
    for li, ci in enumerate(cores):
        lo = cores[li - 1] if li else -1
        g = [r for r in groups if lo < r[0] and r[1] <= ci]
        if li > 0 and len(g) != 2:
            return None, f"layer {li}: {len(g)} collectives between cores {li - 1} and {li} (expected 2)"
        if not g:
            return None, "no embedding all-reduce before layer 0"
        starts.append(g[-1][1])
    after = [r for r in groups if r[0] > cores[-1]]
    if len(after) < 2:
        return None, f"last layer: {len(after)} collectives after its core (expected >= 2)"
    end = after[1][1]
    b = starts + [end]
    return {"head": w[:starts[0]], "layers": [w[b[i]:b[i + 1]] for i in range(nl)], "tail": w[end:],
            "bounds": b, "cores": cores}, "ok"


def observed_variant(seg):
    """从一层的 kernel 内容看它是什么层：有 indexer kernel（或稀疏 PA core）= sparse；有 MoE kernel = moe。
    和 config 的逐层类型比对，就是切层正确性的证明。"""
    names = [k["name"] for k in seg]
    core = next((n for n in names if CORE.search(n)), "")
    attn = "sparse" if (any(INDEXER.search(n) for n in names) or SPARSE_CORE.search(core)) else "full"
    ffn = "moe" if any(EXPERT.search(n) or n.startswith("moe_") or "topk_gating" in n for n in names) else "mlp"
    return f"{attn}+{ffn}"


# ---------------------------------------------------------------- kernel -> 模块
def _pre_core(n, attn, st):
    if SPLITK.search(n):
        return "@splitk"
    if EXPERT.search(n):
        return "gemm_unmapped"
    if GEMM.search(n):
        st["pre_gemm"] += 1
        return ("qkv_idx_proj" if attn == "sparse" else "qkv_proj") if st["pre_gemm"] == 1 else "gemm_unmapped"
    if "fusedQKNormIdxrQKNorm" in n or "qknorm_idxrqknorm" in n:
        return "attn_qk_idx_norm_rope_cache"
    if "qkv_norm_rope" in n or "rope_cache" in n or "reshape_and_cache" in n or "rotary" in n:
        return "attn_qk_norm_rope_cache"
    if "gather_cache" in n:
        return "attn_prefix_kv_gather"
    if "index_block_score" in n or "index_score" in n:
        return "indexer_score"
    if "topk_index" in n:
        return "indexer_topk"
    if "pa_decode_plan" in n:
        return "sparse_attn_plan"
    if COPY.search(n):
        return "aux_hidden_clone"
    if NORM.search(n) and st["pre_gemm"] == 0:
        return "input_norm"
    if "quant" in n:
        return "@quant"
    return "misc_small"


def _post_core(n, attn, st):
    if "pa_decode_reduce" in n:
        return f"{attn}_attn_core_reduce"
    if SPLITK.search(n):
        return "@splitk"
    if GEMM.search(n) and not EXPERT.search(n):
        st["post_gemm"] += 1
        return "o_proj" if st["post_gemm"] == 1 else "gemm_unmapped"
    if "quant" in n:
        return "@quant"
    return "attn_out_misc"


def _ffn(n, ffn, st, hidden):
    if SPLITK.search(n):
        return "@splitk"
    if ffn == "moe":
        if EXPERT.search(n):
            st["expert"] += 1
            kk = re.search(r"_K(\d+)_", n)
            if kk:
                return "moe_expert_gate_up" if int(kk.group(1)) == hidden else "moe_expert_down"
            return "moe_expert_gate_up" if st["expert"] == 1 else "moe_expert_down"
        if GEMM.search(n):
            st["ffn_gemm"] += 1
            return "moe_router" if st["ffn_gemm"] == 1 and st["expert"] == 0 else "gemm_unmapped"
        if any(s in n for s in ("topk_gating", "topk_softmax", "grouped_topk", "sigmoid_biased_topk")):
            return "moe_topk"
        if any(s in n for s in ("moe_gather_reduce", "moe_sum", "moe_combine", "moe_finalize")):
            return "moe_combine"
        if "scatter_preshuffle_scale" in n or "scale_shuffle" in n:
            return "moe_expert_scale_shuffle"
        if n.startswith("moe_") and "quant" in n:
            return "moe_expert_act_quant"
        if n.startswith("moe_") or "moe_align" in n or "moe_sorting" in n:
            return "moe_route"
        if NORM.search(n) and st["ffn_gemm"] == 0:
            return "post_attn_norm"
        if "quant" in n:
            return "@quant"
        if "elementwise" in n and st["expert"] == 0:
            return "moe_route"
        return "misc_small"
    if GEMM.search(n):
        st["ffn_gemm"] += 1
        return {1: "dense_gate_up", 2: "dense_down"}.get(st["ffn_gemm"], "gemm_unmapped")
    if any(s in n for s in ("swiglu", "silu_and_mul", "act_and_mul", "gelu")):
        return "dense_act"
    if NORM.search(n) and st["ffn_gemm"] == 0:
        return "post_attn_norm"
    if "quant" in n:
        return "@quant"
    return "misc_small"


def _resolve(rows):
    """split-k 的 reduce 归到它前面的 GEMM；激活量化归到它喂的下一个 GEMM（模块名 + '_quant'）。
    往后找 GEMM 时跳过中间的小 kernel（比如 aux 层在量化和 qkv GEMM 之间的 clone），碰到 core 或通信就停。"""
    for i, r in enumerate(rows):
        if r["m"] == "@splitk":
            r["m"] = next((x["m"] for x in reversed(rows[:i]) if x["role"] == "gemm"), "gemm_unmapped")
            r["role"] = "splitk"
    for i, r in enumerate(rows):
        if r["m"] == "@quant":
            target = "act_quant"
            for x in rows[i + 1:]:
                if x["m"] in GEMM_MODULES or x["m"] in EXPERT_MODULES:
                    target = x["m"] + "_quant"
                    break
                if x["m"].endswith("_attn_core") or x["m"].startswith("comm_"):
                    break
            r["m"], r["role"] = target, "quant"
    return rows


def _role(m, n):
    if m in GEMM_MODULES and not SPLITK.search(n):
        return "gemm"
    if m in EXPERT_MODULES:
        return "expert"
    return ""


def classify_layer(seg, var, hidden):
    """一层 -> [{'k': kernel, 'm': module, 'role': gemm|expert|splitk|quant|''}]，按位置定角色。"""
    attn, ffn = var.split("+")
    ci = next(i for i, k in enumerate(seg) if CORE.search(k["name"]))
    isc = comm_mask(seg)
    g = [r for r in runs(isc) if r[0] > ci]
    end = (len(seg), len(seg))
    g1, g2 = (g[0] if g else end), (g[1] if len(g) > 1 else end)
    st = collections.Counter()
    rows = []
    for i, k in enumerate(seg):
        n = k["name"]
        if i == ci:
            m = f"{attn}_attn_core"
        elif g1[0] <= i < g1[1]:
            m = "comm_ar_attn_out"
        elif g2[0] <= i < g2[1]:
            m = "comm_ar_moe_out" if ffn == "moe" else "comm_ar_mlp_out"
        elif isc[i]:
            m = "comm_other"
        elif i < ci:
            m = _pre_core(n, attn, st)
        elif i < g1[0]:
            m = _post_core(n, attn, st)
        elif i < g2[0]:
            m = _ffn(n, ffn, st, hidden)
        else:
            m = "misc_small"
        rows.append({"k": k, "m": m, "role": _role(m, n)})
    return _resolve(rows)


def classify_head(head):
    isc = comm_mask(head)
    return [{"k": k, "m": "comm_ar_embedding" if isc[i] else ("embedding" if "embedding" in k["name"] else "head_misc"),
             "role": ""} for i, k in enumerate(head)]


def classify_tail(tail):
    """final norm（decode；prefill 的 final norm 融在最后一层的 all-reduce 里）-> EAGLE3 aux hidden state 拷贝
    -> 取出要算 logits 的行 -> lm_head -> logits all-gather。"""
    isc = comm_mask(tail)
    rows, n_gemm = [], 0
    for i, k in enumerate(tail):
        n = k["name"]
        if isc[i]:
            m = "comm_ag_logits"
        elif SPLITK.search(n):
            m = "@splitk"
        elif GEMM.search(n) and not EXPERT.search(n):
            n_gemm += 1
            m = "lm_head" if n_gemm == 1 else "gemm_unmapped"
        elif NORM.search(n):
            m = "final_norm"
        elif COPY.search(n):
            m = "aux_hidden_copy"
        elif "quant" in n:
            m = "@quant"
        elif n_gemm == 0 and ("elementwise" in n or "index" in n or "gather" in n):
            m = "logits_row_select"
        else:
            m = "tail_misc"
        rows.append({"k": k, "m": m, "role": _role(m, n)})
    return _resolve(rows)


def classify_forward(seg, cfg, var_seq=None):
    """整个 forward：head 行、每层的行、tail 行。var_seq 默认用 config 的逐层类型。"""
    var_seq = var_seq or cfg.variants()
    return (classify_head(seg["head"]),
            [classify_layer(s, v, cfg.hidden) for s, v in zip(seg["layers"], var_seq)],
            classify_tail(seg["tail"]))


# ---------------------------------------------------------------- shape 与派生指标
def dtype_from_name(n, default):
    if "a8w4_tdm" in n and "fp4" in n:
        return ("fp4", "fp4")  # a8w4_tdm*_fp4：_fp4 后缀是激活格式（MXFP4），不是 uint8
    if "gemm_a8w8" in n:
        return ("fp8", "fp8")
    if "a16w16" in n:
        return ("bf16", "bf16")
    if "a4w4" in n:
        return ("fp4", "fp4")
    return default


BYTES = {"bf16": 2.0, "fp8": 1.0, "fp4": 0.5 + 1 / 32}  # fp4 带每 32 个元素一个 e8m0 scale


def recorded_mkn(trace, k):
    """录了 shape（record_shapes=1）时，从发射 kernel 的 cpu_op 的 Input Dims 读 (M, K, N)：找 [M, K] x [N, K]。"""
    o = trace.op_of(k) if trace else None
    dims = ((o or {}).get("args") or {}).get("Input Dims") or []
    two = [d for d in dims if isinstance(d, list) and len(d) == 2]
    for i in range(len(two) - 1):
        x, w = two[i], two[i + 1]
        if x[1] == w[1] and x[0] and w[0]:
            return x[0], x[1], w[0]
    return None


def load_peaks(path):
    """峰值 json：{"bf16": TFLOP/s, "fp8": ..., "fp4": ..., "hbm_tbs": TB/s}，用于"超过峰值 = 标签或 shape 错"的保护。"""
    if not path:
        return None
    p = json.load(open(path))
    return {k: float(v) for k, v in p.items() if not k.startswith("_") and isinstance(v, (int, float))}


def _guard(tf, tb, wd, peaks):
    if not peaks:
        return None
    pk = peaks.get(wd)
    if (pk and tf > pk) or (peaks.get("hbm_tbs") and tb > peaks["hbm_tbs"]):
        return f"over peak ({tf:.0f} TF/s, {tb:.1f} TB/s) -> label or shape wrong, blanked"
    return None


def gemm_metrics(row, M, cfg, trace=None, peaks=None):
    """普通 GEMM 行（qkv/o/dense/router/lm_head）：M,K,N,dtype,TFLOP/s,TB/s。
    有录到的 Input Dims 就用（src=recorded），否则 K/N 来自 config@TP、M 来自 annotation 的 tok（src=derived）。"""
    m, k = row["m"], row["k"]
    if row["role"] != "gemm" or m not in cfg.gemm_kn():
        return {}
    K, N, ad, wd = cfg.gemm_kn()[m]
    ad, wd = dtype_from_name(k["name"], (ad, wd))
    src = f"derived: K,N from config at TP{cfg.tp}, M = annotation tok"
    rec = recorded_mkn(trace, k)
    if rec:
        note = "" if (rec[1], rec[2]) == (K, N) else f" (config gives K,N = {K},{N})"
        M, K, N = rec
        src = "recorded Input Dims" + note
    flops = 2.0 * M * K * N
    byts = M * K * BYTES[ad] + K * N * BYTES[wd] + M * N * 2
    tf, tb = flops / k["dur"] / 1e6, byts / k["dur"] / 1e6
    g = _guard(tf, tb, wd, peaks)
    return {"m": int(M), "k": K, "n": N, "dtype": f"{ad}/{wd}", "tflops": "" if g else round(tf, 1),
            "tbytes_s": "" if g else round(tb, 2), "src": g or src}


def expert_metrics(row, M, cfg, peaks=None):
    """融合 MoE（MoE TP：每个 rank 持有全部专家、intermediate 切 TP 份；共享专家作为第 E+1 个专家一起算）。
    每个 token 有 topk 个路由专家 + 共享专家，(token, expert) 对数 = tok*(topk+shared)；
    权重读取量按均匀路由下被触达的专家数估计：E*(1-(1-topk/E)^tok) + shared。都是推导值。"""
    k = row["k"]
    E, tk, sh = cfg.experts, cfg.topk, cfg.shared
    pairs = M * (tk + sh)
    touched = E * (1 - (1 - tk / E) ** M) + sh if E else 0
    inter = cfg.moe_inter // cfg.tp
    if row["m"] == "moe_expert_gate_up":
        K, N = cfg.hidden, 2 * inter
        out = pairs * inter * BYTES["fp4"]  # 融合激活后写 fp4 + scale
    else:
        K, N = inter, cfg.hidden
        out = pairs * N * 2
    flops = 2.0 * pairs * K * N
    byts = touched * K * N * BYTES["fp4"] + pairs * K * BYTES["fp4"] + out
    tf, tb = flops / k["dur"] / 1e6, byts / k["dur"] / 1e6
    g = _guard(tf, tb, "fp4", peaks)
    return {"m": round(pairs), "k": K, "n": N, "dtype": "fp4/fp4", "tflops": "" if g else round(tf, 1),
            "tbytes_s": "" if g else round(tb, 2),
            "src": g or (f"derived: M = tok*(topk+shared) = {int(M)}*{tk + sh}; weight read for ~{touched:.1f} of "
                         f"{E + sh} experts (uniform routing)")}


def core_metrics(row, ann_name, cfg, peaks=None):
    """attention core：FLOPs = 4 * heads/TP * head_dim * query-key 对数（QK^T 和 PV 各 2 FLOP/MAC）。
    对数从 annotation 推导；decode 的全注意力 core 另给 KV 读取带宽（fp8 KV，K+V 各 head_dim 字节/kv head）。"""
    k = row["k"]
    full, sparse, how = attn_pairs(ann_name, cfg)
    if full is None:
        return {"src": how}
    pairs = sparse if row["m"].startswith("sparse") else full
    if row["m"].endswith("_reduce") or pairs is None:
        return {}
    hpr = cfg.heads // cfg.tp
    flops = 4.0 * hpr * cfg.head_dim * pairs
    tf = flops / k["dur"] / 1e6
    tb = ""
    f = ann_fields(ann_name)
    if ann_name.startswith("decode[") and row["m"].startswith("full") and "sk" in f:
        kvb = f["sk"] * max(1, cfg.kv_heads // cfg.tp) * cfg.head_dim * 2 * 1.0
        tb = round(kvb / k["dur"] / 1e6, 2)
    g = _guard(tf, tb or 0.0, "bf16", peaks)
    return {"m": "", "k": "", "n": "", "dtype": "bf16 q / fp8 kv", "tflops": "" if g else round(tf, 1),
            "tbytes_s": "" if g else tb, "src": g or f"{how}; pairs used {pairs:.4g}"}


def comm_metrics(row, M, cfg):
    """all-reduce 的算法带宽：bytes = tok*hidden*2（bf16），推导值。"""
    if not row["m"].startswith("comm_ar_") or row["k"]["name"].startswith("Memcpy") or "sync_kernel" in row["k"]["name"]:
        return {}
    byts = M * cfg.hidden * 2
    return {"tbytes_s": round(byts / row["k"]["dur"] / 1e6, 3),
            "src": f"derived: algbw, bytes = tok*{cfg.hidden}*2 (bf16)"}


def row_metrics(row, ann_name, cfg, trace=None, peaks=None):
    M = ann_fields(ann_name).get("tok", 0)
    if row["role"] == "gemm":
        return gemm_metrics(row, M, cfg, trace, peaks)
    if row["role"] == "expert":
        return expert_metrics(row, M, cfg, peaks)
    if row["m"].endswith("_attn_core"):
        return core_metrics(row, ann_name, cfg, peaks)
    if row["m"].startswith("comm_ar_"):
        return comm_metrics(row, M, cfg)
    if row["role"] == "splitk":
        return {"src": "split-k reduce of the GEMM above"}
    return {}


# ---------------------------------------------------------------- 小工具
def union(iv):
    """区间并集的总长度。"""
    tot, end = 0.0, None
    for a, b in sorted(iv):
        if end is None or a > end:
            tot += b - a
            end = b
        elif b > end:
            tot += b - end
            end = b
    return tot


def overlap_us(k, iv):
    lo, hi = k["ts"], k["ts"] + k["dur"]
    return min(k["dur"], sum(max(0.0, min(hi, b) - max(lo, a)) for a, b in iv if a < hi and b > lo))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def write_csv(path, rows, fields=None):
    import csv
    if not rows:
        open(path, "w").close()
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
