#!/usr/bin/env python3
"""MiniMax-M3 TP4 的逐层 kernel breakdown 做成一个 Excel tab（analyze.sh 之后的可选步骤，需要 openpyxl）。

输入：analyze.sh 跑完的 OUT 目录（rank A 的精简 trace、model_config.json、server.log、capture_meta.json，以及
breakdown/ 下的 prefill_meta.json、decode_meta.json、window_summary.json）；切层和 kernel 分类用 m3_common.py。
  - prefill = 一个 forward（默认 analyze.sh 选的代表 prefill），decode = 一个形状键的全部 step（默认 analyze.sh
    选的形状键）；
  - 每种层（dense L0-2，sparse L3-59）取 kernel 序列和这种层最常见序列相同的所有层实例（decode 再乘所有 step），
    每个 kernel 取它们的中位数；只算主 stream；
  - decode 是 HIP graph 回放：profiler 可能丢掉 kernel 时间戳记录，从那一处起这个 step 里每个 kernel 名字配的是
    下一个 kernel 的时长（层总和不变）。逐层和 L0-L8 的中位数比对，每个 step 从第一个错位的层起（连同它前一层）
    都不用；
  - GEMM 形状按每个 rank（TP4）从 config 推导：M = annotation 的 token 数，MoE M = token 数 x 5；
  - WORKLOAD=fixed 的采集一样能用：备注写明是固定 shape 重放，不开 profiler 的步长取窗口前客户端量的步长。
平台、tab 名、标题和配置行来自 server.log / capture_meta.json；--stack 写被采集镜像里的版本，标题取 " stack" 前面
的部分。
用法: m3_xlsx.py --run OUT --stack "11-04 stack: ATOM <rev>, aiter <rev>, FlyDSL <rev>" [--rank 0] [--config C]
                 [--peaks peaks.json] [--master 汇总.xlsx] [--check]
写出 OUT/breakdown/M3_TP4_layer_breakdown_<tab>.xlsx（只有这个 tab）；给了 --master 时把 tab 加进这个 workbook
（同名 tab 原位替换，文件不存在就新建）。--check 只打印表格，不写文件。
"""
import argparse
import collections
import datetime
import json
import math
import os
import re
import statistics
import sys

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

import m3_common as mc

PLATFORM = {"gfx1250": "MI455", "gfx950": "MI355X"}
HBM_TBS = {"gfx950": 8.0}       # 没给 --peaks 时 HBM 峰值（TB/s）的后备
KINDS = {"full+mlp": "dense", "sparse+moe": "sparse"}
SYNC = re.compile(r"start_sync_kernel|end_sync_kernel")
# 一个层实例里有这么多个 kernel 偏离参考值 1.8 倍以上，就算错位
DRIFT_KERNELS, DRIFT_RATIO = 4, 1.8

# skill 的模块名 -> 表里的模块名
MOD = {"input_norm": "attn_input_norm", "qkv_proj_quant": "attn_input_quant", "qkv_idx_proj_quant": "attn_input_quant",
       "qkv_proj": "attn_qkv_gemm", "qkv_idx_proj": "attn_qkv_gemm",
       "attn_qk_norm_rope_cache": "attn_qk_norm_rope_cache", "attn_qk_idx_norm_rope_cache": "attn_qk_norm",
       "attn_prefix_kv_gather": "attn_kv_gather", "aux_hidden_clone": "attn_input_misc",
       "indexer_score": "sparse_index_score", "indexer_topk": "sparse_index_topk", "sparse_attn_plan": "attn_core_plan",
       "full_attn_core": "attn_core_dense", "sparse_attn_core": "attn_core_sparse",
       "full_attn_core_reduce": "attn_core_reduce", "sparse_attn_core_reduce": "attn_core_reduce",
       "o_proj_quant": "attn_out_quant", "o_proj": "attn_out_gemm",
       "comm_ar_attn_out": "all_reduce_attn", "comm_ar_mlp_out": "all_reduce_mlp", "comm_ar_moe_out": "all_reduce_mlp",
       "comm_ag_indexer_cp": "all_gather_index",
       "dense_gate_up_quant": "mlp_quant", "dense_down_quant": "mlp_quant", "dense_gate_up": "mlp_gemm",
       "dense_down": "mlp_gemm", "dense_act": "mlp_act", "moe_router": "moe_gate_gemm", "moe_route": "moe_sort",
       "moe_expert_act_quant": "moe_quant", "moe_expert_gate_up_quant": "moe_quant",
       "moe_expert_down_quant": "moe_quant", "moe_expert_scale_shuffle": "moe_misc",
       "moe_expert_gate_up": "moe_expert_gemm1", "moe_expert_down": "moe_expert_gemm2", "moe_combine": "moe_reduce"}


def sheet_module(m, name):
    if SYNC.search(name):
        return "all_reduce_wait"
    if name.startswith("Memcpy DtoD"):
        return "device_copy"
    return MOD.get(m, m)


def pretty(n):
    """表里显示的 kernel 名；aiter 的 Itanium mangled 名字（c++filt 读不了其中的 fp8 模板参数）手工拆。"""
    if not n.startswith("_ZN"):
        return mc.short(n)
    i, parts = 3, []
    while i < len(n) and n[i].isdigit():
        j = i
        while j < len(n) and n[j].isdigit():
            j += 1
        ln = int(n[i:j])
        parts.append(n[j:j + ln])
        i = j + ln
    base = "::".join(parts)
    ints = re.findall(r"Li(\d+)E", n[i:]) if n[i:i + 1] == "I" else []
    return base + (f"<{','.join(ints)}>" if ints and "quant" in base else "")


def layers_of(T, cfg, a):
    seg, _ = mc.segment(T.main_window(a), cfg.nl)
    if not seg or [mc.observed_variant(x) for x in seg["layers"]] != cfg.variants():
        return None
    _, rows, _ = mc.classify_forward(seg, cfg)
    starts = [L[0]["ts"] for L in seg["layers"]] + [seg["tail"][0]["ts"] if seg["tail"] else a["ts"] + a["dur"]]
    return [{"li": i, "rows": rows[i], "span": starts[i + 1] - starts[i]} for i in range(cfg.nl)]


def sig_of(x):
    return tuple(mc.short(r["k"]["name"]) for r in x["rows"])


def pick_sig(insts):
    """这种层最常见的 kernel 序列；并列时（比如 3 个 dense 层各不相同；或在最高次数的 1% 以内，少数 decode step
    多出一个别的 kernel）取时间居中的那个层的序列。"""
    cnt = collections.Counter(sig_of(x) for x in insts)
    top = max(cnt.values())
    tied = sorted((sum(r["k"]["dur"] for r in x["rows"]), sig_of(x)) for x in insts if cnt[sig_of(x)] >= 0.99 * top)
    return tied[len(tied) // 2][1]


def step_stats(T, a, L):
    """一个 forward 在 GPU 时间线上的 kernel busy、kernel 之间的空闲、最大的 gap，以及 gap 两边的
    (层, 模块, kernel)（不在层里时为 None）。"""
    w = T.main_window(a)
    pos = {id(r["k"]): (x["li"], r["m"], r["k"]["name"]) for x in L for r in x["rows"]}
    end, idle, big = w[0]["ts"] + w[0]["dur"], 0.0, (0.0, None, None)
    for k0, k1 in zip(w, w[1:]):
        end = max(end, k0["ts"] + k0["dur"])
        g = k1["ts"] - end
        idle += max(0.0, g)
        if g > big[0]:
            big = (g, pos.get(id(k0)), pos.get(id(k1)))
    return {"wall": a["dur"], "busy": sum(k["dur"] for k in w), "idle": idle, "gap": big[0], "at": big[1:]}


def drop_misaligned(steps, VAR):
    """每个 decode step 只留第一个错位的层实例之前的层，再去掉紧挨着它的前一层。"""
    ref = collections.defaultdict(list)
    for L in steps:
        for x in L[:9]:
            for r in x["rows"]:
                ref[(VAR[x["li"]], r["m"], r["k"]["name"])].append(r["k"]["dur"])
    ref = {k: statistics.median(v) for k, v in ref.items() if len(v) >= 50}
    kept, onsets = [], []
    for L in steps:
        onset = None
        for x in L:
            off = sum(1 for r in x["rows"] if (t := ref.get((VAR[x["li"]], r["m"], r["k"]["name"])))
                      and abs(math.log(max(r["k"]["dur"], 1e-3) / t)) > math.log(DRIFT_RATIO))
            if off >= DRIFT_KERNELS:
                onset = x["li"]
                break
        onsets.append(onset)
        kept.append(L if onset is None else L[:max(0, onset - 1)])
    return kept, onsets


def spec(phase, x, f, cfg):
    """一行的 (MxKxN 标签, FLOPs, 字节数或 None, dtype)，算不了时返回 None。f = annotation 的字段。
    GEMM / MoE 的 M = forward 实际跑的行数（tok_run：token graph 的 bucket 会补到调度 token 数以上）；
    attention 和 indexer 按调度的 token 数算。"""
    r, m, n = x, x["m"], x["k"]["name"]
    tok = int(f["tok"])
    rows = int(f.get("tok_run", tok))
    full = f["sqsk"] - (f["sqsq"] - f["tok"]) / 2
    hpr, d, kvh = cfg.heads // cfg.tp, cfg.head_dim, max(1, cfg.kv_heads // cfg.tp)
    B = mc.BYTES
    ctx = f"{int(f['sk'])}" if phase == "prefill" else f"({int(f['bs'])}x{f['sk'] / f['bs'] / 1e3:.0f}K)"
    if r["role"] == "gemm" and m in cfg.gemm_kn():
        K, N, ad, wd = cfg.gemm_kn()[m]
        ad, wd = mc.dtype_from_name(n, (ad, wd))
        return f"{rows}x{K}x{N}", 2.0 * rows * K * N, rows * K * B[ad] + K * N * B[wd] + rows * N * 2, f"{ad}/{wd}"
    if r["role"] == "expert":
        E, tk, sh = cfg.experts, cfg.topk, cfg.shared
        pairs = rows * (tk + sh)
        touched = E * (1 - (1 - tk / E) ** rows) + sh
        inter = cfg.moe_inter // cfg.tp
        act = "bf16" if "a16w4" in n else "fp4"
        K, N, out = ((cfg.hidden, 2 * inter, pairs * inter * B[act]) if m == "moe_expert_gate_up"
                     else (inter, cfg.hidden, pairs * cfg.hidden * 2))
        return (f"{pairs}x{K}x{N}", 2.0 * pairs * K * N, touched * K * N * B["fp4"] + pairs * K * B[act] + out,
                f"{'bf16' if act == 'bf16' else 'mxfp4'}/mxfp4")
    if m == "full_attn_core":
        if phase == "prefill":
            return f"{tok}x{ctx}x{hpr}h", 4.0 * hpr * d * full, None, "bf16/bf16"
        return f"{tok}x{ctx}x{hpr}h", 4.0 * hpr * d * full, f["sk"] * kvh * d * 2 * B["fp8"], "bf16/fp8"
    if m == "sparse_attn_core":
        return f"{tok}x{cfg.sparse_keys}x{hpr}h", 4.0 * hpr * d * min(full, tok * cfg.sparse_keys), None, "bf16/fp8"
    if m == "indexer_score":
        # indexer CP（只在 decode）：全部 index head 各算 1/tp 的 block；prefill 只算本 rank 自己的 head
        cp = cfg.indexer_cp and phase == "decode"
        heads = cfg.idx_heads if cp else max(1, cfg.kv_heads // cfg.tp)
        frac = 1 / cfg.tp if cp else 1
        return (f"{tok}x{ctx}" + (f"/{cfg.tp}x{cfg.idx_dim}x{heads}h" if cp else f"x{cfg.idx_dim}"),
                2.0 * full * frac * heads * cfg.idx_dim,
                None if phase == "prefill" else f["sk"] * frac * cfg.idx_dim * B["fp8"], "fp8 index")
    return None


def block(phase, insts_by_kind, f, cfg, pool_by_kind=None):
    """层类型 -> (行 [(表里的模块, kernel, 中位数 us, spec)], span 中位数 us, 层数, 用到的层实例)。
    kernel 序列从 pool_by_kind 里选（decode：错位过滤之前的全部层实例）。"""
    out = {}
    for kd, insts in insts_by_kind.items():
        sig = pick_sig((pool_by_kind or insts_by_kind)[kd])
        same = [x for x in insts if sig_of(x) == sig]
        rows = []
        for i, r in enumerate(same[0]["rows"]):
            dur = statistics.median(x["rows"][i]["k"]["dur"] for x in same)
            rows.append((sheet_module(r["m"], r["k"]["name"]), pretty(r["k"]["name"]), dur, spec(phase, r, f, cfg)))
        out[kd] = (rows, statistics.median(x["span"] for x in same), cfg.variant_counts()[kd], same)
    return out


def gap_place(at):
    """一个 gap 两边的 ((层, 模块, kernel), (层, 模块, kernel)) -> (gap 开始所在的层, 'L0 a -> b' 标签,
    'in L0 between a and b' 短语)；任一边不在层里时返回 None。"""
    p0, p1 = at
    if not p0 or not p1:
        return None
    m0, m1 = sheet_module(p0[1], p0[2]), sheet_module(p1[1], p1[2])
    if p0[0] == p1[0]:
        return p0[0], f"L{p0[0]} {m0} -> {m1}", f"in L{p0[0]} between {m0} and {m1}"
    return p0[0], f"L{p0[0]} {m0} -> L{p1[0]} {m1}", f"between L{p0[0]} {m0} and L{p1[0]} {m1}"


def collect(trace, config, prefill, decode, indexer_cp=False):
    cfg = mc.M3Config(config, tp=4, indexer_cp=indexer_cp)
    T = mc.Trace(trace)
    VAR = cfg.variants()
    a = next((x for x in T.forwards("prefill") if x["name"].startswith(prefill)), None)
    if a is None:
        sys.exit(f"no prefill forward starts with {prefill!r} in {trace}")
    L = layers_of(T, cfg, a)
    if not L:
        sys.exit(f"{a['name']} does not segment into {cfg.nl} layers of the config's types")
    corr = {(k.get("args") or {}).get("correlation") for k in T.main_window(a)}
    print(f"prefill {a['name'][:60]}: {len(T.main_window(a))} kernels, {len(corr)} correlation ids "
          f"({'eager launches' if len(corr) > 100 else 'graph replay'})")
    pre = block("prefill", {kd: [x for x in L if VAR[x["li"]] == kd] for kd in KINDS},
                mc.ann_fields(a["name"]), cfg)
    pst = step_stats(T, a, L)

    steps, fields, names, sst = [], [], [], []
    for s in T.forwards("decode"):
        if mc.ann_key(s["name"]) == decode:
            L = layers_of(T, cfg, s)
            if L:
                steps.append(L)
                fields.append(mc.ann_fields(s["name"]))
                names.append(s["name"])
                sst.append(step_stats(T, s, L))
    if not steps:
        sys.exit(f"no decode step of {decode} segments")
    kept, onsets = drop_misaligned(steps, VAR)
    n_on = sorted(o for o in onsets if o is not None)
    mis = (f"misaligned from layer (p10/p50/p90 over steps that misalign): {n_on[len(n_on) // 10]}/"
           f"{n_on[len(n_on) // 2]}/{n_on[9 * len(n_on) // 10]}, " if n_on else "no step misaligned, ")
    print(f"decode {decode}: {len(steps)} steps; {mis}{onsets.count(None)} steps never; "
          f"kept {sum(map(len, kept))} of {len(steps) * cfg.nl} layer instances")
    fmed = {k: statistics.median(f[k] for f, L in zip(fields, kept) if L) for k in ("tok", "bs", "sqsq", "sqsk", "sk")}
    dec = block("decode", {kd: [x for L in kept for x in L if VAR[x["li"]] == kd] for kd in KINDS}, fmed, cfg,
                {kd: [x for L in steps for x in L if VAR[x["li"]] == kd] for kd in KINDS})
    dst = {k: statistics.median(s[k] for s in sst) for k in ("wall", "busy", "idle", "gap")}
    where = collections.Counter((s["at"][0] or (None,) * 3)[:2] + (s["at"][1] or (None,) * 3)[:2] for s in sst)
    loc, dst["n_loc"] = where.most_common(1)[0]
    at_loc = [s for s in sst if (s["at"][0] or (None,) * 3)[:2] + (s["at"][1] or (None,) * 3)[:2] == loc]
    dst["loc_gap"] = statistics.median(s["gap"] for s in at_loc)
    dst["place"] = gap_place(at_loc[0]["at"]) if 2 * dst["n_loc"] >= len(sst) else None
    print(f"decode GPU timeline p50: wall {dst['wall'] / 1e3:.2f} ms, busy {dst['busy'] / 1e3:.2f} ms, idle "
          f"{dst['idle'] / 1e3:.2f} ms; largest gap {dst['loc_gap'] / 1e3:.2f} ms at "
          f"{dst['place'][1] if dst['place'] else 'no fixed place'} in {dst['n_loc']} of {len(sst)} steps; "
          f"prefill wall {pst['wall'] / 1e3:.2f} ms busy {pst['busy'] / 1e3:.2f} ms")
    heads = (a["name"].split(" sqsq")[0] + "]", names[len(names) // 2].split(" sqsq")[0] + "]")
    npre_ok = sum(1 for p in T.forwards("prefill") if layers_of(T, cfg, p))
    return dict(cfg=cfg, a=a, pre=pre, fmed=fmed, dec=dec, kept=kept, nsteps=len(steps), pst=pst, dst=dst,
                heads=heads, onsets=onsets, npre_ok=npre_ok, pre_eager=len(corr) > 100,
                ndec_all=sum(1 for _ in T.forwards("decode")), npre_all=sum(1 for _ in T.forwards("prefill")))


def check(dec, kept, cfg):
    """留下的 decode 层实例不应再有错位的样子：同一位置的 kernel 在各层组里的中位数应该持平。"""
    for kd in KINDS:
        rows, _, _, same = dec[kd]
        idx = sorted({x["li"] for x in same})
        groups = [idx[j:j + 10] for j in range(0, len(idx), 10)]
        print(f"\n{kd}: {len(same)} instances, layers {idx[0]}-{idx[-1]}; per-slot median by layer group")
        print("  " + " " * 52 + " ".join(f"L{g[0]:>2}-{g[-1]:<2}" for g in groups))
        for i, (mod, name, dur, _) in enumerate(rows):
            cells = []
            for g in groups:
                v = [x["rows"][i]["k"]["dur"] for x in same if x["li"] in g]
                cells.append(f"{statistics.median(v):6.2f}" if v else "    - ")
            print(f"  {mod[:20]:20s} {name[:30]:30s} " + " ".join(cells) + f"   n/group {[sum(1 for x in same if x['li'] in g) for g in groups]}" * (i == 0))
    per_layer = collections.Counter(x["li"] for L in kept for x in L)
    print("\nkept instances per layer:", " ".join(f"L{li}:{per_layer[li]}" for li in range(0, cfg.nl, 6)))


def server_env(run):
    """server.log 第一行 'ARCH=.. CONC=.. KV=.. SHAPES=.. ACCEPT_RATE=.. env: A=1 B=2 ..'
    -> (开头的字段, 实际生效的环境；同一个变量赋值多次时后面的生效，和 env(1) 一样)。"""
    first = open(f"{run}/server.log", errors="replace").readline()
    pre, _, envs = first.partition(" env: ")
    head = dict(t.split("=", 1) for t in pre.split() if "=" in t)
    env = dict(t.split("=", 1) for t in envs.split() if "=" in t)
    return head, env


def platform(run):
    head, env = server_env(run)
    meta = json.load(open(f"{run}/capture_meta.json"))
    w = meta["windows"]["main"]
    when = datetime.datetime.fromtimestamp(w["start_profile_sent"], datetime.timezone.utc)
    return dict(head=head, env=env, conc=meta["conc"], offset=w.get("offset_s"), window=w["window_s"],
                workload=meta.get("workload", "agentx"), fixed=(meta.get("fixed") or {}).get("params"),
                date=when.strftime("%m-%d"), name=PLATFORM.get(head.get("ARCH"), head.get("ARCH", "?")))


def touched(cfg, tok):
    return cfg.experts * (1 - (1 - cfg.topk / cfg.experts) ** tok) + cfg.shared


def notes_text(R, A, P, win):
    cfg, a, pre, fmed, dec, kept, dst = (R[k] for k in ("cfg", "a", "pre", "fmed", "dec", "kept", "dst"))
    head, env, F = P["head"], P["env"], P["fixed"] if P["workload"] == "fixed" else None
    itl, tps = win.get("aiperf_itl_finished_p50_ms"), win["toks_per_fwd"]
    fp = mc.ann_fields(a["name"])
    fd = mc.ann_fields(R["heads"][1])
    bs, ctx_k, nsteps = int(fmed["bs"]), round(fmed["sk"] / fmed["bs"] / 1e3), R["nsteps"]
    dec_dense = dec["full+mlp"][3][0]["li"]
    n_kept = f"{sum(map(len, kept)):,} of {nsteps * cfg.nl:,} layer instances kept"
    use = f"the tables use {len(dec['full+mlp'][3]):,} (dense L{dec_dense}) and {len(dec['sparse+moe'][3]):,}"
    flags = ([f"ATOM_MONO_ENABLE={env['ATOM_MONO_ENABLE']}"] if "ATOM_MONO_ENABLE" in env else []) + [
        "prefill token graph " + ("on" if env.get("ATOM_MINIMAX_M3_PREFILL_TOKEN_GRAPH") == "1" else "off")] + (
        ["indexer CP on (ATOM_M3_INDEXER_CP=1)"] if cfg.indexer_cp else [])
    rate = head.get("ACCEPT_RATE", "")
    kn = cfg.gemm_kn()
    q, kv = cfg.heads // cfg.tp * cfg.head_dim, max(1, cfg.kv_heads // cfg.tp) * cfg.head_dim
    iq = kn["qkv_idx_proj"][1] - kn["qkv_proj"][1] - cfg.idx_dim
    qkv_line = (f"- QKV N: dense {kn['qkv_proj'][1]} (q {q} + k {kv} + v {kv}); sparse {kn['qkv_idx_proj'][1]} "
                f"(+ index q {iq}{f' = all {cfg.idx_heads} index heads' if cfg.indexer_cp else ''} + index k {cfg.idx_dim})")
    pre_rows = fp.get("tok_run", fp["tok"])
    acc = (f"  draft acceptance forced to {rate} ({tps:.2f} tokens/step)" if rate
           else f"  real draft acceptance ({tps:.2f} tokens/step)")
    if F:
        how = [f"Trace (fixed-shape replay, not AgentX; traced {P['date']})",
               f"- server started with the AgentX c{P['conc']} recipe settings, no aiperf; {F['bs']} prompts of "
               f"{F['ctx']:,} random token ids", "  prefilled one at a time into the prefix cache, then sent together: "
               f"every decode step is bs={F['bs']};",
               f"- one {P['window']:.0f} s profiler window: {F['pre_s']:g} s of decode, one prefill, {F['post_s']:g} s "
               "of decode;", acc]
        pick = "  (the only prefill in the window" + (f"; same shape as the representative prefill of {A.match})"
                                                   if A.match else ")")
    else:
        how = [f"Trace (in-run AgentX, not a matched replay; traced {P['date']})",
               f"- AgentX c{P['conc']}, one {P['window']:.0f} s profiler window {P['offset']:.0f} s into the run;", acc,
               *([f"  forwards span only {win['window_ms'] / 1e3:.1f} s of the window: no request was running in "
                  "the rest"] if win.get("window_ms", 1e3 * P["window"]) < 900 * P["window"] else [])]
        pick = ("  (picked by --prefill)" if A.prefill and not A.match else
                f"  (chosen to match the {A.match} tab; the median-wall prefill of this window is "
                f"{A.skill_pre.split(' nctx')[0]}])" if A.prefill else
                "  (the only prefill in the window)" if R["npre_all"] == 1 else
                f"  (median wall of the {R['npre_ok']} prefills in the window that segment into {cfg.nl} layers)")
    t = [f"MiniMax-M3-MXFP4 TP4 + EAGLE3 K3, rank {A.rank}",
         f"{head.get('KV', 'fp8').upper()} KV, online PTPC FP8, Quick Reduce "
         f"{env.get('AITER_QUICK_REDUCE_QUANTIZATION', 'off')}, ATOM_PA_FLYDSL={env.get('ATOM_PA_FLYDSL', '0')}, "
         "prefix caching",
         A.hw,
         f"{A.stack}; {', '.join(flags)}",
         "",
         *how,
         f"- {R['heads'][0]}: {int(fp['sk'] - fp['tok']):,} cached + {int(fp['tok']):,} new tokens, "
         + (f"one token-graph forward ({int(pre_rows):,}-row bucket)" if "tok_run" in fp
            else f"one {'eager' if R['pre_eager'] else 'graph-replayed'} forward"),
         pick,
         f"- {R['heads'][1]}: {bs} request{'s' * (bs != 1)} x ~{ctx_k}K ctx, {int(fmed['tok'] / fmed['bs'])} tokens each "
         f"(EAGLE3 K{int(fd.get('spec', 3))} verify), {nsteps} steps"
         + (f" of the window's {R['ndec_all']} decode steps" if R["ndec_all"] > nsteps else ""),
         *([f"  ({f'chosen to match the {A.match} tab' if A.match else 'picked by --decode'}; the most common decode "
            f"shape of this window is {A.skill_dec['key']}, {A.skill_dec['steps']} steps)"]
           if A.decode and A.decode != A.skill_dec["key"] else []),
         "- dense = L0-L2 (full attention, dense MLP); sparse = L3-L59 (indexer + sparse attention, MoE)",
         *([f"- indexer CP, decode only: each rank scores all {cfg.idx_heads} index heads over 1/{cfg.tp} of the KV "
            "blocks,", "  then local top-k -> all_gather_index -> merge top-k; prefill keeps the TP indexer "
            "(this rank's head)"] if cfg.indexer_cp else []),
         "",
         "How the numbers are built",
         "- layers and kernel modules as in the minimax-m3-agentx-trace skill",
         "- each kernel = median over every layer of the kind with the same kernel sequence (decode: x all steps);",
         f"  the 3 dense layers all differ, the one of median time is shown (prefill L{pre['full+mlp'][3][0]['li']}, "
         f"decode L{dec_dense})",
         "- ratio = kernel / layer_time; layer_time = sum of kernel durations (GPU busy, incl. wait kernels)",
         "- layer_span = first kernel start of the layer to first kernel start of the next layer (median),",
         "  gpu_idle = layer_span - layer_time = GPU gaps between kernels"]
    has_sync = any(r[0] == "all_reduce_wait" for data in (pre, dec) for kd in KINDS for r in data[kd][0])
    has_arn = any("ar_rmsnorm" in r[1] for kd in KINDS for r in pre[kd][0])
    if has_sync:
        t.append(f"- all_reduce_wait ({head.get('ARCH', '')} start_sync / end_sync) = waiting for the other ranks, "
                 f"not work{';' if has_arn else ''}")
        if has_arn:
            t.append("  prefill all-reduce = aiter ar_rmsnorm (reduce-scatter + ar_rmsnorm kernels, next RMSNorm fused in)")
    else:
        t.append("- all-reduce kernels include the wait for the other ranks (no separate sync kernel)")
    on = sorted(o for o in R["onsets"] if o is not None)
    if on:
        t += ["- decode runs as HIP graph replay: the profiler drops some kernel timestamp records there, and from",
              "  the first drop on every kernel name in the step carries the next kernel's duration (layer sums",
              f"  unchanged; first misaligned layer p50 L{on[len(on) // 2]}, {R['onsets'].count(None)} of {nsteps} "
              "steps never misalign); per step, layers",
              f"  from the first misaligned one on (and the one before) are left out: {n_kept},",
              f"  {use} (sparse, common kernel sequence) of them"]
    else:
        t += ["- decode runs as HIP graph replay; no misaligned kernel timestamps found (each layer checked against",
              f"  the L0-L8 medians): {n_kept}, {use} (sparse) of them"]
    t += ["", "Profiler overhead (decode)"]
    line = (f"- a step takes {dst['wall'] / 1e3:.2f} ms on the GPU timeline: {dst['busy'] / 1e3:.2f} ms kernels + "
            f"{dst['idle'] / 1e3:.2f} ms idle;")
    place = dst["place"]
    if place:
        li = place[0]
        where = (f"L{li} is not the dense layer shown, so no layer_span in this tab contains it"
                 if cfg.variants()[li] == "full+mlp" and li != dec_dense else
                 "it is part of the dense layer_span shown" if li == dec_dense else
                 "layer_span is a median over the sparse layers, so it does not contain it")
        t += [f"{line} {dst['loc_gap'] / 1e3:.2f} ms of the",
              f"  idle is one gap {place[2]} ({dst['n_loc']} of {nsteps} steps);",
              f"  {where} (see the rows under all_layer_span)"]
    elif dst["gap"] >= 50:
        t.append(f"{line} the largest gap per step (p50 {dst['gap'] / 1e3:.2f} ms) has no fixed place")
    else:
        t.append(f"{line} spread over the kernel boundaries (largest single gap p50 {dst['gap']:.1f} us)")
    period = win["step_period_ms"]
    noprof = win["fixed_ref_step_ms"] if F else itl * tps
    wpd = win.get("window_per_decode_ms")
    if F:
        t += [f"- without the profiler a step is {noprof:.2f} ms (median gap between a stream's chunks over the "
              f"{F['ref_s']:g} s", f"  before the window, same {F['bs']} streams); with it the median decode step "
              f"period of the window is {period:.2f} ms,", f"  so the profiler adds ~{100 * (period / noprof - 1):.0f}%"]
    elif period < noprof and wpd:
        t += [f"- aiperf ITL {itl:.2f} ms x {tps:.2f} tokens/step = {noprof:.1f} ms is more than the median decode step "
              f"period ({period:.2f} ms): a request's", "  ITL also covers the prefills run between its decode steps; "
              f"the window spends {wpd:.2f} ms per decode step, prefills included,", f"  so the profiler adds "
              f"~{100 * (wpd / noprof - 1):.0f}%"]
    elif period < 1.2 * noprof:
        t += [f"- without the profiler a step is ~{noprof:.1f} ms (aiperf ITL {itl:.2f} ms x {tps:.2f} tokens/step); "
              "with it the median", f"  decode step period of the window (all shapes) is {period:.2f} ms, so the "
              f"profiler adds ~{100 * (period / noprof - 1):.0f}%"]
    elif noprof < dst["busy"] / 1e3:
        t += [f"- without the profiler a step is ~{noprof:.1f} ms (aiperf ITL {itl:.2f} ms x {tps:.2f} tokens/step), "
              "less than", "  the kernel busy alone, so decode kernel times read high too"]
    else:
        t.append(f"- without the profiler a step is ~{noprof:.1f} ms (aiperf ITL {itl:.2f} ms x {tps:.2f} tokens/step)")
    if A.profiler_ref:
        t.append(f"  ({A.profiler_ref})")
    pairs_m = (fp["sqsk"] - (fp["sqsq"] - fp["tok"]) / 2) / 1e6
    peak = A.hbm_tbs or HBM_TBS.get(head.get("ARCH"))
    moe_tbs = max((s[2] / d / 1e6 for kd in KINDS for mod, _, d, s in dec[kd][0]
                   if mod.startswith("moe_expert") and s and s[2]), default=0.0)
    t += ["",
          "GEMM columns (per rank, TP4)",
          "- MxKxN with M = tokens; MoE M = tokens x 5 routes (4 routed + 1 shared), 129 experts",
          *([f"  prefill M = the {int(pre_rows):,} rows of the token-graph bucket ({int(fp['tok']):,} new tokens + "
             f"{int(pre_rows - fp['tok']):,} padding); attention and indexer: the new tokens"] if "tok_run" in fp else []),
          qkv_line,
          "- o_proj 2048 -> 6144; dense MLP 6144 -> 6144 (gate+up), 3072 -> 6144; MoE 6144 -> 1536, 768 -> 6144",
          "- TFLOP/s = 2MKN / time; TB/s = (A + W + C bytes) / time, output bf16; mxfp4 = 0.5 B + 1 B scale per 32",
          f"- MoE W = experts touched under uniform routing (prefill {touched(cfg, pre_rows):.0f}, "
          f"decode ~{touched(cfg, fmed['tok']):.0f} of 129)",
          *([f"  decode MoE TB/s up to {moe_tbs:.2f} vs the {peak:g} TB/s HBM peak: real routing touches fewer "
             "experts, so these TB/s are upper bounds"] if peak and moe_tbs > peak else []),
          "- attention FLOPs = 4*h*d*pairs, h=16 d=128, pairs = causal query-key pairs from the annotation",
          f"  (prefill {pairs_m:.1f}M); sparse attention: 2048 keys per query;",
          f"  decode dense TB/s = K+V fp8 bytes of {bs} x ~{ctx_k}K tokens read once",
          *([f"- index score FLOPs = 2*pairs*128 per index head: prefill this rank's head, decode all "
             f"{cfg.idx_heads} heads over 1/{cfg.tp}", f"  of the pairs; decode TB/s = fp8 index-K bytes of 1/{cfg.tp} "
             f"of {bs} x ~{ctx_k}K tokens"] if cfg.indexer_cp else
            [f"- index score FLOPs = 2*pairs*128; decode TB/s = fp8 index-K bytes of {bs} x ~{ctx_k}K tokens"]),
          "- top-k / norm / quant / all-reduce" + (" / all-gather" if cfg.indexer_cp else "")
          + ": no FLOP/byte estimate; sparse attention: FLOPs only"]
    return "\n".join(t) + "\n"


FILL = {"attn_core": "FCE4D6", "gemm": "FFF2CC", "moe_expert": "E2EFDA", "all_reduce": "DDEBF7",
        "wait": "D9D9D9", "index": "EDE7F6"}


def fill_for(mod):
    if mod == "all_reduce_wait":
        return FILL["wait"]
    for p, k in (("attn_core", "attn_core"), ("moe_expert", "moe_expert"), ("all_reduce", "all_reduce"),
                 ("all_gather", "all_reduce"), ("sparse_index", "index")):
        if mod.startswith(p):
            return FILL[k]
    return FILL["gemm"] if "gemm" in mod else None


def write_sheet(ws, blocks, notes, title):
    """blocks = [(标题, block() 的结果, 额外的汇总行 [(标签, ms)])]。"""
    ws.column_dimensions["A"].width = 70
    for cols in ("BCDEFGHI", "JKLMNOPQ"):
        for col, w in zip(cols, (18, 70, 10, 8, 26, 12, 10, 8)):
            ws.column_dimensions[col].width = w
    bold = Font(bold=True)
    hdr_fill = PatternFill("solid", fgColor="BFBFBF")
    row0 = 3
    for head, data, extra in blocks:
        grand = []
        height = max(len(data[kd][0]) for kd in KINDS)
        for kd, cols in zip(KINDS, ("BCDEFGHI", "JKLMNOPQ")):
            rows, spn, nlay, _ = data[kd]
            L = lambda i: cols[i]
            first, last = row0 + 2, row0 + 1 + len(rows)
            tot = sum(r[2] for r in rows)
            ws[f"{L(0)}{row0}"] = (f"{head}  {KINDS[kd].upper()}  kernel sum={tot:.1f}us x{nlay} = {tot * nlay / 1e3:.2f}ms"
                                   f"  |  span={spn:.1f}us x{nlay} = {spn * nlay / 1e3:.2f}ms")
            ws[f"{L(0)}{row0}"].font = bold
            for j, h in enumerate(("module", "kernel", "time-us", "ratio", "GEMM / attn | MxKxN", "dtype", "TFLOP/s", "TB/s")):
                cell = ws[f"{L(j)}{row0 + 1}"]
                cell.value, cell.font, cell.fill = h, bold, hdr_fill
            lt = f"{L(2)}{last + 1}"
            for i, (mod, name, d, s) in enumerate(rows):
                r = first + i
                ws[f"{L(0)}{r}"], ws[f"{L(1)}{r}"], ws[f"{L(2)}{r}"] = mod, name, round(d, 3)
                ws[f"{L(3)}{r}"] = f"={L(2)}{r}/{L(2)}${last + 1}"
                ws[f"{L(3)}{r}"].number_format = "0.0%"
                if s:
                    label, flops, nbytes, dt = s
                    ws[f"{L(4)}{r}"], ws[f"{L(5)}{r}"] = label, dt
                    if flops:
                        ws[f"{L(6)}{r}"] = f"={flops:.6g}/({L(2)}{r}*1E6)"
                        ws[f"{L(6)}{r}"].number_format = "0.00"
                    if nbytes:
                        ws[f"{L(7)}{r}"] = f"={nbytes:.6g}/({L(2)}{r}*1E6)"
                        ws[f"{L(7)}{r}"].number_format = "0.000"
                fl = fill_for(mod)
                if fl:
                    for j in range(2):
                        ws[f"{L(j)}{r}"].fill = PatternFill("solid", fgColor=fl)
            ws[f"{L(1)}{last + 1}"], ws[lt] = "layer_time", f"=SUM({L(2)}{first}:{L(2)}{last})"
            ws[f"{L(3)}{last + 1}"] = "us"
            ws[f"{L(1)}{last + 2}"], ws[f"{L(2)}{last + 2}"] = "total_time", f"={lt}*{nlay}/1000"
            ws[f"{L(3)}{last + 2}"] = "ms"
            waits = [first + i for i, r in enumerate(rows) if r[0] == "all_reduce_wait"]
            ws[f"{L(1)}{last + 3}"] = "all_reduce_wait (in layer_time)"
            ws[f"{L(2)}{last + 3}"] = ("=" + "+".join(f"{L(2)}{w}" for w in waits)) if waits else "n/a (inside all_reduce)"
            ws[f"{L(3)}{last + 3}"] = "us"
            span = f"{L(2)}{last + 4}"
            ws[f"{L(1)}{last + 4}"], ws[span] = "layer_span (start to next layer start)", round(spn, 1)
            ws[f"{L(3)}{last + 4}"] = "us"
            ws[f"{L(1)}{last + 5}"], ws[f"{L(2)}{last + 5}"] = "gpu_idle = layer_span - layer_time", f"={span}-{lt}"
            ws[f"{L(3)}{last + 5}"] = "us"
            ws[f"{L(4)}{last + 5}"] = f"={L(2)}{last + 5}/{span}"
            ws[f"{L(4)}{last + 5}"].number_format = "0.0%"
            ws[f"{L(1)}{last + 6}"], ws[f"{L(2)}{last + 6}"] = "total_span", f"={span}*{nlay}/1000"
            ws[f"{L(3)}{last + 6}"] = "ms"
            for rr in (last + 1, last + 2, last + 4, last + 6):
                ws[f"{L(1)}{rr}"].font = ws[f"{L(2)}{rr}"].font = bold
            for rr in (last + 4, last + 5, last + 6):
                ws[f"{L(1)}{rr}"].fill = ws[f"{L(2)}{rr}"].fill = PatternFill("solid", fgColor="FFE699")
            grand.append((f"{L(2)}{last + 2}", f"{L(2)}{last + 6}"))
        end = row0 + 2 + height + 8
        ws[f"K{end}"], ws[f"L{end}"], ws[f"M{end}"] = "all_layer_time (kernel sum)", f"={grand[0][0]}+{grand[1][0]}", "ms"
        ws[f"K{end + 1}"], ws[f"L{end + 1}"], ws[f"M{end + 1}"] = "all_layer_span", f"={grand[0][1]}+{grand[1][1]}", "ms"
        for rr in (end, end + 1):
            ws[f"K{rr}"].font = ws[f"L{rr}"].font = bold
        for i, (label, ms) in enumerate(extra):
            ws[f"K{end + 2 + i}"], ws[f"L{end + 2 + i}"], ws[f"M{end + 2 + i}"] = label, round(ms, 2), "ms"
        row0 = end + 4 + len(extra)
    ws["A3"] = title
    ws["A3"].font = Font(bold=True, size=14)
    ws["A4"] = notes
    ws["A4"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(f"A4:A{max(40, ws.max_row)}")
    ws.freeze_panes = "B3"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the skill's OUT directory, after analyze.sh")
    ap.add_argument("--stack", required=True,
                    help='versions in the traced image, e.g. "11-04 stack: ATOM <rev>, aiter <rev>, FlyDSL <rev>"')
    ap.add_argument("--rank", type=int, default=0, help="the rank analyze.sh broke down (its rank A, default 0)")
    ap.add_argument("--config", help="model config (default: <run>/model_config.json, as in analyze.sh)")
    ap.add_argument("--peaks", default=None, help="peaks json (default: $M3_TRACE_PEAKS); hbm_tbs is used in the notes")
    ap.add_argument("--hw", help='default from server.log, e.g. "MI355X (gfx950), 4 GPUs (ROCR_VISIBLE_DEVICES=0,1,4,5)"')
    ap.add_argument("--tab", help="default <platform>-M3-TP4-<MMDD of the trace>")
    ap.add_argument("--title")
    ap.add_argument("--profiler-ref", default="", help="extra profiler-overhead reference line for the notes")
    ap.add_argument("--prefill", help="annotation prefix of the prefill forward (default: analyze.sh's representative)")
    ap.add_argument("--decode", help="decode shape key, e.g. 'decode[bs=4 tok=16 d=4 spec=3]' (default: analyze.sh's)")
    ap.add_argument("--match", help="tab whose shapes --prefill / --decode follow, e.g. MI455-M3-TP4-1103 (notes only); "
                                    "for a WORKLOAD=fixed run, the capture its prefill shape comes from, e.g. "
                                    "'the 11-05 AgentX c32 window'")
    ap.add_argument("--master", help="workbook to add the tab to (a tab of the same name is replaced in place); "
                                     "created when missing")
    ap.add_argument("--single", help="workbook with this tab only (default: <run>/breakdown/M3_TP4_layer_breakdown_<tab>.xlsx)")
    ap.add_argument("--check", action="store_true", help="print the tables, write nothing")
    A = ap.parse_args(argv)
    run = A.run.rstrip("/")
    trace = f"{run}/reduced/traces/rank_{A.rank}/reduced.pt.trace.json.gz"
    config = A.config or f"{run}/model_config.json"
    need = [trace, config, f"{run}/server.log", f"{run}/capture_meta.json"] + [
        f"{run}/breakdown/{n}" for n in ("prefill_meta.json", "decode_meta.json", "window_summary.json")]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        sys.exit(f"missing {', '.join(missing)}: run analyze.sh on {run} first")
    A.hbm_tbs = (mc.load_peaks(A.peaks or os.environ.get("M3_TRACE_PEAKS")) or {}).get("hbm_tbs")
    P = platform(run)
    win = json.load(open(f"{run}/breakdown/window_summary.json"))
    if P["workload"] != "fixed" and win.get("aiperf_itl_finished_p50_ms") is None:
        sys.exit(f"{run}/breakdown/window_summary.json has no aiperf ITL: is {run}/aiperf there?")
    gpus = P["env"].get("ROCR_VISIBLE_DEVICES")
    A.hw = A.hw or f"{P['name']} ({P['head'].get('ARCH', '?')}), 4 GPUs" + (f" (ROCR_VISIBLE_DEVICES={gpus})" if gpus else "")
    A.tab = A.tab or f"{P['name']}-M3-TP4-{P['date'].replace('-', '')}"
    A.title = A.title or (f"{P['name']} MiniMax-M3 TP4 breakdown ({A.stack.split(' stack')[0]} stack, "
                          + (f"fixed-shape bs={P['fixed']['bs']} trace of {P['date']})" if P["workload"] == "fixed"
                             else f"in-run AgentX c{P['conc']} trace of {P['date']})"))
    A.skill_pre = json.load(open(f"{run}/breakdown/prefill_meta.json"))["annotation"].split(" sqsq")[0]
    A.skill_dec = json.load(open(f"{run}/breakdown/decode_meta.json"))
    prefill = A.prefill or A.skill_pre
    decode = A.decode or A.skill_dec["key"]

    R = collect(trace, config, prefill, decode, indexer_cp=P["env"].get("ATOM_M3_INDEXER_CP") == "1")
    cfg, pre, dec, kept, pst, dst = (R[k] for k in ("cfg", "pre", "dec", "kept", "pst", "dst"))
    check(dec, kept, cfg)
    for name, data in (("prefill", pre), ("decode", dec)):
        for kd in KINDS:
            rows, spn, nlay, same = data[kd]
            print(f"\n{name:7s} {kd:10s} kernels {len(rows):2d}  sum {sum(r[2] for r in rows):8.1f} us  span {spn:8.1f} us"
                  f"  x{nlay}  ({len(same)} layer instances, layers {sorted({x['li'] for x in same})[:6]}...)")
            for mod, kn, d, s in rows:
                perf = ""
                if s:
                    perf = f"{s[0]:>18s} {s[3]:>11s} {s[1] / d / 1e6:8.1f} TF/s" + (f" {s[2] / d / 1e6:6.2f} TB/s" if s[2] else "")
                print(f"  {mod:22s} {kn[:58]:58s} {d:8.2f}  {perf}")
    if A.check:
        return 0

    notes = notes_text(R, A, P, win)
    gap_row = (f"of the idle: {dst['place'][1]} gap p50", dst["loc_gap"] / 1e3) if dst["place"] else \
        ("largest single gap per step p50", dst["gap"] / 1e3)
    if P["workload"] == "fixed":
        noprof, noprof_label = win["fixed_ref_step_ms"], "step without profiler (client, just before the window)"
    else:
        noprof = win["aiperf_itl_finished_p50_ms"] * win["toks_per_fwd"]
        noprof_label = ("aiperf ITL x tokens/step (includes the prefills between steps)"
                        if win["step_period_ms"] < noprof else "step without profiler (aiperf ITL x tokens/step)")
    blocks = [(R["heads"][0], pre, [("forward wall (GPU timeline)", pst["wall"] / 1e3),
                                    ("forward kernel busy (main stream, incl. embedding / lm_head)", pst["busy"] / 1e3)]),
              (R["heads"][1], dec, [("step wall p50 (GPU timeline, profiler on)", dst["wall"] / 1e3),
                                    ("step kernel busy p50 (main stream)", dst["busy"] / 1e3), gap_row,
                                    (noprof_label, noprof)])]
    if A.master:
        if os.path.exists(A.master):
            wb = load_workbook(A.master)
        else:
            wb = Workbook()
            wb.remove(wb.active)
        at = wb.sheetnames.index(A.tab) if A.tab in wb.sheetnames else None
        if at is not None:
            del wb[A.tab]
        write_sheet(wb.create_sheet(A.tab, at), blocks, notes, A.title)
        wb.save(A.master)
        print(A.master)
    one = Workbook()
    one.remove(one.active)
    write_sheet(one.create_sheet(A.tab), blocks, notes, A.title)
    single = A.single or f"{run}/breakdown/M3_TP4_layer_breakdown_{A.tab}.xlsx"
    one.save(single)
    print(single)
    return 0


if __name__ == "__main__":
    sys.exit(main())
