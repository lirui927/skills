#!/usr/bin/env python3
"""MiniMax-M3 decode kernel breakdown（ATOM，TP，HIP graph 回放的 decode）。

方法照 trace-kernel-breakdown 的 graph decode 做法（见 NOTICE）移植到 M3：
  - 选一类 decode step：annotation 去掉每步都变的 ctx/sqsq/sqsk/sk 之后的形状键（bs/tok/d/spec），
    默认取窗口里最多的那一类，把这一类的所有 step 合起来选代表层；
  - 每个 step 的主 stream 按 attention core 切层（同 prefill），逐层类型必须和 config 一致；
  - 配对检查：graph 回放时 kineto 可能把 kernel 名字和时长配错。同名 kernel 的 p90/p50 中位数 > 2 判为
    SCRAMBLED，这时逐 kernel 的数字不可信，只能用窗口总量、step 的 union busy、每层的 union busy；
  - 每层时间 = 该层时间跨度内所有 stream 的 union busy；代表层 = 只含"同类层共有模块"的层里，
    union busy 的中位数（所有 step 合并取）；
  - 行：主 stream 按时间序，再接 side stream（overlap=yes）；每行只记它没被前面的行覆盖的部分，
    所以各行加起来正好等于这一层的 union busy；
  - GEMM 的 K/N 来自 config@TP（graph 回放不录 shape），M = annotation 的 tok；trace 里 prefill 录了 shape 时，
    逐模块对比 K/N；attention core 的 FLOPs 用这个 step 的 annotation（sqsq/sqsk/sk）推导。
用法: m3_decode.py <reduced.pt.trace.json.gz> <config.json> <out_dir> [--tp 4] [--indexer-cp]
                   [--annotation PREFIX] [--peaks peaks.json] [--ui-zero-us TS]
写出 decode_detail.csv、decode_wide.csv、decode_module_summary.csv、decode_comm.csv、decode_extras.csv、
decode_rep_timeline.csv（代表层的每个 kernel，所有 stream 按时间序，带原始 ts/end/dur）、decode_picks.txt。
"""
import argparse
import collections
import csv
import json
import os
import statistics
import sys

import m3_common as mc
from m3_prefill import WF, WK, comm_calls, common_signature, extras_of


def spread_test(kernels, min_n=20):
    """同名 kernel 的 p90/p50；返回 (中位数, 参与的名字数)。正确配对时同名时长很紧（~1.0-1.3）。"""
    by = collections.defaultdict(list)
    for k in kernels:
        by[k["name"]].append(k["dur"])
    r = sorted(mc.pct(v, 0.9) / max(1e-9, mc.pct(v, 0.5)) for v in by.values() if len(v) >= min_n)
    return (r[len(r) // 2] if r else float("nan")), len(r)


def critical_rows(main_ks, side_ks):
    """主 stream 先按时间序，再 side stream；每行记 (kernel, 关键路径时长, 被覆盖时长)。"""
    out, iv = [], []
    for k in list(main_ks) + list(side_ks):
        cov = mc.overlap_us(k, mc_merge(iv))
        out.append((k, k["dur"] - cov, cov))
        iv.append((k["ts"], k["ts"] + k["dur"]))
    return out


def mc_merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--indexer-cp", action="store_true")
    ap.add_argument("--annotation", default=None,
                    help="decode shape key or name prefix, e.g. 'decode[bs=3 tok=12'; default: the most frequent key")
    ap.add_argument("--peaks", default=None)
    ap.add_argument("--ui-zero-us", type=float, default=None)
    args = ap.parse_args(argv)
    cfg = mc.M3Config(args.config, tp=args.tp, indexer_cp=args.indexer_cp)
    T = mc.Trace(args.trace, args.ui_zero_us)
    peaks = mc.load_peaks(args.peaks or os.environ.get("M3_TRACE_PEAKS"))
    os.makedirs(args.out, exist_ok=True)
    VAR, NV = cfg.variants(), cfg.variant_counts()

    dec = T.forwards("decode")
    if not dec:
        sys.exit("no decode[...] annotation on the main stream")
    keys = collections.Counter(mc.ann_key(a["name"]) for a in dec)
    if args.annotation:
        key = next((k for k in keys if k.startswith(args.annotation)), None) or next(
            (mc.ann_key(a["name"]) for a in dec if a["name"].startswith(args.annotation)), None)
        if key is None:
            sys.exit(f"no decode annotation matches {args.annotation!r}; keys: {keys.most_common(8)}")
    else:
        key = keys.most_common(1)[0][0]
    steps_all = [a for a in dec if mc.ann_key(a["name"]) == key]
    print(f"decode forwards in window: {len(dec)}; shape keys: {keys.most_common(6)}")
    print(f"chosen key: {key}  ({len(steps_all)} steps)")

    steps = []
    for a in steps_all:
        w = T.main_window(a)
        seg, why = mc.segment(w, cfg.nl)
        s = {"ann": a, "w": w, "why": why, "ok": False}
        if seg and [mc.observed_variant(x) for x in seg["layers"]] == VAR:
            s.update(ok=True, seg=seg)
            s["head"], s["rows"], s["tail"] = mc.classify_forward(seg, cfg)
            # 每层的时间跨度：本层第一个 kernel 到下一层第一个 kernel（最后一层到 tail 开头）
            starts = [L[0]["ts"] for L in seg["layers"]] + [seg["tail"][0]["ts"] if seg["tail"] else a["ts"] + a["dur"]]
            s["layers"] = []
            for i, L in enumerate(seg["layers"]):
                side = T.side_window(starts[i], starts[i + 1])
                s["layers"].append({"li": i, "t0": starts[i], "t1": starts[i + 1], "main": L, "side": side,
                                    "union": mc.union([(k["ts"], k["ts"] + k["dur"]) for k in L + side])})
            sw = T.side_window(a["ts"], a["ts"] + a["dur"])
            s["union"] = mc.union([(k["ts"], k["ts"] + k["dur"]) for k in w + sw])
            s["busy"] = sum(k["dur"] for k in w)
        steps.append(s)
    ok = [s for s in steps if s["ok"]]
    print(f"segmented steps: {len(ok)} of {len(steps)}" +
          ("" if len(ok) == len(steps) else f"  (first failure: {next(s['why'] for s in steps if not s['ok'])})"))
    if not ok:
        sys.exit("no decode step segments into the config's layers: check the core / collective patterns in "
                 "m3_common.py against this image's kernel names; until then only the window totals (m3_window.py) hold")

    # 配对检查（graph 回放的名字-时长配对）
    ratio, nn = spread_test([k for s in ok for k in s["w"]])
    pre = [k for a in T.forwards("prefill") for k in T.main_window(a)]
    pratio, pn = spread_test(pre, 10) if pre else (float("nan"), 0)
    SCRAMBLED = ratio == ratio and ratio > 2.0
    print(f" same-name duration spread: median p90/p50 {ratio:.2f} over {nn} kernel names"
          f"{f' (eager prefill in this trace: {pratio:.2f} over {pn})' if pn else ''}"
          + ("  -> SCRAMBLED name/duration pairing: per-kernel rows below are not trustworthy, use window and per-layer "
             "union totals" if SCRAMBLED else "  -> pairing looks intact"))
    if not peaks:
        print("peaks not given (--peaks / M3_TRACE_PEAKS): the over-peak guard is off")
    PV = "pairing unverified (SCRAMBLED); " if SCRAMBLED else ""

    # 每种层的代表层
    PICKS = [f"decode key: {key}  ({len(ok)} steps pooled; UI zero {T.ui_zero:.3f} us: {T.ui_src}; file {T.path}, "
             f"main stream {T.main_tid})"]
    detail, timeline, extras = [], [], []
    rep = {}
    for v in NV:
        idx = [i for i, x in enumerate(VAR) if x == v]
        common = common_signature(ok[0]["rows"], idx)
        pool = [(s, s["layers"][i]) for s in ok for i in idx
                if collections.Counter(r["m"] for r in s["rows"][i]) == common]
        allv = [s["layers"][i]["union"] for s in ok for i in idx]
        if not pool:
            pool = [(s, s["layers"][i]) for s in ok for i in idx]
        pool.sort(key=lambda x: x[1]["union"])
        s, L = pool[len(pool) // 2]
        rep[v] = (s, L)
        li = L["li"]
        print(f"  {v:11s} representative layer #{li:2d} union busy {L['union']:8.1f} us x{NV[v]:2d}  "
              f"(eligible {len(pool)} of {len(allv)} layer instances; all p50 {statistics.median(allv):.1f} us, "
              f"p10 {mc.pct(allv, 0.1):.1f}, p90 {mc.pct(allv, 0.9):.1f})")
        for i in idx:
            for r in extras_of(ok[0]["rows"][i], common):
                extras.append({"where": f"layer {i} ({v})", "layer_idx": i, "module": r["m"],
                               "kernel": mc.short(r["k"]["name"]), "time_us": round(r["k"]["dur"], 2),
                               "kernel_ts_us": round(r["k"]["ts"], 3), "kernel_ui": T.ui(r["k"]["ts"])})
        a = s["ann"]
        PICKS += [f"  {v:11s} step {a['name']}",
                  f"        annotation ts {a['ts']:.3f} us, dur {a['dur'] / 1e3:.3f} ms, UI {T.ui(a['ts'])}",
                  f"        layer {li}: ts {L['t0']:.3f} .. {L['t1']:.3f} us, annotation +{L['t0'] - a['ts']:.1f} .. "
                  f"+{L['t1'] - a['ts']:.1f} us, UI {T.ui(L['t0'])} .. {T.ui(L['t1'])}"]
        modrow = {id(r["k"]): r for r in s["rows"][li]}
        base = {"phase": "decode", "annotation": a["name"], "ann_ts_us": round(a["ts"], 3), "variant": v,
                "layer_idx": li, "layer_t0_us": round(L["t0"], 3), "layer_t1_us": round(L["t1"], 3),
                "ann_ui": T.ui(a["ts"]), "layer_ui": f"{T.ui(L['t0'])} .. {T.ui(L['t1'])}", "layer_count": NV[v]}
        rows = critical_rows(L["main"], L["side"])
        tot = 0.0
        for k, crit, cov in rows:
            side = k["tid"] != T.main_tid
            r = modrow.get(id(k), {"k": k, "m": "side_stream", "role": ""})
            mt = {} if side else mc.row_metrics(r, a["name"], cfg, T, peaks)
            tot += crit
            detail.append({**base, "kernel_ts_us": round(k["ts"], 3), "kernel_ui": T.ui(k["ts"]),
                           "group": mc.group(r["m"]), "module": r["m"], "kernel": mc.short(k["name"]),
                           "stream": k["tid"], "overlap": "yes" if side else "",
                           "overlapped_us": round(cov, 2) if side else "", "time_us": round(k["dur"], 2),
                           "critical_us": round(crit, 2), "ratio": "",
                           "m": mt.get("m", ""), "k": mt.get("k", ""), "n": mt.get("n", ""), "dtype": mt.get("dtype", ""),
                           "tflops": "" if SCRAMBLED else mt.get("tflops", ""),
                           "tbytes_s": "" if SCRAMBLED else mt.get("tbytes_s", ""),
                           "note": PV + mt.get("src", "")})
        for x in detail[-len(rows):]:
            x["ratio"] = f"{100 * x['critical_us'] / L['union']:.2f}%"
        assert abs(tot - L["union"]) < 0.5, (v, tot, L["union"])
        for k in sorted(L["main"] + L["side"], key=lambda k: k["ts"]):
            other = L["side"] if k["tid"] == T.main_tid else L["main"]
            timeline.append({"variant": v, "layer_idx": li, "stream": k["tid"],
                             "module": modrow.get(id(k), {"m": "side_stream"})["m"], "kernel": mc.short(k["name"]),
                             "ts_us": round(k["ts"], 3), "end_us": round(k["ts"] + k["dur"], 3),
                             "dur_us": round(k["dur"], 3), "ui": T.ui(k["ts"]),
                             "overlap_other_stream_us": round(mc.overlap_us(k, [(o["ts"], o["ts"] + o["dur"]) for o in other]), 3)})
    s0 = ok[len(ok) // 2]
    for where, rows in (("forward head", s0["head"]), ("forward tail", s0["tail"])):
        for r in rows:
            extras.append({"where": where, "layer_idx": "", "module": r["m"], "kernel": mc.short(r["k"]["name"]),
                           "time_us": round(r["k"]["dur"], 2), "kernel_ts_us": round(r["k"]["ts"], 3),
                           "kernel_ui": T.ui(r["k"]["ts"])})
    fields = ["phase", "annotation", "ann_ts_us", "variant", "layer_idx", "layer_t0_us", "layer_t1_us", "ann_ui",
              "layer_ui", "layer_count", "kernel_ts_us", "kernel_ui", "group", "module", "kernel", "stream", "overlap",
              "overlapped_us", "time_us", "critical_us", "ratio", "m", "k", "n", "dtype", "tflops", "tbytes_s", "note"]
    mc.write_csv(f"{args.out}/decode_detail.csv", detail, fields)
    mc.write_csv(f"{args.out}/decode_rep_timeline.csv", timeline)
    mc.write_csv(f"{args.out}/decode_extras.csv", extras,
                 ["where", "layer_idx", "module", "kernel", "time_us", "kernel_ts_us", "kernel_ui"])

    with open(f"{args.out}/decode_wide.csv", "w", newline="") as f:
        wr = csv.writer(f)
        cols = {v: [r for r in detail if r["variant"] == v] for v in NV}
        hdr = []
        for v in NV:
            c0, u, n = cols[v][0], rep[v][1]["union"], NV[v]
            side = sum(r["time_us"] for r in cols[v] if r["overlap"])
            hdr += [f"{c0['annotation']} @UI {c0['ann_ui']}  {v} layer#{c0['layer_idx']} [UI {c0['layer_ui']}]  "
                    f"union busy = {u:.1f}us x{n} = {u * n / 1000:.2f}ms  (side stream {side:.1f}us"
                    f"{', pairing unverified' if SCRAMBLED else ''})"] + [""] * len(WF)
        wr.writerow(hdr)
        wr.writerow((WF + [""]) * len(NV))
        wk = [x if x != "time_us" else "critical_us" for x in WK]
        for i in range(max(len(c) for c in cols.values())):
            line = []
            for v in NV:
                r = cols[v][i] if i < len(cols[v]) else None
                line += ([r[x] for x in wk] if r else [""] * len(WF)) + [""]
            wr.writerow(line)

    # 每个模块：每个 step 全部层加总（主 stream），取各 step 的中位数
    def mt_step(s):
        t = collections.defaultdict(float)
        for r in s["head"] + [x for L in s["rows"] for x in L] + s["tail"]:
            t[r["m"]] += r["k"]["dur"]
        return t
    allt = [mt_step(s) for s in ok]
    U = statistics.median(s["union"] for s in ok)
    B = statistics.median(s["busy"] for s in ok)
    W = statistics.median(s["ann"]["dur"] for s in ok)
    mods = sorted(set().union(*allt), key=lambda m: -statistics.median(t.get(m, 0) for t in allt))
    summ = [{"group": mc.group(m), "module": m,
             "step_median_us": round(statistics.median(t.get(m, 0) for t in allt), 2),
             "step_min_us": round(min(t.get(m, 0) for t in allt), 2),
             "step_max_us": round(max(t.get(m, 0) for t in allt), 2),
             "share_of_union": f"{100 * statistics.median(t.get(m, 0) for t in allt) / U:.2f}%"} for m in mods]
    mc.write_csv(f"{args.out}/decode_module_summary.csv", summ)
    gt = collections.defaultdict(list)
    for t in allt:
        g = collections.defaultdict(float)
        for m, x in t.items():
            g[mc.group(m)] += x
        for k_, x in g.items():
            gt[k_].append(x)
    print(f"\n group totals per step (median over {len(ok)} steps; step main-stream busy p50 {B / 1e3:.3f} ms, union "
          f"{U / 1e3:.3f} ms, wall {W / 1e3:.3f} ms){' -- pairing unverified' if SCRAMBLED else ''}:")
    for g_, xs in sorted(gt.items(), key=lambda x: -statistics.median(x[1])):
        print(f"   {g_:26s} {statistics.median(xs):9.1f} us  {100 * statistics.median(xs) / B:5.1f}% of busy")

    calls = collections.defaultdict(list)
    for s in ok:
        for rows in [s["head"]] + s["rows"] + [s["tail"]]:
            for m, d, _ in comm_calls(rows):
                calls[m].append(d)
    comm_rows = []
    print("\n communication per step (floor = fastest same-kind call over all chosen steps):")
    for m in sorted(calls):
        per = len(calls[m]) / len(ok)
        fl, med = min(calls[m]), statistics.median(calls[m])
        tot = sum(calls[m]) / len(ok)
        comm_rows.append({"module": m, "calls_per_step": round(per, 2), "step_total_us": round(tot, 1),
                          "floor_us": round(fl, 2), "p50_us": round(med, 2), "p90_us": round(mc.pct(calls[m], 0.9), 2),
                          "excess_over_floor_us": round(tot - fl * per, 1)})
        print(f"   {m:20s} calls/step={per:5.1f} total={tot:8.1f} us  floor={fl:6.2f} us  p50={med:6.2f} us  "
              f"p90={mc.pct(calls[m], 0.9):7.2f} us  excess={tot - fl * per:7.1f} us")
    mc.write_csv(f"{args.out}/decode_comm.csv", comm_rows)

    model = sum(NV[v] * rep[v][1]["union"] for v in NV)
    ht = sum(r["k"]["dur"] for r in s0["head"] + s0["tail"])
    ext = sum(e["time_us"] for e in extras if e["where"].startswith("layer"))
    print(f"\n reconcile: representative-layer model {model / 1e3:.3f} ms + head/tail {ht / 1e3:.3f} ms + minority-layer "
          f"kernels {ext / 1e3:.3f} ms = {(model + ht + ext) / 1e3:.3f} ms  vs  step union p50 {U / 1e3:.3f} ms "
          f"(wall p50 {W / 1e3:.3f} ms; the wall-union gap is idle inside the graph replay)")
    mins = collections.defaultdict(lambda: [0, 0.0, []])
    for e in extras:
        if e["where"].startswith("layer"):
            x = mins[(e["module"], e["kernel"][:50])]
            x[0] += 1
            x[1] += e["time_us"]
            x[2].append(e["layer_idx"])
    for (m, k), (n, t, ls) in sorted(mins.items(), key=lambda x: -x[1][1]):
        print(f"   minority kernel: {m:24s} {k:50s} x{n} at layers {ls}  {t:.1f} us")
    if T.has_shapes():
        # graph 回放不录 shape：用同一 trace 里 prefill 录到的 K/N 逐模块核对 config 推导
        recs = collections.defaultdict(set)
        for a in T.forwards("prefill"):
            w = T.main_window(a)
            seg, _ = mc.segment(w, cfg.nl)
            if not seg:
                continue
            h, rows, t = mc.classify_forward(seg, cfg)
            for r in h + [x for L in rows for x in L] + t:
                if r["role"] == "gemm":
                    rr = mc.recorded_mkn(T, r["k"])
                    if rr:
                        recs[r["m"]].add((rr[1], rr[2]))
        kn = cfg.gemm_kn()
        for m, s_ in sorted(recs.items()):
            exp = kn.get(m, (None, None))[:2]
            print(f"   K,N check vs prefill-recorded dims: {m:14s} recorded {sorted(s_)} config {exp} "
                  f"{'ok' if s_ == {exp} else 'MISMATCH -> fix M3Config.gemm_kn or the module map'}")
    print("\n where to look in the trace:\n" + "\n".join(PICKS))
    open(f"{args.out}/decode_picks.txt", "w").write("\n".join(PICKS) + "\n")
    UNC = ("misc_small", "gemm_unmapped", "act_quant", "head_misc", "tail_misc", "comm_other", "attn_out_misc")
    ut = statistics.median(sum(v for m, v in t.items() if m in UNC) for t in allt)
    json.dump({"key": key, "steps": len(ok), "spread": ratio, "scrambled": SCRAMBLED, "union_ms": U / 1e3,
               "busy_ms": B / 1e3, "wall_ms": W / 1e3, "model_ms": (model + ht + ext) / 1e3,
               "unclassified_share": ut / B, "rep_layers": {v: rep[v][1]["li"] for v in NV},
               "module_set": sorted(set().union(*allt))}, open(f"{args.out}/decode_meta.json", "w"), indent=1)
    print(f"\n wrote {args.out}/decode_detail.csv ({len(detail)} rows), decode_wide.csv, decode_module_summary.csv "
          f"({len(summ)}), decode_comm.csv, decode_extras.csv ({len(extras)}), decode_rep_timeline.csv "
          f"({len(timeline)}), decode_picks.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
