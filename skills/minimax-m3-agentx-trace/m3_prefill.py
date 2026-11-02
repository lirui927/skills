#!/usr/bin/env python3
"""MiniMax-M3 prefill kernel breakdown（ATOM，TP，eager prefill）。

方法照 trace-kernel-breakdown 的 TP prefill 做法（见 NOTICE）移植到 M3：
  - Memcpy HtoD 丢掉（挂在主 stream 上，实际在 DMA 引擎上和计算并行）；
  - 每层恰好一个 attention core，层边界在上一层 MLP/MoE 输出的 all-reduce 之后；每个 forward 的逐层类型
    必须和 config 一致（MiniMax-M3：3 层 full+mlp + 57 层 sparse+moe），否则这个 forward 不用；
  - 每个 forward 的模块总时间按全部层加总（精确，和主 stream busy 对得上）；每种层取中位层逐 kernel 输出；
  - 只在少数层出现的 kernel（第一个稀疏层的 plan、EAGLE3 aux 层的 clone、第 0 层自己的 input norm）
    以"同类层共有模块"为基准算出来，单独写进 prefill_extras.csv，不会因为取中位层而丢掉；
  - side stream 上的 kernel 按层列出，只把没被主 stream 覆盖的部分算作关键路径；
  - 通信每次调用 = 一组连续的通信 kernel（拷贝 + 同步 + 通信本体），拆成 floor（所有 steady forward 里
    同种调用的最快一次）和超出 floor 的部分（等待其他 rank 或争用；要分清得看第二个 rank 的 trace）；
  - GEMM 的 M/K/N：录了 shape 就用录到的，否则 K/N 来自 config@TP、M = annotation 的 tok（标 derived）。
用法: m3_prefill.py <reduced.pt.trace.json.gz> <config.json> <out_dir> [--tp 4] [--indexer-cp]
                    [--min-tok 0] [--min-busy 0] [--annotation PREFIX] [--peaks peaks.json] [--ui-zero-us TS]
写出 prefill_detail.csv、prefill_wide.csv、prefill_module_summary.csv、prefill_comm.csv、prefill_extras.csv、
prefill_picks.txt（选中的 annotation 和代表层在 trace 里的位置，Perfetto UI 时间）。
"""
import argparse
import collections
import csv
import json
import os
import statistics
import sys

import m3_common as mc

WF = ["module", "kernel", "UI", "stream", "overlap", "time_us", "overlapped_us", "ratio", "M", "K", "N", "dtype",
      "TFLOP/s", "TB/s", "note"]
WK = ["module", "kernel", "kernel_ui", "stream", "overlap", "time_us", "overlapped_us", "ratio", "m", "k", "n", "dtype",
      "tflops", "tbytes_s", "note"]


def layer_time(rows):
    return sum(r["k"]["dur"] for r in rows)


def common_signature(rows_by_layer, idx):
    """同类层共有的模块（多重集交集）：代表层从"只有这些模块"的层里选，其余层多出来的就是少数层 kernel。"""
    common = None
    for i in idx:
        c = collections.Counter(r["m"] for r in rows_by_layer[i])
        common = c if common is None else (common & c)
    return common or collections.Counter()


def extras_of(rows, common):
    """一层里超出共有模块的那些行（同一模块多出 s 个时取最后 s 个）。"""
    surplus = collections.Counter(r["m"] for r in rows) - common
    out = []
    for r in reversed(rows):
        if surplus[r["m"]] > 0:
            out.append(r)
            surplus[r["m"]] -= 1
    return list(reversed(out))


def comm_calls(rows):
    """一串行里的通信调用：连续的同模块 comm_* 行算一次调用，返回 [(module, 总时长, 其中同步 kernel 时长)]。"""
    calls, cur = [], None
    for r in rows:
        m = r["m"]
        if m.startswith("comm_"):
            sync = r["k"]["dur"] if "sync_kernel" in r["k"]["name"] else 0.0
            if cur and cur[0] == m:
                cur[1] += r["k"]["dur"]
                cur[2] += sync
            else:
                cur = [m, r["k"]["dur"], sync]
                calls.append(cur)
        else:
            cur = None
    return [tuple(c) for c in calls]


def analyse_forward(T, cfg, a):
    w = T.main_window(a)
    rec = {"ann": a, "w": w, "busy": sum(k["dur"] for k in w), "ok": False,
           "tok": mc.ann_fields(a["name"]).get("tok", 0)}
    seg, why = mc.segment(w, cfg.nl)
    rec["why"] = why
    if not seg:
        return rec
    obs = [mc.observed_variant(s) for s in seg["layers"]]
    if obs != cfg.variants():
        bad = next(i for i, (x, y) in enumerate(zip(obs, cfg.variants())) if x != y)
        rec["why"] = f"layer {bad} looks {obs[bad]}, config says {cfg.variant(bad)}"
        return rec
    rec["seg"] = seg
    rec["head"], rec["rows"], rec["tail"] = mc.classify_forward(seg, cfg)
    rec["ok"] = True
    return rec


def mod_totals(rec):
    t = collections.defaultdict(float)
    for r in rec["head"] + [x for L in rec["rows"] for x in L] + rec["tail"]:
        t[r["m"]] += r["k"]["dur"]
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--indexer-cp", action="store_true",
                    help="server ran with ATOM_M3_INDEXER_CP=1 (qkv_idx_proj carries every index head)")
    ap.add_argument("--min-tok", type=int, default=0, help="steady forwards: tok at least this")
    ap.add_argument("--min-busy", type=float, default=0.0,
                    help="steady forwards: main-stream busy / wall at least this (AgentX chunks can be half idle)")
    ap.add_argument("--annotation", default=None, help="use the prefill whose name starts with this")
    ap.add_argument("--peaks", default=None, help='json {"bf16":..,"fp8":..,"fp4":..,"hbm_tbs":..}; env M3_TRACE_PEAKS')
    ap.add_argument("--ui-zero-us", type=float, default=None,
                    help="ts the trace UI counts from (the original trace's 'Trace' event); default: that event in "
                         "this file, else its earliest event")
    args = ap.parse_args(argv)
    cfg = mc.M3Config(args.config, tp=args.tp, indexer_cp=args.indexer_cp)
    T = mc.Trace(args.trace, args.ui_zero_us)
    peaks = mc.load_peaks(args.peaks or os.environ.get("M3_TRACE_PEAKS"))
    os.makedirs(args.out, exist_ok=True)
    VAR = cfg.variants()
    NV = cfg.variant_counts()

    fw = [analyse_forward(T, cfg, a) for a in T.forwards("prefill")]
    if not fw:
        sys.exit("no prefill[...] annotation on the main stream")
    ok = [r for r in fw if r["ok"]]
    for r in fw:
        if not r["ok"]:
            print(f"  not segmented: {r['ann']['name'][:80]}  ({r['why']})")
    if args.annotation:
        steady = [r for r in ok if r["ann"]["name"].startswith(args.annotation)]
    else:
        steady = [r for r in ok if r["tok"] >= args.min_tok and r["busy"] / r["ann"]["dur"] >= args.min_busy]
    print(f"prefill forwards: {len(fw)}; segmented: {len(ok)}; steady (tok>={args.min_tok}, "
          f"busy>={100 * args.min_busy:.0f}%{', name ' + args.annotation if args.annotation else ''}): {len(steady)}")
    print(f"layer check: every segmented forward = {dict(NV)} by layer index  [config]")
    if not steady:
        sys.exit("no steady prefill forward; lower --min-tok / --min-busy or pick one with --annotation")
    if not peaks:
        print("peaks not given (--peaks / M3_TRACE_PEAKS): the over-peak guard on derived TFLOP/s and TB/s is off")
    print(f"shapes recorded in this trace: {'yes' if T.has_shapes() else 'no -> every M/K/N/dtype below is derived'}")

    rep = sorted(steady, key=lambda r: r["ann"]["dur"])[len(steady) // 2]
    A = rep["ann"]
    print(f"\nrepresentative prefill: {A['name']}\n  wall={A['dur'] / 1e3:.2f} ms  main-stream busy="
          f"{rep['busy'] / 1e3:.2f} ms ({100 * rep['busy'] / A['dur']:.1f}% of wall)")
    if rep["busy"] / A["dur"] < 0.8:
        print("  busy < 80% of wall: the main stream idles inside this forward (launch gaps, a cold JIT on a new batch "
              "bucket, or waiting on another rank), so the per-kernel numbers are kernel costs, not this forward's cost; "
              "m3_rank_skew.py tells waiting from transfer")

    PICKS = [f"prefill annotation: {A['name']}",
             f"  ts {A['ts']:.3f} us  dur {A['dur'] / 1e3:.3f} ms  UI {T.ui(A['ts'])}  (UI zero {T.ui_zero:.3f} us: "
             f"{T.ui_src}; file {T.path}, stream {T.main_tid})"]
    detail, extras = [], []
    rep_li = {}
    for v in NV:
        idx = [i for i, x in enumerate(VAR) if x == v]
        common = common_signature(rep["rows"], idx)
        cand = [i for i in idx if collections.Counter(r["m"] for r in rep["rows"][i]) == common] or idx
        cand.sort(key=lambda i: layer_time(rep["rows"][i]))
        li = cand[len(cand) // 2]
        rep_li[v] = li
        rows = rep["rows"][li]
        tot = layer_time(rows)
        lt = [layer_time(rep["rows"][i]) for i in idx]
        cv = statistics.pstdev(lt) / statistics.mean(lt) * 100 if len(lt) > 1 else 0.0
        nc = [sum(r["k"]["dur"] for r in rep["rows"][i] if not r["m"].startswith("comm_")) for i in idx]
        cv2 = statistics.pstdev(nc) / statistics.mean(nc) * 100 if len(nc) > 1 else 0.0
        print(f"  {v:11s} median layer #{li:2d} = {tot:9.1f} us  x{NV[v]:2d}   layer-time CV {cv:5.1f}%  "
              f"(excluding comm: {cv2:4.1f}%)   chosen among {len(cand)} layers with only the common modules")
        for i in idx:
            for r in extras_of(rep["rows"][i], common):
                extras.append({"where": f"layer {i} ({v})", "layer_idx": i, "module": r["m"],
                               "kernel": mc.short(r["k"]["name"]), "time_us": round(r["k"]["dur"], 2),
                               "kernel_ts_us": round(r["k"]["ts"], 3), "kernel_ui": T.ui(r["k"]["ts"])})
        seg = [r["k"] for r in rows]
        t0, t1 = seg[0]["ts"], max(k["ts"] + k["dur"] for k in seg)
        nxt = rep["seg"]["layers"][li + 1][0]["ts"] if li + 1 < cfg.nl else (
            rep["seg"]["tail"][0]["ts"] if rep["seg"]["tail"] else t1)
        core = next(k for k in seg if mc.CORE.search(k["name"]))
        PICKS += [f"  {v:11s} layer {li}: ts {t0:.3f} .. {t1:.3f} us ({t1 - t0:.1f} us span), annotation "
                  f"+{t0 - A['ts']:.1f} .. +{t1 - A['ts']:.1f} us, UI {T.ui(t0)} .. {T.ui(t1)}",
                  "        first: " + " | ".join(f"{mc.short(k['name'])[:44]} @{k['ts']:.1f}" for k in seg[:3]),
                  "        last : " + " | ".join(f"{mc.short(k['name'])[:44]} @{k['ts']:.1f}" for k in seg[-3:]),
                  f"        core : {mc.short(core['name'])[:60]} @{core['ts']:.1f} (annotation +{core['ts'] - A['ts']:.1f} us)"]
        side = T.side_window(t0, nxt)
        iv = [(k["ts"], k["ts"] + k["dur"]) for k in seg]
        srows = [(k, mc.overlap_us(k, iv)) for k in side]
        if srows:
            st_, so_ = sum(k["dur"] for k, _ in srows), sum(o for _, o in srows)
            print(f"        + side stream: {len(srows)} kernels {st_:.1f} us, overlapped by main-stream work {so_:.1f} us "
                  f"(not overlapped {st_ - so_:.1f} us; layer time above is main stream only)")
        base = {"phase": "prefill", "annotation": A["name"], "ann_ts_us": round(A["ts"], 3), "variant": v,
                "layer_idx": li, "layer_t0_us": round(t0, 3), "layer_t1_us": round(t1, 3), "ann_ui": T.ui(A["ts"]),
                "layer_ui": f"{T.ui(t0)} .. {T.ui(t1)}", "layer_count": NV[v]}
        for r in rows:
            mt = mc.row_metrics(r, A["name"], cfg, T, peaks)
            detail.append({**base, "kernel_ts_us": round(r["k"]["ts"], 3), "kernel_ui": T.ui(r["k"]["ts"]),
                           "group": mc.group(r["m"]), "module": r["m"], "kernel": mc.short(r["k"]["name"]),
                           "op": (T.op_of(r["k"]) or {}).get("name", ""), "stream": r["k"]["tid"], "overlap": "",
                           "overlapped_us": "", "time_us": round(r["k"]["dur"], 2),
                           "ratio": f"{100 * r['k']['dur'] / tot:.2f}%", "m": mt.get("m", ""), "k": mt.get("k", ""),
                           "n": mt.get("n", ""), "dtype": mt.get("dtype", ""), "tflops": mt.get("tflops", ""),
                           "tbytes_s": mt.get("tbytes_s", ""), "note": mt.get("src", "")})
        for k, ov in srows:
            detail.append({**base, "kernel_ts_us": round(k["ts"], 3), "kernel_ui": T.ui(k["ts"]), "group": "other",
                           "module": "side_stream", "kernel": mc.short(k["name"]),
                           "op": (T.op_of(k) or {}).get("name", ""), "stream": k["tid"], "overlap": "yes",
                           "overlapped_us": round(ov, 2), "time_us": round(k["dur"], 2), "ratio": "0 (side stream)",
                           "m": "", "k": "", "n": "", "dtype": "", "tflops": "", "tbytes_s": "", "note": ""})
    for where, rows in (("forward head", rep["head"]), ("forward tail", rep["tail"])):
        for r in rows:
            mt = mc.row_metrics(r, A["name"], cfg, T, peaks)
            extras.append({"where": where, "layer_idx": "", "module": r["m"], "kernel": mc.short(r["k"]["name"]),
                           "time_us": round(r["k"]["dur"], 2), "kernel_ts_us": round(r["k"]["ts"], 3),
                           "kernel_ui": T.ui(r["k"]["ts"]), "tflops": mt.get("tflops", ""),
                           "tbytes_s": mt.get("tbytes_s", ""), "note": mt.get("src", "")})

    mc.write_csv(f"{args.out}/prefill_detail.csv", detail)
    mc.write_csv(f"{args.out}/prefill_extras.csv", extras,
                 ["where", "layer_idx", "module", "kernel", "time_us", "kernel_ts_us", "kernel_ui", "tflops",
                  "tbytes_s", "note"])

    # 宽表：每种层一栏，表头带 annotation（工作点跟着数字走）
    with open(f"{args.out}/prefill_wide.csv", "w", newline="") as f:
        wr = csv.writer(f)
        cols = {v: [r for r in detail if r["variant"] == v] for v in NV}
        hdr = []
        for v in NV:
            rs = cols[v]
            t = sum(r["time_us"] for r in rs if not r["overlap"])
            sd = [r for r in rs if r["overlap"]]
            side = (f"  + side stream {sum(r['time_us'] for r in sd):.1f}us, not overlapped "
                    f"{sum(r['time_us'] - r['overlapped_us'] for r in sd):.1f}us" if sd else "")
            hdr += [f"{rs[0]['annotation']} @UI {rs[0]['ann_ui']}  {v} layer#{rs[0]['layer_idx']} "
                    f"[UI {rs[0]['layer_ui']}] = {t:.1f}us x{NV[v]} = {t * NV[v] / 1000:.2f}ms (main stream){side}"] \
                + [""] * len(WF)
        wr.writerow(hdr)
        wr.writerow((WF + [""]) * len(NV))
        for i in range(max(len(c) for c in cols.values())):
            line = []
            for v in NV:
                r = cols[v][i] if i < len(cols[v]) else None
                line += ([r[x] for x in WK] if r else [""] * len(WF)) + [""]
            wr.writerow(line)

    # 模块总时间：整个 forward 全部层加总，精确
    tot_rep = mod_totals(rep)
    allt = [mod_totals(r) for r in steady]
    mods = sorted(set().union(*allt), key=lambda m: -tot_rep.get(m, 0))
    summ = []
    for m in mods:
        vals = [t.get(m, 0.0) for t in allt]
        summ.append({"group": mc.group(m), "module": m, "rep_forward_ms": round(tot_rep.get(m, 0) / 1e3, 3),
                     "rep_share": f"{100 * tot_rep.get(m, 0) / rep['busy']:.2f}%",
                     "steady_median_ms": round(statistics.median(vals) / 1e3, 3),
                     "steady_min_ms": round(min(vals) / 1e3, 3), "steady_max_ms": round(max(vals) / 1e3, 3)})
    mc.write_csv(f"{args.out}/prefill_module_summary.csv", summ)
    gt = collections.defaultdict(float)
    for m, t in tot_rep.items():
        gt[mc.group(m)] += t
    print(f"\n group totals, representative forward (sum over all {cfg.nl} layers + head + tail = main-stream busy "
          f"{rep['busy'] / 1e3:.2f} ms):")
    for g, t in sorted(gt.items(), key=lambda x: -x[1]):
        vals = sorted(sum(v for m, v in tt.items() if mc.group(m) == g) for tt in allt)
        print(f"   {g:26s} {t / 1e3:8.2f} ms  {100 * t / rep['busy']:5.1f}%   steady median "
              f"{statistics.median(vals) / 1e3:7.2f}  [{vals[0] / 1e3:.2f} .. {vals[-1] / 1e3:.2f}]")
    print(f"   {'sum':26s} {sum(gt.values()) / 1e3:8.2f} ms   (check vs busy: diff {abs(sum(gt.values()) - rep['busy']):.1f} us)")

    # 通信：floor vs 超出部分
    print("\n communication in the representative forward (a call = one run of copy + sync + collective kernels; "
          "floor = fastest same-kind call across all steady forwards):")
    calls = collections.defaultdict(list)
    for r in steady:
        for rows in [r["head"]] + r["rows"] + [r["tail"]]:
            for m, d, _ in comm_calls(rows):
                calls[m].append(d)
    comm_rows = []
    for m in sorted(calls):
        rs = [(d, s) for rows in [rep["head"]] + rep["rows"] + [rep["tail"]] for mm, d, s in comm_calls(rows) if mm == m]
        tot = sum(d for d, _ in rs)
        sync = sum(s for _, s in rs)
        fl, med = min(calls[m]), statistics.median(calls[m])
        comm_rows.append({"module": m, "calls_per_forward": len(rs), "rep_total_ms": round(tot / 1e3, 3),
                          "of_which_sync_kernels_ms": round(sync / 1e3, 3), "floor_us": round(fl, 1),
                          "p50_us": round(med, 1), "floor_x_calls_ms": round(fl * len(rs) / 1e3, 3),
                          "excess_over_floor_ms": round((tot - fl * len(rs)) / 1e3, 3)})
        print(f"   {m:20s} calls={len(rs):3d} total={tot / 1e3:7.2f} ms (sync kernels {sync / 1e3:6.2f})  "
              f"floor={fl:7.1f} us  p50={med:7.1f} us  floor*calls={fl * len(rs) / 1e3:6.2f} ms  "
              f"excess={(tot - fl * len(rs)) / 1e3:6.2f} ms")
    mc.write_csv(f"{args.out}/prefill_comm.csv", comm_rows)

    # 中位层模型 vs 代表 forward
    print("\n reconcile (representative forward):")
    model = sum(NV[v] * layer_time(rep["rows"][rep_li[v]]) for v in NV)
    ht = layer_time(rep["head"]) + layer_time(rep["tail"])
    ext = sum(e["time_us"] for e in extras if e["where"].startswith("layer"))
    print(f"   median-layer model {model / 1e3:.2f} ms + head/tail {ht / 1e3:.2f} ms + minority-layer kernels "
          f"{ext / 1e3:.2f} ms = {(model + ht + ext) / 1e3:.2f} ms  vs  main-stream busy {rep['busy'] / 1e3:.2f} ms  "
          f"vs wall {A['dur'] / 1e3:.2f} ms")
    mins = collections.defaultdict(lambda: [0, 0.0, []])
    for e in extras:
        if e["where"].startswith("layer"):
            x = mins[(e["module"], e["kernel"][:50])]
            x[0] += 1
            x[1] += e["time_us"]
            x[2].append(e["layer_idx"])
    for (m, k), (n, t, ls) in sorted(mins.items(), key=lambda x: -x[1][1]):
        print(f"   minority kernel (pitfall: a median layer drops it): {m:24s} {k:50s} x{n} at layers {ls}  {t:.1f} us")
    UNC = ("misc_small", "gemm_unmapped", "act_quant", "head_misc", "tail_misc", "comm_other", "attn_out_misc")
    allrows = rep["head"] + [x for L in rep["rows"] for x in L] + rep["tail"]
    um = collections.Counter(r["m"] for r in allrows if r["m"] in UNC)
    ut = sum(r["k"]["dur"] for r in allrows if r["m"] in UNC)
    print(f"   unclassified {sorted(UNC)}: {ut / 1e3:.3f} ms = {100 * ut / rep['busy']:.2f}% of busy  {dict(um)}")
    sk = T.side_window(A["ts"], A["ts"] + A["dur"])
    if sk:
        iv = [(x["ts"], x["ts"] + x["dur"]) for x in rep["w"]]
        st = sum(k["dur"] for k in sk)
        so = sum(mc.overlap_us(k, iv) for k in sk)
        print(f"   side streams in the representative forward: {len(sk)} kernels {st / 1e3:.2f} ms, overlapped by "
              f"main-stream work {so / 1e3:.2f} ms, not overlapped {(st - so) / 1e3:.2f} ms (not in the totals above; "
              f"tids {dict(collections.Counter(k['tid'] for k in sk))})")
    print("\n where to look in the trace:\n" + "\n".join(PICKS))
    open(f"{args.out}/prefill_picks.txt", "w").write("\n".join(PICKS) + "\n")
    json.dump({"annotation": A["name"], "wall_ms": A["dur"] / 1e3, "busy_ms": rep["busy"] / 1e3,
               "steady": len(steady), "segmented": len(ok), "forwards": len(fw), "shapes_recorded": T.has_shapes(),
               "unclassified_share": ut / rep["busy"], "model_ms": (model + ht + ext) / 1e3,
               "rep_layers": rep_li, "module_set": sorted(tot_rep)}, open(f"{args.out}/prefill_meta.json", "w"), indent=1)
    print(f"\n wrote {args.out}/prefill_detail.csv ({len(detail)} rows), prefill_wide.csv, prefill_picks.txt, "
          f"prefill_module_summary.csv ({len(summ)}), prefill_comm.csv, prefill_extras.csv ({len(extras)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
