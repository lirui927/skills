#!/usr/bin/env python3
"""MiniMax-M3 的 torch-profiler 窗口里，每个 forward 是否完整（照 trace-kernel-breakdown 的 trace_completeness 做法）。

窗口是在服务运行中途开关的，第一个和最后一个 forward 可能被截断；分析之前先确认每个 forward 都是完整的。
对主 stream 上每个 prefill[...] / decode[...] annotation（只看 annotation 内的 GPU 事件，Memcpy HtoD 丢掉）：
  - attention core 个数 = num_hidden_layers，其中稀疏层的 core（bs16 tile）个数 = config 里稀疏层数；
  - prefill（eager，kernel 顺序可信）：按 attention core 切层（m3_common.segment），逐层类型必须和 config 一致；
  - decode（HIP graph）：先比每步的名字计数（core / 稀疏 core / indexer / 通信 kernel / 总 kernel 数）
    和众数 step 是否一致；配对没乱（m3_decode.py 的 p90/p50 检查）时同样切层核对；
  - drafter：相邻两个 decode 之间 propose_eagle annotation 的个数（EAGLE3，应等于 spec）；
  - 窗口边界：第一个 / 最后一个 forward 离第一个 / 最后一个 GPU 事件多远，以及 profiler 'Trace' 事件的跨度。
用法: m3_trace_completeness.py <reduced.pt.trace.json.gz> <config.json> [label]
"""
import argparse
import bisect
import collections
import sys

import m3_common as mc


def count_all(allts, a):
    """annotation 内所有 stream 的 GPU 事件个数（HtoD 已丢）。"""
    return bisect.bisect_right(allts, a["ts"] + a["dur"]) - bisect.bisect_left(allts, a["ts"])


def decode_sig(w):
    return (sum(bool(mc.CORE.search(k["name"])) for k in w), sum(bool(mc.SPARSE_CORE.search(k["name"])) for k in w),
            sum(bool(mc.INDEXER.search(k["name"])) for k in w), sum(bool(mc.COMM.search(k["name"])) for k in w))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("config")
    ap.add_argument("label", nargs="?", default=None)
    args = ap.parse_args(argv)
    cfg = mc.M3Config(args.config)
    T = mc.Trace(args.trace)
    VAR = cfg.variants()
    n_sparse = sum(cfg.sparse)
    ann = [a for a in T.main_ann if a.get("name", "").split("[")[0] in ("prefill", "decode")]
    tr = T.trace_events[0] if T.trace_events else None
    print(f"== {args.label or args.trace}")
    print(f"   profiler Trace span: {(tr['dur'] / 1e6 if tr and tr.get('dur') else float('nan')):.2f} s; GPU events "
          f"{T.g0:.0f} .. {T.g1:.0f} ({(T.g1 - T.g0) / 1e6:.2f} s); main stream {T.main_tid}; forwards: {len(ann)}")
    if not ann:
        print("   RESULT: no prefill[...] / decode[...] annotation on the main stream (server not started with "
              "ATOM_ENABLE_DETAILED_ANNOTATION=1, or the window caught no forward)")
        return 1
    bad = []
    pre = [a for a in ann if a["name"].startswith("prefill[")]
    dec = [a for a in ann if a["name"].startswith("decode[")]
    print(f"   prefill: {len(pre)}  (expected per forward: {cfg.nl} attention cores, {n_sparse} of them sparse; "
          f"layers {dict(cfg.variant_counts())})")
    for a in pre:
        w = T.main_window(a)
        seg, why = mc.segment(w, cfg.nl)
        st = why
        if seg:
            obs = [mc.observed_variant(s) for s in seg["layers"]]
            st = "ok" if obs == VAR else f"layer types {dict(collections.Counter(obs))} != config"
        edge = " FIRST" if a is ann[0] else " LAST" if a is ann[-1] else ""
        print(f"     {st[:40]:40s} {a['name'][:72]:72s} wall {a['dur'] / 1e3:8.1f} ms  kernels(main) {len(w):6d}  ends "
              f"{(T.g1 - a['ts'] - a['dur']) / 1e3:8.1f} ms before the last GPU event{edge}")
        if st != "ok":
            bad.append(a["name"])
    sigs = {id(a): decode_sig(T.main_window(a)) for a in dec}
    mode = collections.Counter(sigs.values()).most_common()
    print(f"   decode: {len(dec)}; (cores, sparse cores, indexer kernels, comm kernels) per step -> steps: {mode[:4]}")
    exp = mode[0][0] if mode else None
    nseg = 0
    for a in dec:
        s = sigs[id(a)]
        odd = s != exp or s[0] != cfg.nl or s[1] != n_sparse
        if not odd:
            seg, why = mc.segment(T.main_window(a), cfg.nl)
            if seg and [mc.observed_variant(x) for x in seg["layers"]] == VAR:
                nseg += 1
            else:
                odd = True
        if odd:
            bad.append(a["name"])
            print(f"     ODD decode {a['name'][:80]} ts {a['ts']:.0f} sig {s}"
                  f"{' FIRST' if a is ann[0] else ' LAST' if a is ann[-1] else ''}")
    if dec:
        allts = sorted(k["ts"] for k in T.K + T.SIDE)
        kc = collections.Counter(count_all(allts, a) for a in dec)
        print(f"   decode kernels per step (all streams, no HtoD): {sorted(kc.items())[:8]}")
        print(f"   decode steps that segment into the config's layers: {nseg} of {len(dec)}")
        props = [x for x in T.main_ann if x.get("name", "").startswith("propose_eagle")]
        pts = sorted(x["ts"] for x in props)
        between = collections.Counter(
            bisect.bisect_left(pts, b["ts"]) - bisect.bisect_right(pts, a["ts"]) for a, b in zip(dec, dec[1:]))
        spec = collections.Counter(int(mc.ann_fields(a["name"]).get("spec", 0)) for a in dec)
        print(f"   drafter: {len(props)} propose_eagle annotations; per decode->decode gap: {sorted(between.items())}  "
              f"(decode spec= {dict(spec)})")
    for tag, a in (("first", ann[0]), ("last", ann[-1])):
        print(f"   {tag} forward {a['name'][:64]} starts {a['ts'] - T.g0:.0f} us after the first GPU event, ends "
              f"{(T.g1 - a['ts'] - a['dur']) / 1e3:.1f} ms before the last one")
    print(f"   RESULT: {len(ann) - len(bad)} of {len(ann)} forwards whole" + (f"; not whole: {bad[:6]}" if bad else ""))
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
