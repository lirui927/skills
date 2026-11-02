#!/usr/bin/env python3
"""两个 rank 的同一批 forward：通信慢是两边都慢（通信本身，或第三个 rank 慢），还是一个 rank 在等另一个（skew）？
照 trace-kernel-breakdown 的 rank_skew 做法：一次集合通信在所有 rank 上一起结束（最后一个到达后才结束），所以用
结束时间对齐两个 rank 的时钟：offset = median(end_a - end_b)；然后每次调用的到达 skew = start_a - (start_b + offset)，
以及各 rank 的时长。M3 上一次调用是主 stream 上一组连续的通信 kernel（拷贝 + start_sync + all-reduce/norm +
end_sync，见 m3_common.comm_mask），整组一起比；start_sync 的时长单独列出：先到的 rank 在 barrier 里等。
两边时长都长、skew 只有几微秒 = 通信本身（或另一个 rank）慢，不是这两个 rank 互相等。
用法: m3_rank_skew.py <reduced rank A> <reduced rank B> [--annotation NAME_PREFIX] [--kind prefill|decode]
默认 kind=prefill，取两边都有的所有同名 forward；--annotation 只取名字以它开头的 forward。
"""
import argparse
import collections
import statistics
import sys

import m3_common as mc


def kind_of(n):
    if "ar_rmsnorm" in n:
        return "allreduce+rmsnorm_aiter"
    if "ag_gfx1250" in n:
        return "allgather_aiter"
    if "ar_gfx1250" in n or "ar_ll_gfx1250" in n or "allreduce" in n.lower() or "all_reduce" in n.lower():
        return "allreduce_aiter"
    if "ncclDevKernel" in n or "mscclKernel" in n:
        return "nccl"
    if "ddaAllReduce" in n:
        return "allreduce_dda"
    if "ddaAllGather" in n:
        return "allgather_dda"
    return "other_collective"


def calls(T, a):
    """主 stream 上这个 forward 的通信调用：[(kind, start, end, 总时长, start_sync 时长)]。"""
    w = T.main_window(a)
    out = []
    for s, e in mc.runs(mc.comm_mask(w)):
        g = w[s:e]
        core = [k for k in g if mc.COMM.search(k["name"]) and "sync_kernel" not in k["name"]]
        kd = kind_of(core[0]["name"]) if core else "sync_only"
        out.append((kd, g[0]["ts"], max(k["ts"] + k["dur"] for k in g), sum(k["dur"] for k in g),
                    sum(k["dur"] for k in g if "start_sync" in k["name"])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_a")
    ap.add_argument("trace_b")
    ap.add_argument("--annotation", default=None)
    ap.add_argument("--kind", default="prefill", choices=("prefill", "decode"))
    args = ap.parse_args(argv)
    A, B = mc.Trace(args.trace_a), mc.Trace(args.trace_b)
    fa = [a for a in A.forwards(args.kind) if not args.annotation or a["name"].startswith(args.annotation)]
    fb = [a for a in B.forwards(args.kind) if not args.annotation or a["name"].startswith(args.annotation)]
    names_b = collections.defaultdict(list)
    for a in fb:
        names_b[a["name"]].append(a)
    pairs = []
    for a in fa:
        if names_b[a["name"]]:
            pairs.append((a, names_b[a["name"]].pop(0)))
    if not pairs:
        sys.exit(f"no {args.kind} forward with the same annotation on both ranks")
    xs = []
    for a, b in pairs:
        ca, cb = calls(A, a), calls(B, b)
        if [c[0] for c in ca] != [c[0] for c in cb]:
            print(f"  skip {a['name'][:70]}: collective sequences differ ({len(ca)} vs {len(cb)} calls)")
            continue
        xs.append((a, b, ca, cb))
    if not xs:
        sys.exit("collective sequences differ on every paired forward")
    by = collections.defaultdict(list)
    offs, mism = [], []
    for a, b, ca, cb in xs:
        off = statistics.median(x[2] - y[2] for x, y in zip(ca, cb))
        offs.append(off)
        mism += [abs(x[2] - y[2] - off) for x, y in zip(ca, cb)]
        for x, y in zip(ca, cb):
            by[x[0]].append((x[3], y[3], x[1] - (y[1] + off), x[4], y[4]))
    a0, b0 = xs[0][0], xs[0][1]
    print(f"{args.kind}: {len(xs)} forwards paired by annotation (first: {a0['name'][:80]})")
    print(f"  wall A {statistics.median(x[0]['dur'] for x in xs) / 1e3:.2f} ms, B "
          f"{statistics.median(x[1]['dur'] for x in xs) / 1e3:.2f} ms (median); calls per forward {len(xs[0][2])}; "
          f"clock offset (A - B, from END times) p50 {statistics.median(offs):.1f} us; |end mismatch| p50 "
          f"{mc.pct(mism, 0.5):.2f} us p90 {mc.pct(mism, 0.9):.2f} us")
    if mc.pct(mism, 0.9) > 5:
        print("  end mismatch p90 > 5 us: the END-time alignment is loose (different collective boundaries or a "
              "clock drift inside the window); read the skew column with care")
    p = lambda v: mc.pct(v, 0.5)
    print(f"  {'kind':24s} {'calls':>6s} {'dur A p50':>9s} {'dur B p50':>9s} {'min p50':>8s} {'max p50':>8s} "
          f"{'skew A-B p50':>12s} {'|skew| p50':>10s} {'A later %':>9s} {'sync A p50':>10s} {'sync B p50':>10s}"
          f"  total A / B / min-sum ms")
    for kd, v in sorted(by.items(), key=lambda t: -sum(x[0] for x in t[1])):
        mn = [min(x[0], x[1]) for x in v]
        mx = [max(x[0], x[1]) for x in v]
        print(f"  {kd:24s} {len(v):6d} {p([x[0] for x in v]):9.1f} {p([x[1] for x in v]):9.1f} {p(mn):8.1f} {p(mx):8.1f} "
              f"{p([x[2] for x in v]):12.1f} {p([abs(x[2]) for x in v]):10.1f} "
              f"{100 * sum(x[2] > 0 for x in v) / len(v):9.0f} {p([x[3] for x in v]):10.1f} {p([x[4] for x in v]):10.1f}"
              f"  {sum(x[0] for x in v) / 1e3:8.2f} / {sum(x[1] for x in v) / 1e3:8.2f} / {sum(mn) / 1e3:8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
