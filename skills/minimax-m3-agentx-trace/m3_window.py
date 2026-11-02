#!/usr/bin/env python3
"""MiniMax-M3 AgentX 采集窗口的总账，以及和 aiperf ITL 的对账。

kernel 时间不是 ITL（照 trace-kernel-breakdown 的做法）：
  - 一个 decode step = 主模型 forward + EAGLE3 drafter（propose_eagle x spec）+ prepare/postprocess；
    每个序列每步出 toks/fwd 个 token（1 + 接受的 draft 数），所以 ITL = step 周期 / toks_per_fwd；
  - AgentX 的 chunked prefill 夹在 decode 之间，序列在 prefill forward 期间不出 token，
    所以要用 窗口 / decode forward 数，而不是 decode forward 本身的时间；
  - toks/fwd 用 server log 里这个窗口自己的 MTP 统计（start_profile 和 stop_profile 之间的
    [MTP Stats Interval] 行，按 draft token 数加权），不用整次运行的平均；
  - 窗口在服务运行中途开关：位置相对 aiperf profiling 阶段开始多少秒（目标 600..900 s），
    以及 stop_profile 导出期间引擎暂停，跨过这段的请求 ITL 会被拉高（aiperf 的数字是被扰动过的）。
用法: m3_window.py <reduced 或原始 trace> [--server-log server.log] [--aiperf-dir DIR] [--toks-per-fwd X]
                   [--out window_summary.json]
"""
import argparse
import bisect
import collections
import json
import os
import re
import statistics
import sys

import m3_common as mc

KINDS = ("prefill", "decode", "propose_eagle", "ATOM::prepare_model", "ATOM::postprocess", "nccl:broadcast")


def kind_of(name):
    return next((k for k in KINDS if name.startswith(k)), "other")


def trace_stem(path):
    b = os.path.basename(path)
    return re.split(r"\.pt\.trace|\.reduced|\.json", b)[0]


def parse_server_log(path, stem):
    """每个 profiler 窗口：start/stop 行号、时刻、stop 耗时（导出暂停）、导出的文件、窗口内的 MTP 统计。"""
    lines = open(path, errors="replace").read().splitlines()
    wins, cur = [], None
    for i, ln in enumerate(lines):
        if "executing utility command: start_profile" in ln:
            cur = {"start_line": i + 1, "start_hms": ln[6:14], "acc": 0, "tot": 0, "intervals": 0, "files": []}
            wins.append(cur)
        elif cur is not None and "executing utility command: stop_profile" in ln:
            cur["stop_line"] = i + 1
            cur["stop_hms"] = ln[6:14]
        elif cur is not None and "trace exported to" in ln:
            m = re.search(r"trace exported to (\S+) \(([\d.]+) MB, ([\d.]+)s\)", ln)
            if m:
                cur["files"].append((m.group(1), float(m.group(2)), float(m.group(3))))
        elif cur is not None and "utility command 'stop_profile' finished in" in ln:
            m = re.search(r"finished in ([\d.]+)s", ln)
            cur["stop_s"] = float(m.group(1)) if m else None
            cur = cur if "stop_line" not in cur else None
        elif cur is not None and "stop_line" not in cur and "[MTP Stats Interval]" in ln:
            m = re.search(r"Accepted/Total Draft tokens: (\d+)/(\d+)", ln)
            if m:
                cur["acc"] += int(m.group(1))
                cur["tot"] += int(m.group(2))
                cur["intervals"] += 1
    pick = next((w for w in wins if any(trace_stem(f) == stem for f, _, _ in w["files"])), None)
    return wins, pick


def load_aiperf(d):
    out = {}
    j = os.path.join(d, "profile_export_aiperf.json")
    if os.path.exists(j):
        a = json.load(open(j))
        out["itl"] = a.get("inter_token_latency") or {}
        out["ttft"] = a.get("time_to_first_token") or {}
        out["duration_s"] = (a.get("benchmark_duration") or {}).get("avg")
    recs = []
    p = os.path.join(d, "profile_export.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            for ln in f:
                r = json.loads(ln)
                md, mt = r.get("metadata", {}), r.get("metrics", {})
                recs.append({"phase": md.get("benchmark_phase"), "t0": md.get("request_start_ns"),
                             "t1": md.get("request_end_ns"),
                             "itl": (mt.get("inter_token_latency") or {}).get("value"),
                             "osl": (mt.get("output_sequence_length") or {}).get("value")})
    out["records"] = recs
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--aiperf-dir", default=None)
    ap.add_argument("--trace-name", default=None,
                    help="original trace file name (as in the server log's 'trace exported to' line), when the trace "
                         "given is a reduced copy")
    ap.add_argument("--toks-per-fwd", type=float, default=None,
                    help="tokens per sequence per decode step, when the server log is not available")
    ap.add_argument("--out", default=None, help="write the numbers as json")
    args = ap.parse_args(argv)
    T = mc.Trace(args.trace)
    ann = sorted(T.main_ann, key=lambda a: a["ts"])
    fw = [a for a in ann if kind_of(a["name"]) in ("prefill", "decode")]
    if not fw:
        sys.exit("no prefill[...] / decode[...] annotation on the main stream")
    t0, t1 = fw[0]["ts"], fw[-1]["ts"] + fw[-1]["dur"]
    span = t1 - t0
    S = {"trace": args.trace, "window_ms": span / 1e3}
    print(f"== window {span / 1e3:.1f} ms (first forward start .. last forward end); GPU events "
          f"{(T.g1 - T.g0) / 1e3:.1f} ms; main stream {T.main_tid}")
    tot = collections.defaultdict(float)
    cnt = collections.Counter()
    for a in ann:
        if t0 <= a["ts"] < t1:
            k = kind_of(a["name"])
            tot[k] += a["dur"]
            cnt[k] += 1
    busy = mc.union([(a["ts"], a["ts"] + a["dur"]) for a in ann if t0 <= a["ts"] < t1])
    for k in KINDS + ("other",):
        if cnt[k]:
            print(f"   {k:22s} x{cnt[k]:5d}  {tot[k] / 1e3:9.1f} ms  {100 * tot[k] / span:5.1f}% of window")
    print(f"   outside any main-stream annotation: {(span - busy) / 1e3:.1f} ms ({100 * (span - busy) / span:.1f}%)")
    mk = [k for k in T.K if t0 <= k["ts"] < t1]
    kb = sum(k["dur"] for k in mk)
    print(f"   main-stream kernel busy {kb / 1e3:.1f} ms ({100 * kb / span:.1f}% of window); side streams "
          f"{sum(k['dur'] for k in T.SIDE if t0 <= k['ts'] < t1) / 1e3:.1f} ms")
    S.update({k: {"count": cnt[k], "ms": tot[k] / 1e3} for k in cnt})
    S["main_kernel_busy_ms"] = kb / 1e3

    pre = [a for a in fw if a["name"].startswith("prefill[")]
    dec = [a for a in fw if a["name"].startswith("decode[")]
    if pre:
        tok = [mc.ann_fields(a["name"]).get("tok", 0) for a in pre]
        print(f"   prefill forwards: {len(pre)}, tok p50 {statistics.median(tok):.0f} (min {min(tok):.0f}, max "
              f"{max(tok):.0f}), total {sum(tok):.0f} tokens; wall p50 {statistics.median(a['dur'] for a in pre) / 1e3:.1f} ms")
    spec = collections.Counter(int(mc.ann_fields(a["name"]).get("spec", 0)) for a in dec).most_common(1)
    spec = spec[0][0] if spec else 0
    tpf, src = args.toks_per_fwd, "--toks-per-fwd" if args.toks_per_fwd else None
    if args.server_log:
        stem = trace_stem(args.trace_name or args.trace)
        wins, w = parse_server_log(args.server_log, stem)
        if w is None and wins:
            print(f"   server log: {len(wins)} profiler windows, none exported {stem}; using the last")
            w = wins[-1]
        if w:
            print(f"   server log: window {w['start_hms']} .. {w.get('stop_hms', '?')}, stop_profile took "
                  f"{w.get('stop_s', float('nan')):.1f} s (engine paused while exporting: requests in flight then see it)")
            S["stop_profile_s"] = w.get("stop_s")
            if w["tot"] and spec and tpf is None:
                tpf = 1 + spec * w["acc"] / w["tot"]
                src = (f"server log, {w['intervals']} [MTP Stats Interval] lines inside the window: accepted "
                       f"{w['acc']}/{w['tot']} draft tokens, spec={spec}")
            elif tpf is None:
                print("   no [MTP Stats Interval] line inside the window (too short or too few steps): pass --toks-per-fwd")
    if dec:
        steps, parts = [], collections.defaultdict(list)
        ats = [a["ts"] for a in ann]
        for a, b in zip(dec, dec[1:]):
            mid = ann[bisect.bisect_right(ats, a["ts"]): bisect.bisect_left(ats, b["ts"])]
            if any(kind_of(x["name"]) == "prefill" for x in mid):
                continue
            p = b["ts"] - a["ts"]
            steps.append(p)
            # 排他归属：postprocess 和下一个 forward 有重叠，按这个顺序每部分只记前面没覆盖的时间，各部分之和 = 周期
            cov = []
            for k in ("decode", "propose_eagle", "ATOM::prepare_model", "nccl:broadcast", "ATOM::postprocess"):
                iv = [(x["ts"], min(x["ts"] + x["dur"], b["ts"])) for x in [a] + mid if kind_of(x["name"]) == k]
                before = mc.union(cov)
                cov += iv
                parts["decode forward" if k == "decode" else k].append(mc.union(cov) - before)
            parts["gap (no main-stream annotation)"].append(p - mc.union(cov))
        print(f"   decode forwards: {len(dec)}; decode->decode with no prefill between: {len(steps)}")
        if steps:
            sp = statistics.median(steps)
            print(f"   step period (decode->decode, no prefill between): median {sp / 1e3:.3f} ms  p10 "
                  f"{mc.pct(steps, 0.1) / 1e3:.3f}  p90 {mc.pct(steps, 0.9) / 1e3:.3f}  max {max(steps) / 1e3:.3f}")
            print("      exclusive parts (each counts only time not covered by the ones above it; they sum to the period):")
            for k, v in parts.items():
                print(f"      {k:34s} median {statistics.median(v) / 1e3:7.3f} ms  mean {statistics.mean(v) / 1e3:7.3f}")
            S["step_period_ms"] = sp / 1e3
            S["step_parts_ms"] = {k: statistics.median(v) / 1e3 for k, v in parts.items()}
        per_dec = span / len(dec)
        S["window_per_decode_ms"] = per_dec / 1e3
        print(f"   window / decode forwards = {per_dec / 1e3:.3f} ms (includes the prefill share of the window)")
        if tpf:
            S["toks_per_fwd"], S["toks_per_fwd_src"] = tpf, src
            print(f"   tokens per sequence per step: {tpf:.3f}  ({src})")
            if steps:
                print(f"   predicted ITL, decode-only steps: {sp / 1e3 / tpf:.3f} ms  (step period / toks_per_fwd)")
                S["itl_pred_decode_only_ms"] = sp / 1e3 / tpf
            print(f"   predicted ITL, whole window:      {per_dec / 1e3 / tpf:.3f} ms  (window / decode forwards / "
                  f"toks_per_fwd; prefill interleaving included)")
            S["itl_pred_window_ms"] = per_dec / 1e3 / tpf
    if args.aiperf_dir:
        A = load_aiperf(args.aiperf_dir)
        itl = A.get("itl") or {}
        if itl:
            print(f"   aiperf ITL (whole profiling phase): avg {itl.get('avg', float('nan')):.3f}  p50 "
                  f"{itl.get('p50', float('nan')):.3f}  p90 {itl.get('p90', float('nan')):.3f} ms  (count {itl.get('count')})")
            S["aiperf_itl_ms"] = {k: itl.get(k) for k in ("avg", "p50", "p90")}
        prof = [r for r in A["records"] if r["phase"] == "profiling" and r["t0"]]
        if prof and T.meta.get("baseTimeNanoseconds"):
            e0, e1 = T.epoch_ns(t0), T.epoch_ns(t1)
            ph0 = min(r["t0"] for r in prof)
            off = (e0 - ph0) / 1e9
            S["window_offset_s"] = off
            print(f"   window starts {off:.0f} s into aiperf's profiling phase (target 600..900 s"
                  f"{'' if 600 <= off <= 900 else ' -- OUTSIDE the target'})")
            inw = [r for r in prof if r["t1"] and r["t0"] < e1 and r["t1"] > e0 and r["itl"]]
            pause = (S.get("stop_profile_s") or 0) * 1e9
            before = [r for r in prof if r["t1"] and r["t1"] <= e0 and r["t1"] > e0 - 300e9 and r["itl"]]
            after = [r for r in prof if r["t0"] >= e1 + pause and r["t0"] < e1 + pause + 300e9 and r["itl"]]
            for tag, rs in (("requests overlapping the window", inw), ("finished in the 300 s before it", before),
                            ("started in the 300 s after the export", after)):
                if rs:
                    v = [r["itl"] for r in rs]
                    print(f"   aiperf ITL, {tag:38s}: n={len(rs):4d}  p50 {statistics.median(v):.3f}  mean "
                          f"{statistics.mean(v):.3f} ms")
                    S[f"aiperf_itl_{tag.split()[0]}_p50_ms"] = statistics.median(v)
        elif prof:
            print("   trace has no baseTimeNanoseconds: cannot place the window against aiperf records")
    if args.out:
        json.dump(S, open(args.out, "w"), indent=1)
        print(f"   wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
