#!/usr/bin/env python3
"""一个 trace 一张宽表：左边 prefill 各层类型的栏（m3_prefill.py 的 prefill_wide.csv），右边 decode
（m3_decode.py 的 decode_wide.csv）。第一行写明 trace、工作点和方法（shape 是录到的还是推导的、decode 配对是否可信），
数字离开这张表时这些信息跟着走。
用法: m3_combine_wide.py <breakdown dir> <out.csv> "<title: 模型, 并发, rank, 镜像, 窗口>"
"""
import csv
import json
import os
import sys


def _meta(d, phase):
    p = os.path.join(d, f"{phase}_meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        sys.exit(__doc__)
    d, out, title = argv
    pf, df = os.path.join(d, "prefill_wide.csv"), os.path.join(d, "decode_wide.csv")
    p = list(csv.reader(open(pf))) if os.path.exists(pf) else []
    q = list(csv.reader(open(df))) if os.path.exists(df) else []
    if not p and not q:
        sys.exit(f"neither prefill_wide.csv nor decode_wide.csv in {d}")
    wp = max((len(r) for r in p), default=0)
    rows = [(p[i] if i < len(p) else []) + [""] * (wp - len(p[i] if i < len(p) else [])) + [""] + (q[i] if i < len(q) else [])
            for i in range(max(len(p), len(q)))]
    pm, dm = _meta(d, "prefill"), _meta(d, "decode")
    left = ("prefill: none in this window" if not p else
            f"prefill (eager; {'recorded shapes' if pm.get('shapes_recorded') else 'M/K/N derived from config at TP'}; "
            f"one median layer per layer type; times as Perfetto UI HH:MM:SS.nnnnnnnnn)")
    if not q:
        right = "decode: none in this window"
    elif dm.get("scrambled"):
        right = (f"decode {dm.get('key', '')} as recorded: HIP-graph name/duration pairing SCRAMBLED "
                 f"(p90/p50 {dm.get('spread', float('nan')):.2f}) -- use the union totals, not the per-kernel numbers")
    else:
        right = (f"decode {dm.get('key', '')}: {dm.get('steps', '?')} steps pooled, same-name p90/p50 "
                 f"{dm.get('spread', float('nan')):.2f} (pairing intact); per-row time is the critical-path share, rows sum "
                 f"to the layer's union busy; M/K/N derived from config at TP")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"{title}. Left: {left}. Right: {right}."])
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
