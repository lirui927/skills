#!/usr/bin/env python3
"""One pass over a large ATOM torch trace (.pt.trace.json.gz, ~20 GB uncompressed with stacks):

1. list every main-stream forward annotation (prefill[...] / decode[...] / dummy_*) with its kernel count and
   GPU-busy share, so the representative annotations can be chosen before the breakdown;
2. write a reduced trace holding only what kimi_k3_breakdown.py reads: GPU kernels / memcpy / memset,
   GPU + CPU user annotations, cuda_runtime launches, the profiler's `Trace` span (the UI's time zero), and
   cpu_op events that carry recorded shapes (Input Dims / Input type / External id). python_function (stack)
   events are left out of the reduced file (they are ~78% of the bytes); the full file keeps them.

Usage: python3 extract_trace.py <trace.pt.trace.json.gz> <out_dir>
Needs orjson and enough RAM for the decompressed trace (run it on a tray, not the jump host).
"""
import bisect
import collections
import gzip
import json
import os
import sys
import time

try:
    import orjson
except ImportError:          # some trays have no orjson on the host python: slower, same result
    class orjson:            # noqa: N801 -- minimal stand-in with the two calls used below
        loads = staticmethod(json.loads)
        dumps = staticmethod(lambda o: json.dumps(o).encode())

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
t0 = time.time()
with gzip.open(src, "rb") as f:
    raw = f.read()
print(f"decompressed {len(raw)/1e9:.1f} GB in {time.time()-t0:.0f}s", flush=True)
doc = orjson.loads(raw)
del raw
ev = doc["traceEvents"]
print(f"parsed {len(ev):,} events in {time.time()-t0:.0f}s", flush=True)
cats = collections.Counter(e.get("cat") for e in ev)
print("categories:", dict(cats.most_common()), flush=True)

GPU = ("kernel", "gpu_memcpy", "gpu_memset")
gpu = sorted((e for e in ev if e.get("cat") in GPU), key=lambda e: e["ts"])
main_tid = collections.Counter(e.get("tid") for e in gpu).most_common(1)[0][0]
main = [e for e in gpu if e.get("tid") == main_tid]
ts = [e["ts"] for e in main]
print(f"main GPU stream tid={main_tid}: {len(main):,} of {len(gpu):,} GPU events", flush=True)

ann = sorted((e for e in ev if e.get("cat") == "gpu_user_annotation" and e.get("tid") == main_tid
              and e.get("name", "").split("[")[0] in ("prefill", "decode", "dummy_decode", "dummy_prefill",
                                                       "eager_decode", "dummy_eager_decode")),
             key=lambda e: e["ts"])
rows = []
for a in ann:
    lo, hi = bisect.bisect_left(ts, a["ts"]), bisect.bisect_right(ts, a["ts"] + a["dur"])
    busy = sum(e.get("dur", 0) for e in main[lo:hi])
    rows.append({"name": a["name"], "ts": a["ts"], "dur_us": a["dur"], "kernels": hi - lo,
                 "busy_us": round(busy, 1), "busy_pct": round(100 * busy / a["dur"], 1) if a["dur"] else 0})
with open(os.path.join(out, "annotations.json"), "w") as f:
    json.dump({"main_tid": main_tid, "annotations": rows}, f, indent=1)
print(f"annotations on main stream: {len(rows)} -> {out}/annotations.json", flush=True)

shape_ops = 0
keep = []
for e in ev:
    c = e.get("cat")
    # "Trace" is the profiler span the trace UI counts its time axis from; without it the reduced file's zero
    # moves to its earliest kept event (66.7 ms later on the K3 14k/16 c32 trace) and every UI time is off.
    if c in GPU or c in ("gpu_user_annotation", "user_annotation", "cuda_runtime", "ac2g", "Trace"):
        keep.append(e)
    elif c == "cpu_op":
        a = e.get("args") or {}
        if a.get("Input Dims") or a.get("Input type"):
            keep.append(e)
            shape_ops += 1
red = {k: v for k, v in doc.items() if k != "traceEvents"}
red["traceEvents"] = keep
with gzip.open(os.path.join(out, "reduced.pt.trace.json.gz"), "wb", compresslevel=3) as f:
    f.write(orjson.dumps(red))
print(f"reduced trace: {len(keep):,} events ({shape_ops:,} cpu_op with recorded shapes) "
      f"-> {out}/reduced.pt.trace.json.gz  total {time.time()-t0:.0f}s", flush=True)
