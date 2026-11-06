#!/usr/bin/env python3
"""固定 shape 重放：decode 固定在 bs=N，同一个 profiler 窗口里再插一个指定 shape 的 prefill。
capture_agentx.sh WORKLOAD=fixed 在容器里调用；只用标准库。

AgentX 回放里 decode 的 batch 随时间起伏（recipe：任一时刻只有 22-41% 的并发在 decode），要一个确定的 batch 时用它：
  1. 一个 2,048 token 的预热请求；
  2. N 个互不相同、各 CTX token 的随机 token-id prompt，逐个发一次（max_tokens=1）进 prefix cache；
     再发一次 prefill 用的前缀（PREFILL_CACHED token，整 block）；
  3. N 个 prompt 一起流式发出（ignore_eos）：每个只重算末尾不满一个 block 的 token，同一步进入 decode，之后每步 bs=N；
  4. 不开 profiler 的对照：REF 秒 decode 的客户端步长（同一流相邻 chunk 的间隔），再发一次同 shape 的 prefill
     （前缀 + 新的 PREFILL_NEW 个 token）量耗时，顺带把这个长度的 plan / JIT 预热掉，窗口里不出现冷编译；
  5. /start_profile -> PRE 秒 decode -> 窗口里的 prefill -> POST 秒 decode -> /stop_profile（返回时导出已完成）。
prompt 是固定 seed 的随机 token id（1000..150000，避开特殊 token），不需要 tokenizer，长度精确；同一个 seed 在
任何机器上发出的 token id 都一样。token id 列表放在 `prompt` 里，服务不接受时改放 `prompt_token_ids`。
--summary 写 json：参数、各时刻（epoch 秒）、对照和窗口里的客户端步长、两次 prefill 的耗时、server log 里的 shape 核对。
用法（容器内）: python3 m3_fixed.py --bs 32 --summary /out/fixed_summary.json --server-log /out/server.log
"""
import argparse
import http.client
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

LO, HI = 1000, 150000  # MiniMax-M3 tokenizer 的普通文本 id 段，避开特殊 / 控制 token
BLOCK = 128


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8890")
    ap.add_argument("--model", default="/models/MiniMax-M3-MXFP4")
    ap.add_argument("--bs", type=int, required=True, help="decode batch")
    ap.add_argument("--ctx", type=int, default=192038, help="prompt tokens per decode request (not a block multiple)")
    ap.add_argument("--prefill-cached", type=int, default=157312, help="cached prefix of the prefill (block multiple)")
    ap.add_argument("--prefill-new", type=int, default=2763, help="new tokens of the prefill")
    ap.add_argument("--ref-s", type=float, default=5.0, help="seconds of decode measured without the profiler")
    ap.add_argument("--pre-s", type=float, default=2.0, help="seconds of decode in the window before the prefill")
    ap.add_argument("--post-s", type=float, default=3.0, help="seconds of decode in the window after the prefill")
    ap.add_argument("--max-tokens", type=int, default=16384, help="per decode stream (ignore_eos)")
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--server-log", default=None, help="to check the scheduled prefill shapes")
    ap.add_argument("--summary", required=True)
    a = ap.parse_args()
    if a.ctx % BLOCK == 0:
        sys.exit(f"--ctx {a.ctx} is a multiple of {BLOCK}: the resend would have nothing left to prefill")
    if a.prefill_cached % BLOCK:
        sys.exit(f"--prefill-cached {a.prefill_cached} is not a multiple of {BLOCK}: part of it would be recomputed")
    U = urllib.parse.urlparse(a.url)
    rng = random.Random(a.seed)
    S = {"params": {k: getattr(a, k) for k in ("bs", "ctx", "prefill_cached", "prefill_new", "ref_s", "pre_s", "post_s",
                                               "max_tokens", "seed")}}

    def log(msg):
        print(f"{time.strftime('%F %T')} {msg}", flush=True)

    def ids(n):
        return [rng.randrange(LO, HI) for _ in range(n)]

    def post(path, body=None, timeout=3600):
        data = json.dumps(body).encode() if body is not None else b""
        req = urllib.request.Request(a.url + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        return json.loads(raw) if raw.strip() else {}

    field = ["prompt"]

    def payload(prompt, max_tokens, stream):
        return {"model": a.model, field[0]: prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
                "stream": stream}

    def complete(prompt, max_tokens):
        t0 = time.time()
        post("/v1/completions", payload(prompt, max_tokens, False))
        return t0, time.time()

    def log_pos():
        return os.path.getsize(a.server_log) if a.server_log and os.path.exists(a.server_log) else 0

    def scheduled(since):
        if not a.server_log or not os.path.exists(a.server_log):
            return []
        with open(a.server_log, errors="replace") as f:
            f.seek(since)
            return [ln.strip()[:240] for ln in f.read().splitlines() if "Scheduled prefill batch" in ln]

    class Stream(threading.Thread):
        def __init__(self, prompt):
            super().__init__(daemon=True)
            self.prompt, self.t, self.err = prompt, [], None
            self.first = threading.Event()

        def run(self):
            try:
                conn = http.client.HTTPConnection(U.hostname, U.port or 80, timeout=3600)
                conn.request("POST", "/v1/completions", json.dumps(payload(self.prompt, a.max_tokens, True)),
                             {"Content-Type": "application/json"})
                r = conn.getresponse()
                if r.status != 200:
                    raise RuntimeError(f"http {r.status}: {r.read(300)!r}")
                while True:
                    ln = r.readline()
                    if not ln:
                        break
                    if ln.startswith(b"data: ") and not ln.startswith(b"data: [DONE]"):
                        self.t.append(time.time())
                        self.first.set()
                if not self.t:
                    raise RuntimeError("stream closed before the first chunk")
            except Exception as e:  # noqa: BLE001 -- reported in the summary
                self.err = repr(e)[:300]
            finally:
                self.first.set()

    def step_stats(streams, lo, hi):
        """同一流相邻 chunk 的间隔；同一步内的多个 chunk（间隔 < 1.5 ms）不算一步。"""
        gaps, chunks = [], 0
        for s in streams:
            ts = [t for t in list(s.t) if lo <= t <= hi]
            chunks += len(ts)
            gaps += [y - x for x, y in zip(ts, ts[1:]) if y - x >= 0.0015]
        if not gaps:
            return None
        gaps.sort()
        q = lambda p: 1e3 * gaps[int(p * (len(gaps) - 1))]
        return {"median_ms": 1e3 * statistics.median(gaps), "p10_ms": q(0.1), "p90_ms": q(0.9), "n_gaps": len(gaps),
                "chunks": chunks, "seconds": hi - lo}

    log(f"args {S['params']}")
    warm = ids(2048)
    try:
        complete(warm, 16)
    except urllib.error.HTTPError as e:
        log(f"the server rejects a token-id list as 'prompt' (http {e.code}: {e.read(300)!r}); trying 'prompt_token_ids'")
        field[0] = "prompt_token_ids"
        try:
            complete(warm, 16)
        except urllib.error.HTTPError as e2:
            log(f"ABORT: 'prompt_token_ids' rejected too (http {e2.code}: {e2.read(300)!r}); this server cannot take "
                "token ids, so the prompt lengths cannot be fixed")
            sys.exit(1)
    S["prompt_field"] = field[0]
    log(f"warm request done (token ids sent as '{field[0]}')")

    t0 = time.time()
    prompts = [ids(a.ctx) for _ in range(a.bs)]
    for i, p in enumerate(prompts):
        s0, s1 = complete(p, 1)
        log(f"decode prompt {i + 1}/{a.bs}: {a.ctx} tokens prefilled in {s1 - s0:.1f} s")
    prefix = ids(a.prefill_cached)
    s0, s1 = complete(prefix, 1)
    log(f"prefill prefix: {a.prefill_cached} tokens prefilled in {s1 - s0:.1f} s")
    S["prewarm_s"] = time.time() - t0

    pos_dec = log_pos()
    streams = [Stream(p) for p in prompts]
    t_go = time.time()
    for s in streams:
        s.start()
    for s in streams:
        s.first.wait(1800)
    bad = [s.err for s in streams if s.err]
    if bad:
        log(f"ABORT: {len(bad)} decode streams failed, first: {bad[0]}")
        S["stream_errors"] = bad
        json.dump(S, open(a.summary, "w"), indent=1)
        os._exit(1)
    t_all = time.time()
    log(f"all {a.bs} streams decoding {t_all - t_go:.2f} s after sending")
    S["decode_start"] = {"sent": t_go, "all_first_chunk": t_all, "scheduled": scheduled(pos_dec)[:6]}

    time.sleep(1.0)
    r0 = time.time()
    time.sleep(a.ref_s)
    S["reference"] = {"step": step_stats(streams, r0, time.time())}
    pos_ref = log_pos()
    p0, p1 = complete(prefix + ids(a.prefill_new), 1)
    S["reference"].update(prefill_sent=p0, prefill_returned=p1, prefill_ms=1e3 * (p1 - p0),
                          scheduled=scheduled(pos_ref)[:4])
    st = S["reference"]["step"] or {}
    log(f"reference (no profiler): step p50 {st.get('median_ms', float('nan')):.2f} ms over {st.get('n_gaps', 0)} gaps; "
        f"prefill {1e3 * (p1 - p0):.1f} ms")

    time.sleep(1.0)
    pos_win = log_pos()
    w0 = time.time()
    post("/start_profile", timeout=120)
    w1 = time.time()
    time.sleep(a.pre_s)
    q0, q1 = complete(prefix + ids(a.prefill_new), 1)
    time.sleep(a.post_s)
    w2 = time.time()
    alive = sum(s.is_alive() for s in streams)  # 流只会结束不会新开：窗口结束时都在，窗口里每步就都是 bs 个
    post("/stop_profile", timeout=1800)
    w3 = time.time()
    S["window"] = {"start_profile_sent": w0, "start_profile_returned": w1, "prefill_sent": q0, "prefill_returned": q1,
                   "prefill_ms": 1e3 * (q1 - q0), "stop_profile_sent": w2, "stop_profile_returned": w3,
                   "streams_alive": alive, "step": step_stats(streams, w1, w2), "scheduled": scheduled(pos_win)[:4]}
    st = S["window"]["step"] or {}
    log(f"window (profiler on): step p50 {st.get('median_ms', float('nan')):.2f} ms over {st.get('n_gaps', 0)} gaps; "
        f"prefill {1e3 * (q1 - q0):.1f} ms; stop_profile took {w3 - w2:.1f} s; "
        f"streams alive at the window end {alive}/{a.bs}")

    want = f"1 reqs, {a.prefill_new} new tokens (done: [{a.prefill_cached}]"
    S["prefill_shape_ok"] = {k: any(want in ln for ln in S[k]["scheduled"]) for k in ("reference", "window")}
    S["stream_errors"] = [s.err for s in streams if s.err]
    json.dump(S, open(a.summary, "w"), indent=1)
    log(f"prefill shape '{want}' in the server log: {S['prefill_shape_ok']}; stream errors {len(S['stream_errors'])}")
    if alive < a.bs:
        log(f"WARN: only {alive} of {a.bs} streams were still decoding at the window end (does the server honour "
            "ignore_eos?): the window's decode batch is not fixed")
    log("DONE")
    sys.stdout.flush()
    os._exit(0)  # 流还开着；capture_agentx.sh 接着停服务


if __name__ == "__main__":
    main()
