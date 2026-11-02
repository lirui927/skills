"""m3_trace_completeness（截断的 forward）、m3_window（step 周期、窗口内的 MTP 统计、aiperf 对账）、m3_rank_skew。"""

import json

import pytest

import m3_common as mc
import m3_rank_skew
import m3_trace_completeness
import m3_window
from m3_synth import write_aiperf


def test_completeness_passes_a_whole_window(window_trace, m3_cfg, capsys):
    assert m3_trace_completeness.main([str(window_trace), str(m3_cfg), "rank 0"]) == 0
    out = capsys.readouterr().out
    assert "RESULT: 41 of 41 forwards whole" in out
    assert "per decode->decode gap: [(3, 39)]" in out


def test_completeness_flags_a_cut_forward(build, m3_cfg, capsys):
    t = build(truncate_last=True)
    assert m3_trace_completeness.main([str(t), str(m3_cfg)]) == 2
    out = capsys.readouterr().out
    assert "ODD decode" in out and "LAST" in out and "RESULT: 40 of 41" in out


SERVER_LOG = """ARCH=gfx1250 CONC=10 env: ATOM_ENABLE_DETAILED_ANNOTATION=1 ATOM_PROFILER_MORE=0
[atom 08:19:00] [MTP Stats Interval] Average toks/fwd: 2.00, Accepted/Total Draft tokens: 1000/3000, x
[atom 08:30:00] Engine Core: executing utility command: start_profile
[atom 08:30:10] [MTP Stats Interval] Average toks/fwd: 2.80, Accepted/Total Draft tokens: 1800/3000, x
[atom 08:30:20] [MTP Stats Interval] Average toks/fwd: 2.80, Accepted/Total Draft tokens: 900/1500, x
[atom 08:30:25] Engine Core: executing utility command: stop_profile
[atom 08:30:40] Rank 0: trace exported to /out/traces/rank_0/{name} (100.0 MB, 14.0s)
[atom 08:30:41] Engine Core: utility command 'stop_profile' finished in 15.20s
[atom 08:31:00] [MTP Stats Interval] Average toks/fwd: 1.50, Accepted/Total Draft tokens: 500/3000, x
"""


def test_window_step_period_toks_per_fwd_and_aiperf(window_trace, tmp_path, capsys):
    log = tmp_path / "server.log"
    log.write_text(SERVER_LOG.format(name=window_trace.name))
    T = mc.Trace(window_trace)
    write_aiperf(tmp_path / "aiperf", T)
    out = tmp_path / "w.json"
    assert m3_window.main([str(window_trace), "--server-log", str(log), "--aiperf-dir", str(tmp_path / "aiperf"),
                           "--out", str(out)]) == 0
    con = capsys.readouterr().out
    S = json.load(open(out))
    # 只用窗口内两行：(1800 + 900) / (3000 + 1500) 的接受率，spec=3
    assert S["toks_per_fwd"] == pytest.approx(1 + 3 * 2700 / 4500)
    assert S["stop_profile_s"] == pytest.approx(15.2)
    parts = S["step_parts_ms"]
    assert sum(parts.values()) == pytest.approx(S["step_period_ms"], rel=0.02)
    assert parts["propose_eagle"] > 0
    assert S["itl_pred_decode_only_ms"] == pytest.approx(S["step_period_ms"] / S["toks_per_fwd"])
    assert S["itl_pred_window_ms"] > S["itl_pred_decode_only_ms"]      # 窗口里还有一个 prefill
    assert S["window_offset_s"] == pytest.approx(650, abs=1)
    assert "OUTSIDE" not in con
    assert S["aiperf_itl_requests_p50_ms"] == pytest.approx(2.5)


def test_rank_skew_aligns_on_end_times_and_finds_the_waiting_rank(build, capsys):
    a = build("a.pt.trace.json.gz")
    b = build("b.pt.trace.json.gz", wait_us=5.0)
    assert m3_rank_skew.main([str(a), str(b), "--kind", "prefill"]) == 0
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.strip().startswith("allreduce+rmsnorm_aiter"))
    f = line.split()
    calls, skew, later, sync_a, sync_b = int(f[1]), float(f[6]), float(f[8]), float(f[9]), float(f[10])
    assert calls == 2 * 4
    assert skew == pytest.approx(5.0, abs=0.2)      # A 到得晚 5 us
    assert later == 100
    assert sync_b == pytest.approx(sync_a + 5.0, abs=0.2)   # B 在 barrier 里等了这 5 us
    assert "clock offset (A - B, from END times) p50 0.0 us" in out
