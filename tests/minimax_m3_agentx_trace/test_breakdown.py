"""m3_prefill / m3_decode 端到端：模块总量精确等于 busy、少数层 kernel 单列、代表层、宽表和 picks、配对检查。"""

import csv
import json

import pytest

import m3_combine_wide
import m3_decode
import m3_prefill
from m3_synth import NL, PREFILL_NAME


def _rows(p):
    return list(csv.DictReader(open(p)))


def test_prefill_totals_reconcile_exactly_and_minority_kernels_are_kept(window_trace, m3_cfg, tmp_path, capsys):
    out = tmp_path / "bd"
    assert m3_prefill.main([str(window_trace), str(m3_cfg), str(out)]) == 0
    con = capsys.readouterr().out
    assert "check vs busy: diff 0.0 us" in con
    meta = json.load(open(out / "prefill_meta.json"))
    assert meta["annotation"] == PREFILL_NAME
    summ = _rows(out / "prefill_module_summary.csv")
    assert sum(float(r["rep_forward_ms"]) for r in summ) == pytest.approx(meta["busy_ms"], abs=1e-3)
    assert meta["model_ms"] == pytest.approx(meta["busy_ms"], rel=0.01)
    assert meta["unclassified_share"] < 0.02
    extras = _rows(out / "prefill_extras.csv")
    layer_extras = {(r["layer_idx"], r["module"]) for r in extras if r["where"].startswith("layer")}
    assert ("1", "sparse_attn_plan") in layer_extras and ("2", "aux_hidden_clone") in layer_extras
    detail = _rows(out / "prefill_detail.csv")
    assert {r["variant"] for r in detail} == {"full+mlp", "sparse+moe"}
    rep = {r["variant"]: r["layer_idx"] for r in detail}
    assert rep["sparse+moe"] == "3"            # 只有第 3 层只含共有模块（第 1 层有 plan，第 2 层有 clone）
    side = [r for r in detail if r["overlap"] == "yes"]
    assert side and all(r["ratio"].startswith("0") for r in side)
    gemm = next(r for r in detail if r["module"] == "qkv_idx_proj")
    assert (gemm["m"], gemm["k"], gemm["n"]) == ("2517", "6144", "2560") and gemm["note"].startswith("derived")
    hdr = next(csv.reader(open(out / "prefill_wide.csv")))
    assert hdr[0].startswith(PREFILL_NAME + " @UI 00:00:00.") and "layer#" in hdr[0]
    picks = (out / "prefill_picks.txt").read_text()
    assert picks.startswith("prefill annotation: " + PREFILL_NAME) and " UI " in picks


def test_decode_rows_sum_to_layer_union_and_pairing_is_checked(window_trace, m3_cfg, tmp_path, capsys):
    out = tmp_path / "bd"
    assert m3_decode.main([str(window_trace), str(m3_cfg), str(out)]) == 0
    con = capsys.readouterr().out
    assert "pairing looks intact" in con
    meta = json.load(open(out / "decode_meta.json"))
    assert meta["key"] == "decode[bs=3 tok=12 d=3 spec=3]" and not meta["scrambled"]
    assert meta["model_ms"] == pytest.approx(meta["union_ms"], rel=0.02)
    detail = _rows(out / "decode_detail.csv")
    for v in ("full+mlp", "sparse+moe"):
        rows = [r for r in detail if r["variant"] == v]
        assert sum(float(r["ratio"].rstrip("%")) for r in rows) == pytest.approx(100.0, abs=0.1)
    assert {r["module"] for r in detail} >= {"input_norm", "post_attn_norm", "full_attn_core_reduce",
                                             "moe_expert_gate_up", "comm_ar_moe_out"}
    timeline = _rows(out / "decode_rep_timeline.csv")
    assert len(timeline) == len(detail)


def test_scrambled_graph_pairing_is_flagged(build, m3_cfg, tmp_path, capsys):
    t = build(scramble=True)
    out = tmp_path / "bd"
    assert m3_decode.main([str(t), str(m3_cfg), str(out)]) == 0
    assert "SCRAMBLED" in capsys.readouterr().out
    meta = json.load(open(out / "decode_meta.json"))
    assert meta["scrambled"]
    detail = _rows(out / "decode_detail.csv")
    assert all(r["tflops"] == "" for r in detail) and all("pairing unverified" in r["note"] for r in detail)
    m3_prefill.main([str(t), str(m3_cfg), str(out)])
    assert m3_combine_wide.main([str(out), str(out / "wide.csv"), "M3 test"]) == 0
    title = next(csv.reader(open(out / "wide.csv")))[0]
    assert title.startswith("M3 test.") and "SCRAMBLED" in title


def test_both_phases_map_the_same_modules_except_the_documented_fusions(window_trace, m3_cfg, tmp_path):
    out = tmp_path / "bd"
    m3_prefill.main([str(window_trace), str(m3_cfg), str(out)])
    m3_decode.main([str(window_trace), str(m3_cfg), str(out)])
    p = set(json.load(open(out / "prefill_meta.json"))["module_set"])
    d = set(json.load(open(out / "decode_meta.json"))["module_set"])
    assert p - d <= {"attn_prefix_kv_gather", "attn_out_misc", "logits_row_select", "moe_expert_scale_shuffle",
                     "misc_small", "input_norm", "qkv_proj_quant"}
    assert d - p <= {"input_norm", "post_attn_norm", "final_norm", "full_attn_core_reduce", "qkv_idx_proj_quant",
                     "dense_gate_up_quant", "aux_hidden_copy"}


def test_prefill_without_annotation_exits_with_a_message(tmp_path, m3_cfg):
    import gzip
    p = tmp_path / "empty.pt.trace.json.gz"
    with gzip.open(p, "wt") as f:
        json.dump({"traceEvents": [{"ph": "X", "cat": "kernel", "name": "k", "ts": 1.0, "dur": 1.0, "pid": 0, "tid": 1}]}, f)
    with pytest.raises(SystemExit, match="prefill"):
        m3_prefill.main([str(p), str(m3_cfg), str(tmp_path / "o")])
    assert NL == 4
