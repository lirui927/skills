"""m3_common：config 推导、annotation 解析、kernel 名字整理、UI 时间、按 attention core 切层和按位置定角色。"""

import collections
import json

import pytest

import m3_common as mc
from m3_synth import NL, PREFILL_NAME, T0, m3_config


def _cfg(tmp_path, nl=60, cp=False):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(m3_config(nl)))
    return mc.M3Config(p, tp=4, indexer_cp=cp)


def test_layer_types_come_from_the_config(tmp_path):
    c = _cfg(tmp_path)
    assert c.variant_counts() == collections.Counter({"full+mlp": 1, "sparse+moe": 59})
    assert c.variants()[:2] == ["full+mlp", "sparse+moe"]
    assert c.sparse_keys == 16 * 128


def test_derived_gemm_shapes_at_tp4_match_reference_md(tmp_path):
    kn = _cfg(tmp_path).gemm_kn()
    assert {m: v[:2] for m, v in kn.items()} == {
        "qkv_proj": (6144, 2304), "qkv_idx_proj": (6144, 2560), "o_proj": (2048, 6144),
        "dense_gate_up": (6144, 6144), "dense_down": (3072, 6144), "moe_router": (6144, 128),
        "lm_head": (6144, 50016)}
    assert _cfg(tmp_path, cp=True).gemm_kn()["qkv_idx_proj"][:2] == (6144, 2944)


def test_causal_pairs_from_annotation_fields(tmp_path):
    # 一个请求：上下文 100，最后 4 个位置是 query：97 + 98 + 99 + 100 = 394 对
    full, sparse, _ = mc.attn_pairs("decode[bs=1 tok=4 d=1 spec=3 sqsq=16 sqsk=400 sk=100]", _cfg(tmp_path))
    assert full == 394
    assert sparse == 394          # 上下文比 2048 短：稀疏层也看全部 key
    f = mc.ann_fields(PREFILL_NAME)
    full, sparse, _ = mc.attn_pairs(PREFILL_NAME, _cfg(tmp_path))
    assert full == pytest.approx(f["sqsk"] - (f["sqsq"] - f["tok"]) / 2)
    assert full == pytest.approx(f["nctx"], rel=1e-3)     # ATOM 自己算的 nctx 就是这个数
    assert sparse == f["tok"] * 2048


def test_annotation_shape_key_drops_per_step_fields():
    assert mc.ann_key("decode[bs=3 tok=12 d=3 spec=3 sqsq=48 sqsk=2681952 sk=670488]") == "decode[bs=3 tok=12 d=3 spec=3]"


def test_short_keeps_the_pytorch_template_tag():
    s = mc.short("void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int)")
    assert s.startswith("at::native::vectorized_elementwise_kernel") and "FillFunctor" in s


def test_ui_time_counts_from_the_trace_event(window_trace):
    T = mc.Trace(window_trace)
    assert T.ui_src.startswith("this file's 'Trace'")
    assert T.ui(T0 + 3723_456_789.123) == "01:02:03.456789123"
    assert T.epoch_ns(T0) == int(1790857026000000000 + T0 * 1000)


def test_memcpy_htod_is_dropped_and_main_stream_is_the_busiest(window_trace):
    T = mc.Trace(window_trace)
    assert T.main_tid == 7
    assert not any(k["name"].startswith("Memcpy HtoD") for k in T.K + T.SIDE)
    a = T.forwards("prefill")[0]
    assert sum(k["dur"] for k in T.main_window(a)) <= a["dur"]


def test_segment_into_config_layers_and_roles_by_position(window_trace, m3_cfg):
    T, cfg = mc.Trace(window_trace), mc.M3Config(m3_cfg)
    for a in T.forwards("prefill") + T.forwards("decode")[:2]:
        seg, why = mc.segment(T.main_window(a), cfg.nl)
        assert why == "ok", why
        assert [mc.observed_variant(s) for s in seg["layers"]] == cfg.variants()
        head, rows, tail = mc.classify_forward(seg, cfg)
        mods = [[r["m"] for r in L] for L in rows]
        # 第一个 GEMM 是 qkv（稀疏层带 indexer），core 之后第一个是 o_proj，MoE 块里 expert 之前的是 router
        gemms = [[r["m"] for r in L if r["role"] == "gemm"] for L in rows]
        assert gemms[0] == ["qkv_proj", "o_proj", "dense_gate_up", "dense_down"]
        for g in gemms[1:]:
            assert g == ["qkv_idx_proj", "o_proj", "moe_router"]
        for L in mods[1:]:
            assert L.index("o_proj") > L.index("sparse_attn_core")
            assert L.index("moe_router") < L.index("moe_expert_gate_up") < L.index("moe_expert_down")
        assert "sparse_attn_plan" in mods[1] and "sparse_attn_plan" not in mods[3]
        assert "aux_hidden_clone" in mods[2]
        assert "embedding" in [r["m"] for r in head] and "comm_ar_embedding" in [r["m"] for r in head]
        tm = [r["m"] for r in tail]
        assert "lm_head" in tm and tm[-1] == "comm_ag_logits"
        unc = {"misc_small", "gemm_unmapped", "act_quant", "comm_other", "head_misc", "tail_misc"}
        everything = [r["m"] for r in head + [x for L in rows for x in L] + tail]
        assert len([m for m in everything if m in unc]) <= 1    # 只有第 1 层 plan 前的那个 fill


def test_segment_rejects_a_forward_missing_a_core(window_trace, m3_cfg):
    T, cfg = mc.Trace(window_trace), mc.M3Config(m3_cfg)
    w = [k for k in T.main_window(T.forwards("prefill")[0]) if "fmha_bf16" not in k["name"]]
    seg, why = mc.segment(w, cfg.nl)
    assert seg is None and f"!= {NL}" in why


def test_load_peaks_ignores_comment_keys(tmp_path):
    p = tmp_path / "peaks.json"
    p.write_text(json.dumps({"_comment": "x", "bf16": 10, "fp8": 20.5}))
    assert mc.load_peaks(p) == {"bf16": 10.0, "fp8": 20.5}
