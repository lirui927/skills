"""合成的 MiniMax-M3 ATOM trace（测试数据生成器），conftest 和各测试共用。

config 用 M3 的真实维度，但只有 4 层：第 0 层 full+mlp，第 1-3 层 sparse+moe（第 1 层是第一个稀疏层，带 plan；
第 2 层是 EAGLE3 aux 层，带 clone）。kernel 名字取自 gfx1250 上 ATOM + aiter 的真实 trace，按它们在 forward 里的
顺序排在主 stream 上；一个 prefill、若干 decode step、每步 3 次 propose_eagle、postprocess，另有一条 side stream、
一个横跨计算的 Memcpy HtoD，以及 profiler 的 'Trace' 事件和 baseTimeNanoseconds。
"""

import gzip
import json
import random
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "minimax-m3-agentx-trace"

NL = 4
BASE_NS = 1790857026000000000
T0 = 2491983241667.0          # profiler 'Trace' 事件的 ts（us）
MAIN, SIDE = 7, 9

QUANT = "_ZN5aiter37dynamic_per_token_scaled_quant_kernelIDF16bDB8_EEvPT0_"
AR_PREFILL = [("Memcpy DtoD", 3.0), ("aiter::start_sync_kernel", 5.0), ("aiter::ar_rmsnorm_reduce_scatter_gfx1250", 20.0),
              ("aiter::end_sync_kernel", 2.0), ("_ZN5aiter18ar_rmsnorm_gfx1250IDF16bDB8_Li4ELb1ELb1EEEvPNS_", 20.0),
              ("aiter::end_sync_kernel", 2.0)]
AR_DECODE = [("aiter::ar_ll_gfx1250", 8.0)]


def m3_config(nl=NL):
    sparse = [0] + [1] * (nl - 1)
    return {"architectures": ["MiniMaxM3ForConditionalGeneration"],
            "text_config": {"num_hidden_layers": nl, "hidden_size": 6144, "num_attention_heads": 64,
                            "num_key_value_heads": 4, "head_dim": 128, "intermediate_size": 3072,
                            "dense_intermediate_size": 12288, "shared_intermediate_size": 3072,
                            "num_local_experts": 128, "num_experts_per_tok": 4, "n_shared_experts": 1,
                            "vocab_size": 200064, "moe_layer_freq": list(sparse),
                            "sparse_attention_config": {"use_sparse_attention": True, "sparse_index_dim": 128,
                                                        "sparse_num_index_heads": 4, "sparse_topk_blocks": 16,
                                                        "sparse_block_size": 128,
                                                        "sparse_attention_freq": list(sparse)}}}


def prefill_layer(i):
    if i == 0:  # full + dense MLP；第 0 层自己有 input norm
        return ([("_ZN5aiter35fused_qk_rmsnorm_group_quant_kernelIDF16bEEvv", 9.0), (QUANT, 4.0),
                 ("gemm_a8w8_ptpc_t256x192x128_mw2_nw2_nb4_sk1_cm1_cn1", 30.0),
                 ("_fused_qkv_norm_rope_cache_kernel", 10.0), ("cp_mha_gather_cache_kernel", 5.0),
                 ("aiter::fmha_bf16_pertokenBf16_hd128_128x256_mask_varlen", 200.0), (QUANT, 4.0),
                 ("gemm_a8w8_ptpc_t256x256x128_mw2_nw2_nb4_sk1_cm1_cn1", 25.0)] + AR_PREFILL +
                [("gemm_a8w8_ptpc_t256x256x128_mw2_nw2_nb4_sk1_cm1_cn1", 50.0), ("_swiglu_oai_kernel", 8.0),
                 (QUANT, 4.0), ("gemm_a8w8_ptpc_t256x256x128_mw2_nw2_nb4_sk1_cm1_cn1", 40.0)] + AR_PREFILL)
    pre = [("triton_poi_fused_clone_0", 18.0)] if i == 2 else []
    plan = ([("at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >", 4.0),
             ("pa_decode_plan_kernel_0", 16.0)] if i == 1 else [])
    return (pre + [("gemm_a8w8_ptpc_t256x256x128_mw2_nw2_nb4_sk1_cm1_cn1", 37.0),
                   ("_ZN5aiter27fused_qknorm_idxrqknorm_ops27fusedQKNormIdxrQKNormEv", 12.0),
                   ("_index_block_score_kernel", 20.0), ("_topk_index_packed_kernel", 10.0)] + plan +
            [("pa_decode_tile_kernel_kvfp8_qbf16_d128_bs16_gqa16_mtp1_pertoken_direct", 70.0), (QUANT, 4.0),
             ("gemm_a8w8_ptpc_t256x256x128_mw2_nw2_nb4_sk1_cm1_cn1", 25.0)] + AR_PREFILL +
            [("_gemm_a16w16_gfx1250_bandwidth_bound_kernel_BLOCK_M_128_BLOCK_N_64", 10.0),
             ("aiter::topk_gating_kernel_opt", 5.0), ("moe_route", 6.0),
             ("moe_token_multidest_quant_fusepre_k5_fd6144_r8_fp4_pk8_hid", 8.0),
             ("moe_scatter_preshuffle_scale_b192_r8_k8_g", 4.0),
             ("a8w4_tdm_fp4_t128x256x256_w1x4_b3_K6144_e129_act2_q1r8_mg10", 72.0),
             ("a8w4_tdm_fp4_t128x256x256_w1x4_b3_K768_e129_mg10", 40.0),
             ("moe_gather_reduce_bf16_d6144_tk5_sk1_v4_wf32_frlds", 9.0)] + AR_PREFILL)


def decode_layer(i):
    if i == 0:
        return ([("aiter::add_rmsnorm_quant_kernel", 3.0), (QUANT, 2.0),
                 ("gemm_a8w8_ptpc_t16x128x256_mw1_nw4_nb3_sk8_cm1_cn1", 6.0), ("gemm_a8w8_splitk_reduce_bf16_sk8_u1", 2.0),
                 ("_fused_qkv_norm_rope_cache_kernel", 3.0),
                 ("pa_decode_tile_kernel_kvfp8_qbf16_d128_bs128_gqa16_mtp4_pertoken", 12.0),
                 ("pa_decode_reduce_kernel_obf16_d128_np256_gqa16_pairs_g16", 2.5), (QUANT, 2.0),
                 ("gemm_a8w8_ptpc_t16x64x512_mw1_nw4_nb4_sk1_cm1_cn1", 5.0)] + AR_DECODE +
                [("aiter::add_rmsnorm_quant_kernel", 3.0), (QUANT, 2.0),
                 ("gemm_a8w8_ptpc_t16x64x128_mw1_nw4_nb4_sk8_cm1_cn1", 7.0), ("gemm_a8w8_splitk_reduce_bf16_sk8_u1", 2.0),
                 ("_swiglu_oai_kernel", 2.0), (QUANT, 2.0), ("gemm_a8w8_ptpc_t16x64x1024_mw1_nw4_nb4_sk1_cm1_cn1", 6.0)]
                + AR_DECODE)
    pre = [("triton_poi_fused_clone_0", 1.8)] if i == 2 else []
    plan = [("pa_decode_plan_kernel_0", 3.6)] if i == 1 else []
    return ([("aiter::add_rmsnorm_quant_kernel", 3.0), (QUANT, 2.0)] + pre +
            [("gemm_a8w8_ptpc_t16x64x512_mw1_nw4_nb3_sk4_cm1_cn1", 6.5), ("gemm_a8w8_splitk_reduce_bf16_sk4_u1", 2.0),
             ("_ZN5aiter27fused_qknorm_idxrqknorm_ops27fusedQKNormIdxrQKNormEv", 3.0),
             ("_decode_index_score_tiled_kernel", 9.0), ("_topk_index_packed_kernel", 4.0)] + plan +
            [("pa_decode_tile_kernel_kvfp8_qbf16_d128_bs16_gqa16_mtp1_pertoken_cluster", 8.0), (QUANT, 2.0),
             ("gemm_a8w8_ptpc_t16x64x512_mw1_nw4_nb4_sk1_cm1_cn1", 5.0)] + AR_DECODE +
            [("aiter::add_rmsnorm_quant_kernel", 3.0),
             ("_gemm_a16w16_gfx1250_bandwidth_bound_kernel_BLOCK_M_16_BLOCK_N_32", 3.0),
             ("aiter::topk_gating_kernel_opt", 2.0), ("moe_route_psum_fused", 3.0),
             ("moe_fused_quant_preshuffle_routeks_fd6144_r1_fp4_pk8_srctk", 3.0),
             ("a8w4_tdm_fp4_t16x64x1024_w1x1_b3_K6144_e129_act2_q1r1_wpt1", 20.0),
             ("a8w4_tdm_fp4_t16x64x256_w1x2_b3_K768_e129_mg10_pw4096", 11.0),
             ("moe_gather_reduce_bf16_d6144_tk5_sk1_v2_wf32_frlds", 3.0)] + AR_DECODE)


PREFILL_HEAD = [("_masked_embedding_kernel", 5.0), ("Memcpy DtoD", 3.0), ("aiter::ar_gfx1250_naive_unroll4", 30.0)]
PREFILL_TAIL = [("Memcpy DtoD", 2.0), ("Memcpy DtoD", 2.0), ("Memcpy DtoD", 2.0),
                ("at::native::index_elementwise_kernel<128, 4, at::native::index_kernel_impl<...> >", 3.0),
                ("_gemm_a16w16_gfx1250_bandwidth_bound_kernel_BLOCK_M_16_BLOCK_N_64", 15.0),
                ("Memcpy DtoD", 2.0), ("aiter::ag_gfx1250_lastdim", 20.0)]
DECODE_HEAD = [("_masked_embedding_kernel", 2.0), ("aiter::ar_ll_gfx1250", 8.0)]
DECODE_TAIL = [("aiter::add_rmsnorm_quant_kernel", 3.0)] + [("__amd_rocclr_copyBuffer", 1.0)] * 4 + \
              [("_gemm_a16w16_gfx1250_bandwidth_bound_kernel_BLOCK_M_16_BLOCK_N_64", 9.0), ("Memcpy DtoD", 2.0),
               ("aiter::ag_gfx1250_lastdim", 60.0)]

PREFILL_NAME = "prefill[bs=1 tok=2517 ctx=105301 nctx=261.9M sqsq=6335289 sqsk=265042617 sk=105301]"


class Builder:
    def __init__(self, seed=0):
        self.ev = [{"ph": "X", "cat": "Trace", "name": "PyTorch Profiler (0)", "ts": T0, "dur": 0.0, "pid": "Spans",
                    "tid": "PyTorch Profiler"}]
        self.t = T0 + 1000.0
        self.rng = random.Random(seed)

    def kernel(self, name, dur, tid=MAIN, ts=None, gap=1.0):
        ts = self.t if ts is None else ts
        cat = "gpu_memcpy" if name.startswith("Memcpy") else "kernel"
        self.ev.append({"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "pid": 0, "tid": tid})
        if tid == MAIN and ts >= self.t:
            self.t = ts + dur + gap
        return ts

    def annotation(self, name, ts, dur, tid=MAIN):
        self.ev.append({"ph": "X", "cat": "gpu_user_annotation", "name": name, "ts": ts, "dur": dur, "pid": 0, "tid": tid})

    def forward(self, name, seq, jitter=0.0, scramble=False, transform=None):
        durs = [d * (1 + self.rng.uniform(-jitter, jitter)) for _, d in seq]
        if scramble:
            self.rng.shuffle(durs)
        if transform:
            durs = transform(seq, durs)
        t0 = self.t
        for (n, _), d in zip(seq, durs):
            self.kernel(n, d)
        self.annotation(name, t0 - 0.5, self.t - t0)
        return t0

    def finish(self, path):
        end = max(e["ts"] + e["dur"] for e in self.ev)
        self.ev[0]["dur"] = end - T0 + 100.0
        doc = {"schemaVersion": 1, "baseTimeNanoseconds": BASE_NS, "traceEvents": self.ev}
        with gzip.open(path, "wt") as f:
            json.dump(doc, f)
        return path


def prefill_seq(nl=NL):
    return PREFILL_HEAD + [k for i in range(nl) for k in prefill_layer(i)] + PREFILL_TAIL


def decode_seq(nl=NL):
    return DECODE_HEAD + [k for i in range(nl) for k in decode_layer(i)] + DECODE_TAIL


def build_window(path, n_decode=40, scramble=False, wait_us=0.0, truncate_last=False, seed=0, htod=True):
    """一个窗口：prefill（带一个横跨计算的 Memcpy HtoD 和 side stream）+ n_decode 个 decode step（每步后 3 次
    propose_eagle、一次 postprocess，side stream 上一个 DtoD）。wait_us > 0：每组通信之前的那个 kernel 短 wait_us、
    start_sync 长 wait_us（这个 rank 先到、在 barrier 里等），结束时间不变。"""
    b = Builder(seed)

    def wait(seq, durs):
        if not wait_us:
            return durs
        out = list(durs)
        for j, (n, _) in enumerate(seq):
            if n == "aiter::start_sync_kernel" and j >= 2:
                out[j - 2] -= wait_us
                out[j] += wait_us
        return out

    p0 = b.forward(PREFILL_NAME, prefill_seq(), transform=wait)
    if htod:
        b.kernel("Memcpy HtoD", 5000.0, ts=p0 + 50.0)
    b.kernel("side_stream_kernel", 15.0, tid=SIDE, ts=p0 + 400.0)
    b.t += 200.0
    seq = decode_seq()
    for s in range(n_decode):
        sk = 670000 + 40 * s
        name = f"decode[bs=3 tok=12 d=3 spec=3 sqsq=48 sqsk={4 * sk} sk={sk}]"
        if truncate_last and s == n_decode - 1:
            seq = seq[: len(seq) // 2]
        d0 = b.forward(name, seq, jitter=0.02, scramble=scramble, transform=wait)
        for i in range(3):
            t = b.t
            b.kernel(f"eagle_draft_step_{i}", 60.0)
            b.annotation(f"propose_eagle[{i}/3 tok=3 bs=3/3 graph]", t - 0.2, b.t - t)
        t = b.t
        b.kernel("postprocess_kernel", 20.0)
        b.annotation("ATOM::postprocess", t - 0.2, 25.0)
        b.kernel("Memcpy DtoD", 2.0, tid=SIDE, ts=t + 1.0)
        b.t += 5.0
    return b.finish(path)


def write_aiperf(d, T):
    """aiperf 的两个导出文件，按 trace T（m3_common.Trace）里第一个到最后一个 forward 的窗口摆请求：
    窗口前 650 s 开始的、与窗口重叠的（ITL 2.5 ms）、窗口前刚结束的。"""
    fw = T.forwards("prefill") + T.forwards("decode")
    e0, e1 = T.epoch_ns(fw[0]["ts"]), T.epoch_ns(fw[-1]["ts"] + fw[-1]["dur"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "profile_export_aiperf.json").write_text(json.dumps(
        {"inter_token_latency": {"unit": "ms", "avg": 0.9, "p50": 0.85, "p90": 1.2, "count": 3}}))
    recs = [(e0 - 650e9, e0 - 600e9, 0.80), (e0 - 100e9, e1 + 1e9, 2.50), (e0 - 50e9, e0 - 1e9, 0.82)]
    with open(d / "profile_export.jsonl", "w") as f:
        for a, b, itl in recs:
            f.write(json.dumps({"metadata": {"benchmark_phase": "profiling", "request_start_ns": int(a),
                                             "request_end_ns": int(b)},
                                "metrics": {"inter_token_latency": {"value": itl, "unit": "ms"},
                                            "output_sequence_length": {"value": 100}}}) + "\n")
