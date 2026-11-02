# 参考：切层、模块映射、推导形状、输出列

## MiniMax-M3 的层结构（来自 config）

| 字段 | 值 | 用途 |
|---|---|---|
| `num_hidden_layers` | 60 | 每个 forward 的 attention core 数 |
| `sparse_attention_config.sparse_attention_freq` | 前 3 层 0，其余 1 | 0 = 全注意力，1 = indexer 稀疏注意力 |
| `moe_layer_freq` | 前 3 层 0，其余 1 | 0 = dense MLP，1 = MoE |
| `hidden_size` / `num_attention_heads` / `num_key_value_heads` / `head_dim` | 6144 / 64 / 4 / 128 | 投影的 K/N、attention FLOPs |
| `dense_intermediate_size` / `intermediate_size` | 12288 / 3072 | dense MLP / 每个 expert |
| `num_local_experts` / `num_experts_per_tok` / `n_shared_experts` | 128 / 4 / 1 | expert GEMM 的 M |
| `sparse_num_index_heads` / `sparse_index_dim` | 4 / 128 | `qkv_idx_proj` 的 N |
| `sparse_topk_blocks` x `sparse_block_size` | 16 x 128 = 2048 | 稀疏层每个 query 最多看的 key 数 |
| `vocab_size` | 200064 | `lm_head` 的 N（按 TP 切） |

于是只有两种层：`full+mlp`（第 0–2 层）和 `sparse+moe`（第 3–59 层）。

## 一个 forward 的结构（主 stream，GPU 时间序）

```
head:  embedding -> comm_ar_embedding
layer i（i = 0..59）:
  [input_norm]  -> qkv_proj | qkv_idx_proj  -> q/k 归一化 + rope + 写 KV cache  -> [indexer 打分 + topk]
  -> attention core（每层恰好一个，切层锚点） -> [core reduce] -> o_proj  -> comm_ar_attn_out
  -> [post_attn_norm] -> dense MLP（gate_up, act, down） | MoE（router, topk, route, quant, gate_up, down, combine）
  -> comm_ar_mlp_out | comm_ar_moe_out          <- 层边界在这组通信结束处
tail:  [final_norm] -> aux_hidden_copy x N -> [logits_row_select] -> lm_head -> comm_ag_logits
```

方括号里的步骤只在某一阶段单独出现：prefill 的 all-reduce 融合了 RMSNorm（`ar_rmsnorm_*`），所以
prefill 没有单独的 `post_attn_norm`、`final_norm`，第 1 层起也没有单独的 `input_norm`；decode 用 `ar_ll_*`，
norm 和 quant 是单独的 kernel。

验证时切出来的模块顺序（每行一个 kernel，同一模块连续多行合并）：

| prefill `full+mlp` | prefill `sparse+moe` | decode `full+mlp` | decode `sparse+moe` |
|---|---|---|---|
| qkv_proj | qkv_idx_proj | input_norm | input_norm |
| attn_qk_norm_rope_cache | attn_qk_idx_norm_rope_cache | qkv_proj_quant | qkv_idx_proj_quant |
| attn_prefix_kv_gather | indexer_score | qkv_proj（+ split-K reduce） | qkv_idx_proj（+ split-K reduce） |
| full_attn_core | indexer_topk | attn_qk_norm_rope_cache | attn_qk_idx_norm_rope_cache |
| attn_out_misc | sparse_attn_core | full_attn_core | indexer_score |
| o_proj_quant | o_proj_quant | full_attn_core_reduce | indexer_topk |
| o_proj | o_proj | o_proj_quant | sparse_attn_core |
| comm_ar_attn_out（6 个 kernel） | comm_ar_attn_out（6 个 kernel） | o_proj | o_proj_quant |
| dense_gate_up | moe_router | comm_ar_attn_out | o_proj |
| dense_act | moe_topk | post_attn_norm | comm_ar_attn_out |
| dense_down_quant | moe_route | dense_gate_up_quant | post_attn_norm |
| dense_down | moe_expert_act_quant | dense_gate_up（+ split-K reduce） | moe_router |
| comm_ar_mlp_out（6 个 kernel） | moe_expert_scale_shuffle | dense_act | moe_topk |
| | moe_expert_gate_up | dense_down_quant | moe_route |
| | moe_expert_down | dense_down | moe_expert_act_quant |
| | moe_combine | comm_ar_mlp_out | moe_expert_gate_up |
| | comm_ar_moe_out（6 个 kernel） | | moe_expert_down |
| | | | moe_combine |
| | | | comm_ar_moe_out |

**两阶段模块集合的预期差异**（第 10 条）：只在 prefill 出现的是 `attn_prefix_kv_gather`（有前缀缓存时把已有 KV
gather 出来给全注意力用）、`attn_out_misc`（full 层 core 之后的一次拷贝）、`logits_row_select`（只取每个序列
最后位置的 logits）、`moe_expert_scale_shuffle`（decode 把它融合进了 expert 的 quant）、`misc_small`（第 3 层
plan 之前的一个 fill）；只在 decode 出现的是 `input_norm`/`post_attn_norm`/`final_norm` 和各 `*_quant`（prefill
融合进了 all-reduce）以及 `full_attn_core_reduce`（split-KV 的 reduce）。除此之外的差异都当映射缺口处理。

## kernel 名字 -> 模块（`m3_common.py`）

名字只用来认 kernel 的类型，角色由位置决定（第 4 条）：

| 类型（正则，节选） | 位置规则 | 模块 |
|---|---|---|
| attention core：`fmha_bf16_pertoken`、`fmha_fwd`、`pa_decode_tile_kernel`、`paged_attention_decode` | 每层恰好一个；`bs16` 的 tile 是稀疏层 | `full_attn_core` / `sparse_attn_core` |
| `pa_decode_reduce` | core 之后 | `*_attn_core_reduce` |
| 通信：`ar_gfx1250`、`ar_ll_gfx1250`、`ar_rmsnorm`、`ag_gfx1250`、`start/end_sync_kernel`、`ncclDevKernel`、... | 紧跟的 `Memcpy DtoD` 并入同一组；core 后第一组 / 第二组 | `comm_ar_attn_out` / `comm_ar_mlp_out`、`comm_ar_moe_out`；头尾 `comm_ar_embedding` / `comm_ag_logits` |
| GEMM：`gemm`、`Cijk_`、`hipblaslt` | core 前第一个 / core 后第一个 / MoE 块 expert 前 / dense 第 1、2 个 / 尾部 | `qkv_proj`、`qkv_idx_proj` / `o_proj` / `moe_router` / `dense_gate_up`、`dense_down` / `lm_head` |
| `splitk_reduce` | 归到前一个 GEMM | 同前一个 GEMM |
| quant（名字含 `quant`） | 归到它后面第一个 GEMM / expert；遇到 core 或通信还没找到就是 `act_quant` | `<模块>_quant` |
| expert：`a8w4_tdm`、`fused_moe`、`ck_moe`、`moe_gemm`、`gemm1_a4w4`、`gemm2_a4w4`、... | 名字含 `_K<hidden>_` 是 gate_up，否则 down | `moe_expert_gate_up` / `moe_expert_down` |
| `topk_gating`、`topk_softmax`、`grouped_topk` | MoE 块 | `moe_topk` |
| `moe_gather_reduce`、`moe_sum`、`moe_combine` | MoE 块 | `moe_combine` |
| indexer：`index_block_score`、`index_score`、`topk_index` | core 前 | `indexer_score`、`indexer_topk` |
| `pa_decode_plan` | 第一个稀疏层 core 前 | `sparse_attn_plan` |
| `triton_poi_fused_clone`、`Memcpy DtoD`、`__amd_rocclr_copyBuffer` | 层内（EAGLE3 aux 层）/ 尾部 | `aux_hidden_clone` / `aux_hidden_copy` |
| rmsnorm / layernorm | core 前第一个 GEMM 之前 / MLP 块 GEMM 之前 / 尾部 | `input_norm` / `post_attn_norm` / `final_norm` |

未归类的去处：`misc_small`、`attn_out_misc`、`gemm_unmapped`、`act_quant`、`comm_other`、`head_misc`、
`tail_misc`。它们合计应 <= 2% of busy。

模块 -> 组（`GROUPS`）：`comm`、`attention_core`、`indexer`（含 plan）、`attention_proj`（qkv、o_proj、
q/k 归一化和 rope、prefix gather）、`moe_routed_shared_experts`、`dense_mlp`、`norm`、`embedding_lm_head`、
`eagle3_aux`、`other`。注意 prefill 的 `comm` 组里含融合进 all-reduce 的 RMSNorm。

## 推导形状（TP4，`M3Config.gemm_kn`）

| 模块 | K | N | dtype（权重/激活） |
|---|---|---|---|
| `qkv_proj` | hidden 6144 | (64/4 + 2 x 4/4) x 128 = 2304 | fp8 / fp8 |
| `qkv_idx_proj` | 6144 | 2304 + 每 rank index q + index k = 2560（`ATOM_M3_INDEXER_CP=1`：2944） | fp8 / fp8 |
| `o_proj` | 64/4 x 128 = 2048 | 6144 | fp8 / fp8 |
| `dense_gate_up` | 6144 | 2 x 12288 / 4 = 6144 | fp8 / fp8 |
| `dense_down` | 12288 / 4 = 3072 | 6144 | fp8 / fp8 |
| `moe_router` | 6144 | 128 | bf16 |
| `lm_head` | 6144 | 200064 / 4 = 50016 | bf16 |
| `moe_expert_gate_up` | 6144 | 2 x 3072 / 4 = 1536 | fp4 / fp4 |
| `moe_expert_down` | 3072 / 4 = 768 | 6144 | fp4 / fp4 |

M：GEMM 取 annotation 的 `tok`；expert 取 `tok x (topk + 1)`（含融合进来的 shared expert）。
`tflops = 2·M·K·N / time`；字节数 = 激活 + 权重 + 输出（fp4 按 0.5 字节 + 每 32 个一个 scale 字节）。

attention：全注意力 query-key 对数 = `sqsk - (sqsq - tok)/2`（annotation 字段，见第 21 条），稀疏层取
`min(该值, tok x 2048)`；`FLOPs = 4 x heads/TP x head_dim x 对数`。decode 的全注意力 core 另给 KV 读取带宽
（`sk x kv_heads/TP x head_dim x 2 x 1 字节`，fp8 KV）。

## annotation 字段（ATOM `ATOM_ENABLE_DETAILED_ANNOTATION=1`）

- `prefill[bs= tok= ctx= nctx=X.YM sqsq= sqsk= sk=]`：`tok` 本次 token 数，`nctx` 全注意力对数（百万），
  `sqsq = Σnq²`、`sqsk = Σnq·nkv`、`sk = Σnkv`。
- `decode[bs= tok= d= spec= sqsq= sqsk= sk=]`：`tok = bs x (1 + spec)`，`d` 解码序列数；形状键只取
  `bs/tok/d/spec`（`ann_key`）。
- 同一主 stream 上还有 `propose_eagle[i/3 ...]`（drafter）、`ATOM::prepare_model`、`ATOM::postprocess`、
  `nccl:broadcast`，`m3_window.py` 用它们拆 step 周期。

## 时间

- Perfetto UI 时间 `HH:MM:SS.nnnnnnnnn` = 事件 ts - profiler 的 `Trace` 事件 ts（精简后的 trace 保留了这个事件；
  没有时脚本退到最早事件并提示用 `--ui-zero-us`）。
- epoch 纳秒 = trace 顶层 `baseTimeNanoseconds` + ts x 1000，用来和 aiperf 记录（`request_start_ns`）、
  `capture_meta.json` 的时间点对齐。

## 输出列（`*_detail.csv`）

`phase, annotation, ann_ts_us, variant, layer_idx, layer_t0_us, layer_t1_us, ann_ui, layer_ui, layer_count,
kernel_ts_us, kernel_ui, group, module, kernel, op, stream, overlap, overlapped_us, time_us, ratio, m, k, n,
dtype, tflops, tbytes_s, note`；decode 多一列 `critical_us`（关键路径部分，各行加起来 = 层的 union busy），
`ratio` 按它算。`note` 写明 shape 和吞吐从哪来（recorded / derived / 超峰值被留空的原因）。

## 移植到别的模型

1. 从 config 读层的分类，不从 trace 推。
2. 按 GPU 时间序 dump 一个 forward 的 kernel 序列，先读结构，再写规则。
3. 数一个 forward 里的结构锚点（每层一个的 core、相邻 core 间的通信组数），和 config 对上，这就是切层的证明。
4. 填 core / 通信 / expert / indexer 的正则、位置规则和推导形状表（`m3_common.py`）。
5. 过一遍 `pitfalls.md`：大部分是引擎层面的，换模型照样成立。
