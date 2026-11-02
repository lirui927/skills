# 踩坑清单

每条写现象、原因和处理。括号里是 trace-kernel-breakdown 原方法里对应的条目号，方便对照原文。
大多数是引擎层面的，换模型也成立；M3 特有的放在最后几条。

## 1. 只算主 stream，side stream 单列（原 1、30）

把所有 stream 的 kernel 时长加起来，会把被主干掩盖的工作也算成串行的。主 stream = GPU 事件最多的那个 tid。
其他 stream 的 kernel 按时间落进哪一层就列在哪一层，标 `overlap=yes`，只统计它没被主 stream 覆盖的部分。

M3 在这个栈上的 decode 是单 stream 的；side stream 上只有 postprocess 的 `Memcpy DtoD`（每步几次，和 forward
基本不重叠）。prefill 同样要查：`m3_prefill.py` 会打印 forward 内 side stream 的总量和未覆盖部分。换镜像后如果
shared expert 或 dense MLP 挪到了第二个 stream，这两行会变大，模块表也要跟着看 side 行。

## 2. annotation 去重（原 2）

每个 forward 在 trace 里有多份 annotation：CPU 侧 `user_annotation` 一份，每个 GPU stream 上
`gpu_user_annotation` 各一份。数 step 只用主 stream 上的 `gpu_user_annotation`，否则 step 数翻倍、周期变短。

## 3. kernel 时间不是 ITL（原 3）

EAGLE3 下一个 decode step 验证 `1 + spec` 个 token，每个序列平均出 `toks_per_fwd` 个：

```
ITL = step 周期 / toks_per_fwd
step 周期 = decode forward + drafter（propose_eagle x spec）+ prepare / broadcast / postprocess 中未重叠的部分
```

`toks_per_fwd` 用这个窗口内的 MTP 统计：server log 里 `start_profile` 和 `stop_profile` 之间的
`[MTP Stats Interval]` 行，按 draft token 数加权，`1 + spec x 接受数 / draft 数`。`ACCEPT_RATE` 固定时它约等于
`1 + 3 x 接受率`。

AgentX 的 chunked prefill 夹在 decode 之间，序列在 prefill forward 期间不出 token，于是

```
ITL = 窗口 / decode forward 数 / toks_per_fwd    # 不是 decode forward 本身的时间
```

`m3_window.py` 两个都算。step 周期用中位数：会话边界上偶尔一个很长的空闲会把均值拉偏很多。

## 4. 通用 GEMM 名字不带角色，按位置定（原 4）

`gemm_a8w8_ptpc_*`、`_gemm_a16w16_*` 这类名字在不同层、不同位置做不同的投影。把某个名字硬映射成某个角色，
会把后面的 GEMM 整体错一位。M3 的规则：core 之前第一个 GEMM 是 qkv（稀疏层是 `qkv_idx_proj`），core 之后第一个
是 `o_proj`；MoE 块里 expert 之前的 GEMM 是 router（bf16，输出 128 列）；dense MLP 依次是 gate_up、down；
forward 尾部唯一的 GEMM 是 `lm_head`。split-K 的 reduce kernel 归到它前面那个 GEMM。

## 5. 中位层会丢掉只在少数层出现的 kernel（原 5）

取中位层逐 kernel 输出时，只在一两层出现的 kernel 不在中位层里。M3 上有三类：

- 第一个稀疏层（第 3 层）的 `pa_decode_plan_kernel`（以及 prefill 里它前面的一个 `FillFunctor`）；
- EAGLE3 aux 层的 `triton_poi_fused_clone`：draft 取第 2、L/2、L-3 层的 hidden state（L=60 -> 2/30/57，
  draft config 里没写，是 EAGLE3 的默认取法）；
- 第 0 层自己的 input norm 和 quant：后面各层的 input norm 融合进了上一层的 all-reduce（prefill）。

处理：模块总时间按全部层精确加总；代表层从"只含同类层共有模块"的层里选；多出来的 kernel 写进 `*_extras.csv`
并在 reconcile 里单独加上。

## 6. 层类型从 config 读，用 trace 验证（原移植步骤 1、原 6）

不要从 trace 推层数和层类型。`sparse_attention_freq`（1 = indexer 稀疏注意力）和 `moe_layer_freq`（1 = MoE）
给出每层的类型；切出来的逐层类型（有没有 indexer / 稀疏 core，有没有 expert GEMM）必须和 config 一致。某类层
整个缺失，说明它的 core kernel 没被认出来，而不是这类层很便宜。

## 7. 切层锚点必须在每个 forward 里每层恰好一个（原 16）

M3 用 attention core 做锚点：prefill 的 full 层是 `fmha_*`，decode 的 full 层是 `pa_decode_tile_* bs128`，
稀疏层两阶段都是 `pa_decode_tile_* bs16`。相邻两个 core 之间恰好两组通信（attention 输出的 all-reduce 和
MLP/MoE 输出的 all-reduce），层边界取前一组结束处。第 0 层之前是 embedding 和它的 all-reduce；最后一层之后是
final norm、lm_head 和 logits 的 all-gather。`m3_common.segment` 任何一条不满足都会拒绝这个 forward 并说明原因。
换镜像后先看 `completeness.txt`，别直接看数字。

## 8. `Memcpy HtoD` 不算（原 27）

它挂在主 stream 上，但实际在 DMA 引擎上和计算并行，`dur` 还包含等 host 数据的时间。算进去会出现 busy 超过
wall。所有脚本一律丢掉它；`Memcpy DtoD` 保留（在 stream 上串行，比如通信前拷进注册缓冲区）。

## 9. 没录 shape 时，一切 shape 都是推导的（原 14、13）

`ATOM_PROFILER_MORE=0` 的 trace 没有 `Input Dims` / `Input type`：GEMM 的 M/K/N、dtype 以及据此算出的
TFLOP/s、TB/s 全是推导的（K/N 来自 config@TP，M = annotation 的 `tok`，dtype 来自 kernel 名字），输出里标
`derived`。计时列不受影响。要核对就开 `SHAPES=1` 的短窗口：`m3_decode.py` 会用同一 trace 里 prefill 录到的
dims 逐模块核对 K/N。

推导值超过硬件峰值，说明这一行的 shape 或模块标签错了，不是 kernel 快：用 `--peaks`（json，键
`bf16`/`fp8`/`fp4`/`hbm_tbs`）打开上限检查，超过的行留空并写明原因。没给 peaks 时脚本会提示检查是关着的。

## 10. 两阶段的模块集合要一致（原 15）

prefill 和 decode 用同一份映射。某个模块只在一边出现，通常是映射缺口。M3 上预期的差异都来自融合方式不同，
列在 `reference.md` 里；出现别的差异先查映射。

## 11. HIP graph 回放可能把 kernel 名字和时长配错（原 28）

graph 回放的 kernel 共用一个 correlation id，kineto 给计时记录挂名字时可能配错。名字的多重集合是对的，所以层数、
kernel 数都能对上，但哪个时长属于哪个 kernel 不可信。检查：同名 kernel 的 p90/p50，正常时接近 1（和 eager
prefill 差不多）；中位数 > 2 判为 SCRAMBLED，这时只能用窗口总量、step 的 union busy、每层的 union busy。
M3 在这个栈上的 decode 是单 stream，验证时配对是完整的；换镜像后每次都要看这一行。

## 12. 先确认选的 forward 是完整的（原 29）

窗口是运行中途开关的，边上的 forward 可能被截断，annotation 也可能比它的 kernel 先结束。先跑
`m3_trace_completeness.py`：每个 forward 数 core、切层、比对众数 step，并报告第一个和最后一个 forward 离
窗口边界多远。两个 rank 都要完整。

## 13. 一次通信是一组 kernel（原 32）

aiter 的 all-reduce 在主 stream 上是一组：`Memcpy DtoD`（拷进注册缓冲区）、`start_sync_kernel`、
reduce-scatter（prefill 融合了 RMSNorm：`ar_rmsnorm_*`）、`end_sync_kernel`、all-gather + norm、
`end_sync_kernel`；decode 是单个 `ar_ll_*`。数调用次数按组数，不按 kernel 数。`start_sync_kernel` 是 barrier：
先到的 rank 在里面等，它的时长就是等待。

PyTorch 的通用 launcher（`elementwise_kernel_manual_unroll`、`vectorized_elementwise_kernel` 等）只在模板参数里
写真正的操作，`short()` 把它放在方括号里保留下来（`[direct_copy float]`、`[FillFunctor<int>]`）。

## 14. 一个 rank 上通信慢，先看第二个 rank（原 33）

floor / excess 只说明这次调用比同种最快的那次慢，不说明为什么。`m3_rank_skew.py` 用通信的**结束**时间对齐两个
rank 的时钟（一次集合通信在所有 rank 上一起结束；|end mismatch| 在微秒级说明对齐可信），再比每次调用的到达
时间和时长：两边时长都长、skew 只有几微秒 = 通信本身（或第三个 rank）慢；一边到得晚、另一边时长长且时间都在
`start_sync` 里 = 后者在等前者。验证用的 trace 里，冷启动后的第一个 prefill 正是后一种：rank 0 的 GPU 大半时间
空闲（host 侧发射慢），每次都晚到，rank 1 上看到的"通信时间"大部分是在等 rank 0。

## 15. busy 远低于 wall（原 8）

主 stream busy 只占 wall 的一小部分时，空闲来自 GPU 之外：冷 JIT（服务启动后的第一个窗口、第一次遇到的 batch
形状）、host 侧发射跟不上、或在通信里等别的 rank（第 14 条）。这时逐 kernel 数字是 kernel 的成本，不是这个
forward 的成本。窗口开在 profiling 阶段 600 s 之后，就是为了避开这些一次性的开销；`--min-busy` 可以把这类
forward 从 steady 里排除。

## 16. 导出暂停会扰动 aiperf

`stop_profile` 要等 4 个 rank 把 trace 写完，期间引擎不处理请求，窗口越大越久（几十秒量级）。跨过这段的请求，
ITL 和 TTFT 都会被拉高，aiperf 的汇总也会受影响。所以采集运行的 aiperf 数字不能当性能结果引用，性能请用不开
profiler 的运行。`m3_window.py` 分别给出与窗口重叠的请求、窗口前 300 s 内完成的请求、导出后 300 s 内开始的
请求的 ITL，用来判断窗口里的状态是不是稳态。

## 17. 拷贝 trace（原 35）

trace 经过 shell 变量（`out=$(ssh ... cat f)` 再 `printf`）会丢掉 NUL 字节，gzip 损坏。二进制文件用普通重定向
（`ssh host cat f > f`）或 scp/rsync 拷，并和源文件核对 md5。`analyze.sh` 在精简之前会先 `gzip -t`。

## 18. kineto 可能丢掉窗口里的一部分 kernel（原 21、26）

GPU 时间戳比 host 时钟快的节点上，kineto 会丢掉大部分 kernel 记录；某些镜像里 torch profiler 只录到 CPU 事件，
要加 `ROCPROFILER_REGISTER_FORCE_LOAD=0`（`EXTRA_ENV` 传给服务）。表现是 completeness 里 core 数对不上，或
GPU 事件跨度远短于 profiler 的 `Trace` 跨度。镜像里有 roctracer 校准脚本时，`m3_server.sh` 会自动测偏移并用
shim 往回拨（`ROCTRACER_SHIFT=0` 关掉）。

## 19. indexer CP 改变 `qkv_idx_proj` 的 N（M3）

`CONC >= 15` 时服务开 `ATOM_M3_INDEXER_CP=1`，每个 rank 的 `qkv_idx_proj` 带上全部 index head，N 变大。
推导 K/N 时要用对的值：`analyze.sh` 从 `server.log` 第一行读这个开关并传 `--indexer-cp`；单独跑脚本时自己加。

## 20. MoE TP 下 expert 的形状（M3）

TP4 下每个 rank 持有全部 129 个 expert（128 个 routed + 融合成第 129 个的 shared expert），intermediate 按 TP
切：gate_up 的 K = hidden、N = 2 x 3072 / 4，down 的 K = 3072 / 4 = 768、N = hidden（kernel 名里的 `_K768_`
可以印证）。expert GEMM 的 M（token-expert 对数）= tok x (topk + 1)。权重读量按均匀路由估计"被选中的 expert
数"，batch 小时远少于 129，TB/s 才有意义。

## 21. 稀疏注意力的工作量（M3）

annotation 的 `sqsq = Σnq²`、`sqsk = Σnq·nkv`：全注意力（causal）的 query-key 对数是 `sqsk - (sqsq - tok)/2`。
稀疏层每个 query 最多看 `sparse_topk_blocks x sparse_block_size` 个 key（M3 是 16 x 128 = 2048），脚本取
`min(全注意力对数, tok x 2048)`；只有当这个 forward 里所有请求的已有上下文都不短于 2048 时这才是精确值。AgentX
的长上下文下通常成立；短请求多时它是上界，算出的稀疏 core TFLOP/s 偏高。
