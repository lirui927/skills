---
name: minimax-m3-agentx-trace
description: 在 AgentX（aiperf inferencex-agentx-mvp）负载下给 MiniMax-M3-MXFP4（ATOM，TP4 + EAGLE3，MI455X / gfx1250）做运行中的 torch-profiler 采集，并把 trace 拆成逐 kernel、逐模块的 prefill / decode breakdown，再和 aiperf 的 ITL 对账。Use when asked to capture, profile or analyze a MiniMax-M3 ATOM trace (.pt.trace.json.gz), to attribute GPU time to attention core / indexer / MoE experts / all-reduce, to explain kernel time vs measured ITL, or to compare two M3 runs kernel by kernel. 采集占 4 张 GPU 约 40 分钟：先跑 check（不用 GPU），GPU 空闲且用户同意后才 run；其他模型或其他 TP 需要先移植切层规则。
---

# minimax-m3-agentx-trace

做法严格照 trace-kernel-breakdown（出处见 `<skill-dir>/NOTICE`）：**一次正常的 AgentX 回放中途**开一个
profiler 窗口，prefill 和 decode 都在这一个窗口里，不另起回放、不分开采。窗口在 aiperf profiling 阶段开始后
600–900 s，长 20–30 s，`ATOM_PROFILER_MORE=0`（不录 shape、不录 stack）。要核对 GEMM 的 shape 时，另起一次
服务开一个只录 shape 的短窗口，计时仍以无 shape 的窗口为准。

## 路径

`<skill-dir>` 是本 skill 的目录：`${CLAUDE_PLUGIN_ROOT}/skills/minimax-m3-agentx-trace`（存在时），否则是
读到这个文件的目录。Python 脚本通过项目自己的命令包装运行（`analyze.sh` 用环境变量 `PY` 指定），不要假定
宿主机的 `python3` 就是对的那个。

## GPU 是共享的：先 check，再 run

- `check` 不碰 GPU：镜像内容用一个**不挂任何设备、不联网**的临时容器检查；GPU 占用只读
  `/sys/class/kfd/kfd/proc`（不打开 GPU），按整机判断：任何人的任何进程开着 KFD 都算占用。任何时候都可以跑。
- `run` 要独占 4 张 GPU 大约 40 分钟（服务启动 + aiperf warmup + 600 s + 窗口 + 导出 + 剩余的 profiling 阶段）。
  只有在用户明确同意、且 check 显示 GPU 空闲时才跑。GPU 被占着时脚本每 `POLL` 秒查一次、一直等，
  **从不终止别人的进程或容器**；`WAIT_GPUS=0` 则直接退出。
- `run` 退出（正常结束、出错、Ctrl-C）时只停掉并删除自己的容器（`NAME`），GPU 随之释放。

## 第 1 步：检查（不用 GPU）

```bash
IMAGE=<服务镜像> MODEL_DIR=<含 MiniMax-M3-MXFP4/ 和 MiniMax-M3-EAGLE3-GQA/ 的目录> OUT=<新的输出目录> \
CONC=10 bash <skill-dir>/capture_agentx.sh check
```

镜像需要带 ATOM（支持 `ATOM_ENABLE_DETAILED_ANNOTATION` 和按选项的 profiler 开关）、aiter、aiperf 和 AgentX
数据集缓存（`semianalysis_cc_traces_weka_062126`）。check 逐项打印 `ok` / `FAIL`：镜像和其中的组件、模型目录、
输出目录可写且剩余空间 >= 30 GB（`MIN_FREE_GB`）、端口（`SERVER_PORT` / `ENGINE_PORT`）空闲、容器名空闲、KFD 看到的 GPU
数和架构、此刻谁在占 GPU（按容器名汇总）、宿主机 python3。全部 ok 才进入第 2 步。

## 第 2 步：采集（用 GPU）

同一组环境变量，把 `check` 换成 `run`：

```bash
IMAGE=... MODEL_DIR=... OUT=... CONC=10 bash <skill-dir>/capture_agentx.sh run
```

依次：等 GPU 空闲 -> 建容器（模型只读挂 `/models`，`OUT` 挂 `/out`，skill 目录只读挂 `/skill`）-> 用
`m3_server.sh` 起服务（profiler 预先挂好，参数照 ATOM 的 MiniMax-M3 AgentX recipe）-> 健康检查和一条冒烟请求
-> `m3_aiperf.sh` 开始 AgentX 回放 -> 等到 aiperf 日志出现 `Phase profiling (profiling) started` ->
`START_OFFSET` 秒后 `POST /start_profile` -> `WINDOW` 秒后 `POST /stop_profile`（等到 4 个 rank 都导出完）->
等 aiperf 跑完 `DURATION` -> （`SHAPES=1` 时重启服务录 shape，再开一个短窗口）-> 清理容器 -> `analyze.sh`。

常用变量：`CONC`（默认 10）、`START_OFFSET`（600）、`WINDOW`（25）、`DURATION`（1200，必须大于
START_OFFSET+WINDOW+60）、`SHAPES`（0）、`ACCEPT_RATE`（0.5933，固定 EAGLE3 接受率，和 AgentX 性能测试一致；
置空 = 真实接受率）。`CONC>=15` 时服务自动开 `ATOM_M3_INDEXER_CP=1`，分析会从 server log 第一行读出来。

产物（都在 `OUT` 下）：`traces/rank_{0..3}/*.pt.trace.json.gz`、`server.log`（第一行是实际生效的环境）、
`aiperf/`（`profile_export_aiperf.json`、`profile_export.jsonl`）、`aiperf_main.log`、`capture_meta.json`
（镜像、各时间点的 epoch）、`start_profile_main.json` / `stop_profile_main.json`（两次 POST 的响应）、
`model_config.json`、`capture.log`。

`stop_profile` 期间引擎暂停导出（大窗口要几十秒），跨过这段的请求 ITL 会被拉高：**这次运行的 aiperf 数字是被
扰动过的，不能当性能结果引用**，只用来和窗口对账（`<skill-dir>/pitfalls.md` 第 16 条）。

### 要确定的 decode batch：`WORKLOAD=fixed`

AgentX 回放里 decode 的 batch 随时间起伏，不等于 `CONC`（任一时刻只有一部分请求在 decode，c32 的窗口里 bs 在
5–14 之间）。要每步都是同一个 bs 时用固定 shape 重放：服务仍按 `CONC` 的 recipe 配置起（max-num-seqs、indexer
CP），不跑 aiperf；`m3_fixed.py` 在容器里把 `FIXED_BS`（默认 `CONC`）个各 `FIXED_CTX`（192038）token 的随机
token-id prompt 逐个 prefill 进 prefix cache，再一起发出（`ignore_eos`），之后每步 decode 都是 bs=`FIXED_BS`。
先不开 profiler 量 `FIXED_REF`（5）秒的客户端步长和一次 prefill 做对照，再开窗口：`FIXED_PRE`（2）秒 decode ->
一个 `FIXED_PREFILL_CACHED`（157312，整 block）+ `FIXED_PREFILL_NEW`（2763）的 prefill -> `FIXED_POST`（3）秒
decode。prefill 的默认值取自一次 c32 AgentX 窗口的代表 prefill。

```bash
IMAGE=... MODEL_DIR=... OUT=... CONC=32 WORKLOAD=fixed FIXED_BS=32 bash <skill-dir>/capture_agentx.sh run
```

多出的产物：`fixed_client.log`、`fixed_summary.json`（各时刻、对照和窗口里的客户端步长、两次 prefill 的耗时、
server log 里 prefill shape 的核对）；`window.txt` 用对照步长给出 profiler 的开销。这不是 AgentX 负载，报告里写明
fixed-shape replay（`wide.csv` 第一行已带上）。

## 第 3 步：分析（只用 CPU）

```bash
PY=python3 bash <skill-dir>/analyze.sh <OUT>          # 默认 rank 0 做 breakdown，rank 1 做对照
```

`analyze.sh` 的步骤，每步也可以单独跑（用法见各脚本开头的说明）：

| 步骤 | 脚本 | 输出 |
|---|---|---|
| 精简 trace（每个 rank） | `extract_trace.py` | `reduced/traces/rank_N/reduced.pt.trace.json.gz` |
| 每个 forward 是否完整（两个 rank） | `m3_trace_completeness.py` | `breakdown/completeness.txt` |
| prefill breakdown | `m3_prefill.py` | `prefill_{detail,wide,module_summary,comm,extras}.csv`、`prefill_picks.txt` |
| decode breakdown | `m3_decode.py` | `decode_{detail,wide,module_summary,comm,extras,rep_timeline}.csv`、`decode_picks.txt` |
| 窗口总账、step 周期、ITL 对账 | `m3_window.py` | `window.txt`、`window_summary.json` |
| 一张宽表（左 prefill、右 decode） | `m3_combine_wide.py` | `wide.csv` |
| rank A vs B 的通信 skew | `m3_rank_skew.py` | `rank_skew.txt` |

最后打印 `breakdown/SUMMARY.txt`（每步的关键行）。不在采集机上分析时，把 trace 用普通重定向拷过来并核对
md5（经过 shell 变量会丢 NUL 字节）。

## 方法要点

切层、模块映射和每条规则的来由见 `<skill-dir>/reference.md`，踩过的坑见 `<skill-dir>/pitfalls.md`。

- **层类型从 config 读**：`sparse_attention_freq`、`moe_layer_freq` -> 3 层 `full+mlp` + 57 层 `sparse+moe`。
  每个 forward 按 attention core 切层（每层恰好一个），层边界在上一层 MLP/MoE 输出的 all-reduce 之后；切出来的
  逐层类型必须和 config 一致，否则这个 forward 不用。
- **角色按位置定**：通用 GEMM 名字不带角色。core 之前第一个 GEMM 是 qkv（稀疏层是带 indexer 的
  `qkv_idx_proj`），core 之后第一个是 `o_proj`；MoE 块里 expert 之前的 GEMM 是 router；dense MLP 依次是
  gate_up、down。
- **总量精确、代表层取中位**：每个模块的 forward 总时间按全部层加总，和主 stream busy 精确相等；每种层取一个
  中位层逐 kernel 输出。只在少数层出现的 kernel（第一个稀疏层的 plan、EAGLE3 aux 层的 clone、第 0 层自己的
  input norm）在 `*_extras.csv` 里单列。
- **decode**：同一形状键（bs/tok/d/spec）的所有 step 合起来选代表层；每层时间是该层时间跨度内所有 stream 的
  union busy，每行只记它在关键路径上的部分，加起来等于 union busy。先做同名 kernel p90/p50 检查（> 2 判为
  SCRAMBLED，此时只用 union 总量）。
- **通信**：一次调用 = 主 stream 上一组连续的通信 kernel（拷贝 + start_sync + all-reduce/norm + end_sync）。
  拆成 floor（同种调用的最快一次）和超出 floor 的部分；超出的部分是等待还是通信本身慢，看 `rank_skew.txt`。
- **ITL**：ITL = step 周期 / 每序列每步 token 数（1 + 接受的 draft 数，取这个窗口内 server log 的 MTP 统计）；
  有 prefill 夹在 decode 之间时用 窗口 / decode forward 数。
- **shape**：`ATOM_PROFILER_MORE=0` 的 trace 没有 `Input Dims`，所有 M/K/N/dtype 都是推导的（K/N 来自
  config@TP，M 来自 annotation 的 tok），输出里标 derived；有 shapes 窗口时，`m3_decode.py` 逐模块核对。

## 验收清单（报告之前逐条确认）

- [ ] `completeness.txt` 两个 rank 都是 `RESULT: N of N forwards whole`；不完整的 forward 不选。
- [ ] prefill：`sum ... (check vs busy: diff 0.0 us)`；中位层模型 + 头尾 + 少数层 kernel 与 busy 相差 <= ~1%。
- [ ] decode：配对检查 `pairing looks intact`；代表层模型与 step union p50 相差 <= ~1%。
- [ ] 未归类（`misc_small`、`gemm_unmapped`、`act_quant`、`attn_out_misc`、`comm_other` 等）<= 2% of busy。
- [ ] 两阶段的模块集合只在 `reference.md` 列出的预期差异处不同；多出别的就是映射缺口。
- [ ] 层时间 CV 低（几个百分点）；CV 高先找出离群层。
- [ ] busy / wall 远低于 100% 时，先解释空闲（冷 JIT、新 batch 形状、等别的 rank，见 pitfalls），不要把逐
      kernel 数字当成这个 forward 的成本。
- [ ] side stream 的总量和未被覆盖的部分已列出。
- [ ] 推导出的 TFLOP/s、TB/s 已用 `--peaks`（或 `PEAKS=`）做过上限检查；超过峰值的行留空并说明。
- [ ] 窗口位置在 profiling 阶段 600–900 s；step 周期 / toks_per_fwd 和 aiperf ITL 的差距有解释。
      `WORKLOAD=fixed`：`fixed_summary.json` 里 `prefill_shape_ok` 两项都是 true、`stream_errors` 为空、
      `window.streams_alive` 等于 FIXED_BS（少了说明服务没按 ignore_eos 跑满），decode 只有一个形状键。

## 报告怎么写

- 每个数字跟着它的工作点：prefill 写 annotation（`tok`、`ctx`、`nctx`），decode 写形状键（`bs`、`d`、`spec`）；
  `wide.csv` 第一行写明模型、并发、rank、镜像、窗口位置和方法，转发数字时别把它丢掉。
- 位置一律给 Perfetto UI 时间 `HH:MM:SS.nnnnnnnnn`，加文件名和 stream（`*_picks.txt` 里都有）。
- 推导的 shape 和吞吐标 derived；decode 配对 SCRAMBLED 时逐 kernel 数字标 "pairing unverified"。
- 通信的超出部分只说"超出 floor"，判断是等待还是通信本身，要引用 rank skew。

## 适用范围

- 模型和并行方式：MiniMax-M3（config 里 60 层、3 full + 57 sparse），ATOM TP4（`--tp` 可改，但 K/N 推导和
  MoE TP 的假设要复核）。换模型要先按 `reference.md` 末尾的移植步骤重写切层和映射。
- kernel 名字来自 gfx1250 上的 ATOM + aiter。换镜像或架构后 `m3_trace_completeness.py` 不通过，说明 core 或
  通信的名字变了，先改 `m3_common.py` 的正则，再看数字。
- 只针对 ATOM 的 torch-profiler trace（`gpu_user_annotation` 带 `prefill[...]` / `decode[...]`）；SGLang /
  vLLM 的 trace 不适用。
