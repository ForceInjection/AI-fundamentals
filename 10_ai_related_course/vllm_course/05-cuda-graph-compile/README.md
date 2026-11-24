# 第 5 讲 · CUDA Graph 与编译

> 120 分钟 | 基线：nano-vllm `b9dd55a`；vLLM `1acec5c009`（v0.34.0rc0-1153,2026-11-24 核对）

本讲回答一个核心问题：**decode 为什么离不开 CUDA Graph，而 piecewise 又解决了「整图」的什么死穴?**

知识链：启动 warmup → `compile_or_warm_up_model`（compile 先行）→ `capture_model` 按两族尺寸网格捕图（decode 请求族 / prefill token 族）→ 每步 `dispatch_cg_and_sync_dp` 把 batch 落桶到「最小能容纳的图」或 eager → replay;attention 被 `splitting_ops` 摘在图外，piecewise 编译与 piecewise 图共享同一套切分。

## 一、nano 已学形态（回顾）

先修课第 8 课你走过的 `capture_cudagraph()`（`nanovllm/engine/model_runner.py:239-274`）与 `run_model`（`:212-229`）:

- 图 = **整段 decode 前向的录像**:capture 时把权重路径录下来，replay 时只把 `slot_mapping`/`context_lens`/`block_tables` 写进固定缓冲再 `replay()`；
- eager 条件是三个硬编码：**prefill、`enforce_eager`、`batch > 512** 任何一条成立就走逐步执行(`：214`);
- capture 尺寸集是硬编码的折中（有限的几个 batch size）。

三个折中在生产环境都会撞墙：prefill 形状千变，无法整图；batch size 连续分布，有限的 capture 尺寸意味着 padding 浪费；attention 依赖每步变化的元数据，根本没法进「录好的」图。V2 的解法逐条对着这三点。

## 二、CUDA Graph 机制回顾（30 秒版）

eager 模式下，每个 kernel 都要一次 CPU→GPU launch;decode 一步有成百上千个 kernel，而每个 kernel 的实际计算可能只有微秒级——**launch 开销在 batch 小的时候占主导**（官方 `cuda_graphs_multimodal.md:54` 明确「batch size 小时 host 侧 launch 开销更显著」）。CUDA Graph 把一串 kernel 录成一张图，**一次 launch 播放全部**；代价是图内张量地址固定——新数据必须先拷进图引用的固定 buffer（`debug_vllm_compile.md:346-360` 的 capture 语义：1+ 个 kernel 装进一张图，replay 读写同一片内存，只捕 CUDA kernel、不捕 CPU 工作）。

## 三、现状走读

### 环节 1:capture 流程（启动期，一次性）

调用链：`GPUWorker.compile_or_warm_up_model`（`gpu_worker.py:1068-1073`）→ **compile 分支先行**（`:1082-1090`，见环节 5）→ 非 eager 时 `capture_model()`（`:1155-1157` → `model_runner.py:1172-1252`）。capture 期间：日志会打「Graph capturing finished in %.0f secs, took %.2f GiB」（`:1246-1251`，官方注「通常 5~20 秒」）,capture 后 `lock_workspace()`（`:1241`）防止 workspace resize 让图指针失效。**注意：capture 不走 `_dummy_run`**（`model_runner.py:914-1013` 的 dummy run 只服务 profiling/warmup/自适应验证采样；「dummy run 触发 capture」是旧教程说法）——capture 由 `ModelCudaGraphManager.capture` 的闭包工厂（`cudagraph_utils.py:743-857`）自己构造 dummy 输入，图内只引用 `input_buffers` 的固定地址（`:759-771` 切片构造）。

### 环节 2：两族尺寸网格

HEAD 的 capture 尺寸分两族（`config/compilation.py`）:

| 族               | 尺寸含义                       | 网格（`compilation.py`）                                                  | 上限（SM100 默认）                  |
| ---------------- | ------------------------------ | ------------------------------------------------------------------------- | ----------------------------------- |
| decode 请求族    | （请求数 n, query_len q） 二维 | ≤6 全捕，此后按 `0.65·n^(2/3)` 取 2 的幂（`:122-125`）                    | `max_num_seqs`（`cuda.py:644-655`） |
| prefill token 族 | 单步 token 数                  | 1,2,4 → 每 8 到 64 → 倍增 4 档 → 之后每 256/512，共 54 个尺寸（`:89-99`） | 8192 token                          |

每一步运行时，`dispatch`（`cudagraph_utils.py:561-605`）按「能容纳实际负载的**最小**图」落桶；没有图 → `CUDAGraphMode.NONE` 直接 eager（`:599-605`）。落桶后的 padding 在 runner 侧补（`model_runner.py:1502`，`num_tokens_after_padding = max(num_tokens, desc.num_tokens)`），输入从静态 buffer 切片（`:1682`）。旧教程「1,2,4,…,512 共 25 档」的口径已被这套平台网格取代。

### 环节 3：每步的中央裁判

`dispatch_cg_and_sync_dp`（`vllm/v1/worker/gpu/dp_utils.py:232-369`）在每步 `execute_model` 开头被调用（`model_runner.py:1933-1949`），做四件事：判断本步是否强制 eager（profile/跳过编译，`:295-301`）；否则调 `cudagraph_manager.dispatch` 选出本批的 `BatchExecutionDescriptor`；DP>1 时跨 rank collective 对齐 padding（`:355-369`）；支持 DBO microbatch 拆分（`:318-333`）。第 1 讲你见过的这个函数，就是「每步一次、batch → 图」的中央裁判。

### 环节 4:piecewise——attention 在图外

为什么 attention 进不了图?它依赖每步变化的变长元数据，对地址固定的图不友好。vLLM 的做法：把整个 attention 包成 custom op `vllm::unified_attention_with_output`，列入 `splitting_ops`（`config/compilation.py:962-987`，约 20 个 op——attention、mamba mixer、gdn、sparse indexer 等所有「读变长 KV 状态」的算子）。效果：Dynamo 视角仍是全图，但切分后**attention 段留在图外 eager 执行**，图只捕「attention 之间的 token-wise 段」。

要辨析的一组概念：**piecewise 编译与 piecewise 图共享同一套切分，但不是同一个东西**——编译侧用切分让每个子图单独过 Inductor 产 kernel（`vllm/compilation/piecewise_backend.py:86`），图侧对每段捕小图消 launch。另有一个不走 compile 的变体：breakable CUDA graph（`vllm/compilation/breakable_cudagraph.py`，运行时遇到 attention 就结束 capture、eager 跑完再续捕，`VLLM_USE_BREAKABLE_CUDAGRAPH=1` 开启，部分平台默认启用）。

FULL 模式（整图，attention 也在图内）什么时候可用?由 attention 后端声明：`AttentionCGSupport` 四级(ALWAYS > UNIFORM_BATCH > UNIFORM_SINGLE_TOKEN_DECODE > NEVER)，`resolve_cudagraph_mode_and_sizes`（`compilation.py:1588-1737`）按后端能力降级 FULL → FULL_AND_PIECEWISE / FULL_DECODE_ONLY / PIECEWISE / NONE。spec decode 的 draft 是特例：只支持 FULL_DECODE_ONLY（PIECEWISE 不支持 draft，`speculator.py:38-43`），且**多步 decode 循环整体融合进一张 FULL 图**（`model_runner_v2.md` §9——不再「逐步各捕一张」）。

### 环节 5:torch.compile 四档

`-O<n>` / `--optimization-level`（`vllm.py:176-188`；**默认 O2**，`vllm.py:530`）:

| 档         | 内容（`vllm.py:301-384` 字典）                                                                              |
| ---------- | ----------------------------------------------------------------------------------------------------------- |
| O0         | 不编译、不捕图；所有融合 pass 关                                                                            |
| O1         | Dynamo+Inductor 编译 + **PIECEWISE 图**；条件开启 norm/act 量化融合；flashinfer autotune 开                 |
| O2（默认） | O1 + **FULL_AND_PIECEWISE**（decode 整图、其余分段）;TP 场景加 allreduce+rmsnorm 融合、rope+kv-cache 融合等 |
| O3         | **当前与 O2 逐字段相同**（`:343-384`）                                                                      |

融合 pass 落地在 `vllm/compilation/passes/fusion/`（rms_quant_fusion、act_quant_fusion、allreduce_rms_fusion 等，由 `PassConfig` 逐个开关）。compile 与 capture 的尺寸逻辑分开：Inductor 对通用 shape 编一份即可，图必须逐尺寸捕（`compilation.py:537-546` docstring）。一个易混点：**`--enforce-eager` 直接关的是 capture（`gpu_worker.py:1155-1157`），不必然关 compile**——compile 分支看 `-O` 档位；旧说法「enforce_eager = 关 compile + 图」在不显式设 `-O` 时才完全成立。

### 环节 6：调试：二分关旗标

`docs/design/debug_vllm_compile.md:8-15` 的旗标二分表，从粗到细定位问题：

| 旗标                      | 关掉什么      |
| ------------------------- | ------------- |
| `--enforce-eager`         | 全关          |
| `-cc.mode=0`              | 只关 compile  |
| `-cc.cudagraph_mode=NONE` | 只关图        |
| `-cc.backend=eager`       | 只关 Inductor |

再深入：`TORCH_TRACE=<目录>` 导出编译全流程日志，`tlparse` 出 HTML（报 bug 时官方要求附上）；`TORCH_LOGS=output_code` 看 Inductor 产物；graph break 按 Dynamo 报告处理；编译缓存出问题就 `rm -rf ~/.cache/vllm /tmp/torchinductor_$(whoami)`。注意：**没有 `vllm compile` 子命令，也没有 `VLLM_COMPILE_DEBUG` 环境变量**（旧教程说法）。

## 四、V0/旧教程 vs HEAD

| 旧说法                                                     | HEAD 事实                                                                                                                |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 默认 PIECEWISE 图                                          | 默认 **-O2 = FULL_AND_PIECEWISE**（decode 整图；`vllm.py:357`）                                                          |
| 「capture 在 initialize 的 dummy run 里触发」              | `capture_model` → 闭包工厂（`cudagraph_utils.py:743-857`）；`_dummy_run` 只服务 profiling                                |
| capture 尺寸「1,2,4,…,512 共 25 档」                       | 两族平台网格（decode 请求族 / prefill token 族 54 尺寸，SM100 上限 8192）；旧字段 deprecated（`compilation.py:788-798`） |
| 「FA3 后端不支持图所以要 piecewise」                       | 后端声明 `AttentionCGSupport` 四级，管理器自动降级（`:1588-1737`）                                                       |
| 「draft 逐步各捕一张图」                                   | 多步融合进一张 FULL 图（`speculator.py`；draft 不支持 PIECEWISE）                                                        |
| `--cudagraph-mode` / `vllm compile` / `VLLM_COMPILE_DEBUG` | 三个都不存在；等价写法 `-cc.cudagraph_mode=NONE`、TORCH_TRACE+tlparse                                                    |
| 「enforce_eager 关 compile + 图」                          | 直接关的只是 capture;compile 看 `-O` 档位                                                                                |
| 文档里的 `CudagraphDispatcher` 类                          | 不存在；dispatch 在 `CudaGraphManager.dispatch`，key 是显式参数+查找表                                                   |

## 五、实验与产出

见 [hands-on-exercise.md](hands-on-exercise.md)：实验 1 三态对照（eager / 默认 -O2 / -O0，沿用第 2 讲探针，拿第 1 讲实验 3 存的数据开局）；实验 2 capture 日志解析（图数量/耗时/显存，脚本在 `code/`）；实验 3（可选）调试二分演练。课后作业见 [homework.md](homework.md)。

## 六、边界

| 内容                            | 本讲                       | 去处                |
| ------------------------------- | -------------------------- | ------------------- |
| Inductor 产出的 kernel 细节     | 不展开                     | torch 官方文档      |
| draft 图的 spec decode 语义     | 只讲「融合进一张 FULL 图」 | 第 7 讲             |
| `AttentionCGSupport` 的后端判定 | 已在第 4 讲支持矩阵给出    | 第 4 讲矩阵交叉引用 |

## 七、前后衔接

第 4 讲的 attention 后端在本讲获得「图内还是图外」的属性（`AttentionCGSupport`）；第 7 讲投机解码的 draft 图融合直接复用本讲的 FULL 机制；第 10 讲生产运维里，「capture 花了多少秒/多少 GiB」是启动排障的常客。
