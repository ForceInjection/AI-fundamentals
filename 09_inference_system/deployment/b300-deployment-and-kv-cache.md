# B300 上的模型部署与 KV Cache：官方手册最佳实践

**as-of 2026-09-15** ｜ 本文整理 vLLM 与 SGLang 官方手册推荐的配置、调优判据和现成配方，以及厂商 model card 上跑通过的组合。每条推荐都标了出处。

## 口径与来源

| 标注           | 含义                                                               |
| -------------- | ------------------------------------------------------------------ |
| **【SGLang】** | SGLang 官方文档，本地检出 `c415f977b8`（2026-09-10）               |
| **【vLLM】**   | vLLM 官方文档，本地检出 `43d691ec6b`（v0.26.1rc0-459，2026-08-07） |
| **【NVIDIA】** | NVIDIA model card、官方博客或 MIG 产品页                           |
| **【源码】**   | 对本地 vLLM/SGLang 检出逐行核对                                    |
| **【实测】**   | 有测量数据的第三方结果                                             |

---

## 一、模型部署

### 1.1 先过三道门槛

**① SM103 是独立 target，不是 SM100。**

`sm_100a` 的 cubin 不能直接跑在 B300 上，必须走 `sm_100f` family 目标或 `sm_103a`。这是大量 `no kernel image is available` 事故的根因。**用 CUDA 13 镜像是充分条件**——不要去找所谓「SM103 专用 tag」，SGLang 安装页里 grep `SM103`/`B300` 是零命中。

**② INT8 在这块卡上不可用。**

PTX 的 `.kind::i8` 目标列表是 `sm_100a / sm_101a / sm_110a`，**从未扩展到 sm_103a**；CUTLASS 按 arch guard 跳过 INT8 UMMA；vLLM 没有 INT8 GEMM。所以所有量化方案只能走 FP8 / NVFP4。

⚠️ **失败时机很坑**：vLLM 是在**模型完整加载之后**才硬报错。POC 阶段先用 ~1B 模型冒烟，不要等 405B 下载完。

**③ vLLM 镜像 tag 有个反转陷阱。**

≥ v0.20.0 起**无后缀 = CUDA 13**，`-cu129` 才是 CUDA 12 退出版；v0.19.x 及更早相反。NVIDIA 自己在 Model-Optimizer PR #2042 里记了这条：`Don't trust the tag name — select a tag reporting CUDA_VERSION >= 13`。

### 1.2 引擎与镜像

| 引擎         | 版本（as-of 2026-09-15） | 镜像                                            | 备注                                                                  |
| ------------ | ------------------------ | ----------------------------------------------- | --------------------------------------------------------------------- |
| **vLLM**     | v0.29.0                  | `vllm/vllm-openai:v0.29.0`                      | 三家唯一正式 release；官方推荐 B300 用 CUDA 13                        |
| **SGLang**   | v0.5.19                  | `lmsysorg/sglang:v0.5.19-cu130-runtime`         | CUDA 12 lane 已退役                                                   |
| TensorRT-LLM | **1.3.0rc26**            | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc26` | **迭代 26 个 RC 仍无 GA**；legacy TensorRT backend 已移除。不建议主线 |

**镜像已知问题（未修复）**：

- vLLM 官方镜像的 `TORCH_CUDA_ARCH_LIST` **不含 `10.3`**（v0.29.0 实测仍只有 `10.0`，PR #44344 未合并）→ 建议自建镜像显式加入
- vLLM：DeepEP MoE all-to-all 在 SM103/GB300 **不可用**（issue #41687 仍 open），实测「通信占 decode kernel 时间 ~93%」
- SGLang：tcgen05 kernel 在 sm_103 失败 Xid 13（#34340 open）；MegaMoE 路径 `CUDA_ERROR_ILLEGAL_ADDRESS`（#37559 open）

### 1.3 vLLM 的四个优化等级

vLLM 提供 `-O0` 到 `-O3` 四档，用启动时间换稳态性能【vLLM】：

| 等级              | 内容                                                      |
| ----------------- | --------------------------------------------------------- |
| `-O0`             | 无优化，启动最快                                          |
| `-O1`             | 简单编译 + 快速融合 + PIECEWISE cudagraphs                |
| **`-O2`（默认）** | **额外编译区间、额外融合、FULL_AND_PIECEWISE cudagraphs** |
| `-O3`             | 激进优化（当前等于 O2）                                   |

**三个官方给的启动加速手段**：

1. **复用编译缓存**——`torch.compile` 产物存在 `VLLM_CACHE_ROOT`（默认 `~/.cache/vllm`），可跨机器拷贝、也可烤进镜像。设 `VLLM_FORCE_AOT_LOAD=1` 让缓存未命中时显式报错，而不是静默重编
2. **`--kv-cache-memory` 跳过显存 profiling**——启动日志会打印能复现当前分配的精确值，下次启动传回去即可跳过测量与 CUDA graph 估算。该值只在同卡、同初始空闲显存下有效
3. **`--enforce-eager` 跳过 CUDA graph**——启动最快，代价是稳态 decode 性能

### 1.4 按模型给配方

以下命令均来自官方 cookbook / model card，可直接作为起点。

#### Kimi-K3 on B300 1×8

节点数由硬件决定，不是独立选择：**B300 1×8 = TP8 / DCP8**（Balanced 档）。全系列节点配置：

| 硬件     | 节点    | 说明                           |
| -------- | ------- | ------------------------------ |
| B200     | 2×8     |                                |
| GB200    | 4×4     |                                |
| **B300** | **1×8** |                                |
| GB300    | 2×4     |                                |
| H200     | 2×8     | Unified High-Throughput 用 4×8 |

**大规模预设（32 GPU @ B300，每节点跑相同命令、只改 `--node-rank`）**【SGLang】：

```bash
SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=20480 \
sglang serve \
  --trust-remote-code \
  --model-path moonshotai/Kimi-K3 \
  --tp-size 32 --ep-size 32 \
  --enable-dp-attention --dp-size 4 --enable-dp-lm-head \
  --nnodes 4 --node-rank <rank> --dist-init-addr <node0-ip>:20000 \
  --moe-a2a-backend megamoe --moe-runner-backend deep_gemm \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --mem-fraction-static 0.92 \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
  --host 0.0.0.0 --port 30000
```

扩展方式：**保持每 replica 的形状不变，只动 replica 数**。

| GPUs | B200/B300 节点 | `--tp-size` / `--ep-size` | `--dp-size` |
| ---- | -------------- | ------------------------- | ----------- |
| 16   | 2×8            | 16                        | 2           |
| 32   | 4×8            | 32                        | 4           |
| 64   | 8×8            | 64                        | 8           |

**两个关键 flag 的理由**：

- `--kv-cache-dtype fp8_e4m3` **是承重的**——cookbook 原文：`bf16 KV does not fit 128 requests per replica`
- `--mamba-ssm-dtype bfloat16`：KDA state 的 dtype 决定每卡账单，只有 attention-TP 宽度、SSM dtype、cache 策略三个旋钮能动它

#### DeepSeek-V4 系列

**单节点 TP=4**（B200/B300/GB200/GB300/H200 通用；RTX PRO 6000 用 TP=2，H100 用 TP=8）【SGLang】。

**Agentic 长上下文 + HiCache DRAM offload**（B200 FP4 + DSpark，TP8，并发 8–16）【SGLang】：

```bash
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Pro-0813 \
  --trust-remote-code --tp 8 \
  --moe-runner-backend flashinfer_mxfp4 \
  --enable-deepseek-v4-fp4-indexer \
  --disable-flashinfer-autotune \
  --mem-fraction-static 0.90 \
  --swa-full-tokens-ratio 0.1 \
  --chunked-prefill-size 8192 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 6 \
  --enable-hierarchical-cache \
  --hicache-ratio 2.75 \
  --hicache-write-policy write_through \
  --hicache-io-backend direct \
  --hicache-mem-layout page_first_direct
```

高并发档（DEP8 + DP attention，并发 64–160）换 `--dp 8 --enable-dp-attention --ep-size 8 --mem-fraction-static 0.88 --swa-full-tokens-ratio 0.02 --hicache-ratio 8`，并把 `--chunked-prefill-size` 提到 49152。

⚠️ **`--chunked-prefill-size` 是全局预算，会被 `--dp` 均分**——上面那个 49152 除以 dp8 才是每 rank 的 6144。

⚠️ **DSv4 的 HiCache host 层用 `--hicache-ratio`（host/device token 比）定容，不是 `--hicache-size`**。

### 1.5 并行策略

**① MLA 模型上，attention TP > 1 会复制 KV，不是切分。**

这一点在【源码】里可以直接看到：

```python
# vllm/config/model.py:1504-1509
if self.use_mla:
    # When using MLA during decode it becomes MQA
    return 1
...
return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)
```

注意 GQA 分支：**KV heads 少于 TP 时也是「复制」**——Qwen3-235B（4 个 KV head）上开 TP8，8 卡各存一份完整副本。

**SGLang cookbook 把这条写成了 Deep PP 的依据**：Deep PP 用 `--tp-size 1 --pp-size 8`（B300/GB300），因为

> `--tp-size 1` is also what buys context: above TP1 the MLA KV is replicated across the TP ranks, so **TP2 × PP8 holds roughly half the tokens of TP1 × PP16 for the same memory**.

实测（GB200，ISL 8192 / 并发 32）【SGLang】：

| 形状           | prefill tok/s/GPU |
| -------------- | ----------------- |
| **PP16 × TP1** | **4550**          |
| PP8 × TP2      | 3596              |
| TEP16          | 2407              |
| TP16           | 1652              |

**但并发低于 ~8 时反过来**：pipeline 填不满，TEP16 领先（1947 vs 1227），此时用 `--tp-size 16 --ep-size 16`。

**② vLLM 官方的「何时用哪种并行」**【vLLM】：

| 策略   | 官方给的适用场景                                                       |
| ------ | ---------------------------------------------------------------------- |
| **TP** | 模型单卡装不下；或需要降低每卡权重占用，以腾出更多 KV 空间             |
| **PP** | TP 已吃满但还要继续切；跨节点；很深很窄的模型                          |
| **EP** | **MoE 专用**。设 `enable_expert_parallel=True`，用 EP 替代 MoE 层的 TP |
| **DP** | 有足够 GPU 复制整个模型；要扩吞吐而非扩模型；多用户环境                |

**SGLang 官方的表述更直接**：`Data parallelism is better for throughput. When there is enough GPU memory, always favor data parallelism for throughput.`【SGLang】所以大 MoE 的默认形态是 Attention DP + MoE EP，不是单一 TP。

**③ GB300 跨 pod MNNVL 传输要加三个环境变量。**

cookbook 原文：某些 GB300 集群上跨 pod NVLink 传 KV 会失败于 `nvlink_transport.cpp:497 Requested address ... not found!`，解法是在 prefill 和 decode 两侧的 `sglang serve` 前都加上【SGLang】：

```bash
MC_FORCE_MNNVL=1 NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1
```

### 1.6 CPU 与 NUMA：一条官方公式

vLLM V1 是多进程架构，每个进程都要 CPU。官方给出的**最低物理核数公式**【vLLM】：

```
单 DP：至少 2 + N 个物理核（1 API server + 1 engine core + N GPU worker）
多 DP：A + DP + N + (1 if DP > 1 else 0)
```

其中 `A` 是 API server 数（默认等于 DP），`N` 是 GPU 总数。官方举的例子：8 卡上 `DP=4, TP=2` → 4 API server + 4 engine core + 8 GPU worker + 1 DP coordinator = **17 个进程**。

> **官方警告原文**：`Using fewer physical CPU cores than processes will cause contention and significantly degrade throughput and latency. The engine core process runs a busy loop and is particularly sensitive to CPU starvation.`

⚠️ 注意是**物理核**：开了超线程的话，1 vCPU = 1 超线程 = 半个物理核，所以要 `2 × (2 + N)` 个 vCPU 起。

**NUMA 绑定**：多路服务器上用 `--numa-bind`，vLLM 会自动探测 GPU-to-NUMA 映射并施加 `--cpunodebind=<node> --membind=<node>`。需要自定义时加 `--numa-bind-nodes` / `--numa-bind-cpus`。容器里可能需要 `--cap-add SYS_NICE`。

### 1.7 PD 分离

**先看官方的定性**：vLLM 文档开篇直写 `Disaggregated prefill DOES NOT improve throughput`【vLLM】。

Dynamo 自己的 GB300 实测（Kimi-K3，agentic 负载）【NVIDIA】：

| 配置                          | 系统吞吐 tok/s/GPU | 每用户 tok/s | TTFT P50   |
| ----------------------------- | ------------------ | ------------ | ---------- |
| SGLang **Aggregated**         | **84.4**           | 51.3         | **573 ms** |
| SGLang **Disaggregated 1P1D** | 61.0               | **75.9**     | 4,809 ms   |
| vLLM **Aggregated**           | **62.2**           | 57.8         | 879 ms     |
| vLLM **Disaggregated 1P1D**   | 50.3               | **90.0**     | 6,357 ms   |

**聚合部署的系统吞吐更高、TTFT 低 5–6 倍**；PD 分离只在 per-user 吞吐上赢。

**它会减少 decode 侧的 KV 容量**（Dynamo 明文）：`Moving prefill to dedicated workers ... reserves GPUs for a pool whose KV cache is not used during decode`。

**Dynamo 明确拒绝给固定 P:D 比**：`Treat replica counts as a response to observed bottlenecks rather than as fixed ratios.`

**实际配 PD 时要处理的细节**【SGLang】：

- 混合模型（如 K3）的传输要搬**两样**：paged MLA KV **和** KDA recurrent state
- 端口：prefill `30000`、decode `30100`，派生的 ZMQ/dist 范围不能在同一台机器上撞车
- `--prefill` 后面那个位置参数 `8998` 必须和 `--disaggregation-bootstrap-port` 一致，否则只有 decode worker 会注册
- `--disaggregation-decode-extra-slots` **要显式钉住**：不钉的话，32 请求以下默认翻倍、以上默认**归零**

---

## 二、KV Cache

### 2.1 先把三个数分清

| 数字       | 含义                           | 来源                             |
| ---------- | ------------------------------ | -------------------------------- |
| **288 GB** | B300 **卡规格**                | 【NVIDIA】Blackwell Ultra 博客   |
| **279 GB** | GB300 NVL72 形态下**单卡可用** | 【NVIDIA】MIG 产品页 `1x 279 GB` |
| **270 GB** | HGX B300 形态下**单卡可用**    | 【NVIDIA】MIG 产品页 `1x 270 GB` |

MIG 页脚注标明这些是 `Preliminary specifications`。**读规格表用 288，做容量规划用 279 或 270。**

**`gpu_memory_utilization` 的分母是总显存，不是剩余显存**【源码】：

```python
# vllm/v1/worker/utils.py:446-456
requested_memory = math.ceil(init_snapshot.total_memory * cache_config.gpu_memory_utilization)
```

默认 **0.92**（`vllm/config/cache.py:68`）——网上大量二手资料写 0.9，会算错。

⚠️ **TensorRT-LLM 用的是相反约定**：`free_gpu_memory_fraction` 分母是初始化时的**空闲**显存。跨引擎混用必错。

### 2.2 量化：官方推荐 `fp8_e4m3`

SGLang 官方手册有专门的 Best Practices 小节，三条【SGLang】：

1. **优先用离线量化的模型**——scaling factor 已包含在 checkpoint 里
2. **格式选 `fp8_e4m3`（推荐）**；`fp8_e5m2` 用于更大动态范围；`nvfp4` / `fp4_mx_block16` 用于最大显存节省（**实验性**）
3. **确认 attention backend 支持**量化 KV

**官方警告**：量化 KV 在 attention 里使用前要反量化，**如果反量化没有和 attention kernel 融合，性能会极慢**，可能抵消掉显存收益。

**显存收益（官方数字）**【SGLang】：

| 对比        | 可容纳 token 数 |
| ----------- | --------------- |
| FP4 vs BF16 | **≈ 3.56×**     |
| FP4 vs FP8  | **≈ 1.78×**     |

（已计入 block scaling factor 的额外开销）

**精度对照表（官方实测，KV16 = BF16 / KV8 = FP8 E4M3 / KV4 = FP4 E2M1）**【SGLang】：

| 模型             | 数据集       | KV16   | KV8    | KV4        |
| ---------------- | ------------ | ------ | ------ | ---------- |
| Qwen3-235B-A22B  | gsm8k        | 0.9168 | 0.9181 | 0.9186     |
| Qwen3-235B-A22B  | aime25       | 0.7733 | 0.7333 | **0.6000** |
| Qwen3-235B-A22B  | gpqa_diamond | 0.7010 | 0.6899 | 0.6778     |
| DeepSeek-R1-0528 | gsm8k        | 0.9157 | 0.9154 | 0.9124     |
| DeepSeek-R1-0528 | aime25       | 0.5067 | 0.4934 | **0.4000** |
| DeepSeek-R1-0528 | gpqa_diamond | 0.7707 | 0.7697 | 0.7273     |
| GPT-OSS-120B     | gsm8k        | 0.9161 | 0.9163 | 0.9152     |
| GPT-OSS-120B     | aime25       | 0.7533 | 0.7667 | **0.3533** |
| GPT-OSS-120B     | gpqa_diamond | 0.5081 | 0.5434 | **0.3202** |

**官方给的三条结论**：

- **简单数据集**（gsm8k）：FP4 在两种规模上都接近 FP8/BF16
- **模型越大越能容忍 FP4**（200B+ 明显好于小模型）
- **长上下文可能退化更明显**——量化误差会累积

> 官方 Tip 原文：`Large models on simpler tasks typically show minimal degradation, while smaller models or complex reasoning tasks may require FP8 or BF16 for acceptable accuracy.`

**工程含义**：`fp8_e4m3` 是安全默认；FP4 KV 只在「大模型 + 简单任务」上考虑，且必须自己复测。

**FP8 在实践里的地位**——它不是可选项，是承重项。SGLang cookbook 写得很直接：`--kv-cache-dtype fp8_e4m3` **is load-bearing**，因为 bf16 KV 装不下每 replica 128 个请求。

**NVFP4 KV 的生产禁用理由**（vLLM issue #55673，2026-09-07 开，**仍 open**）【实测】：

4×B200、Qwen3.5-397B、FlashInfer TRT-LLM attention，1,319 题 GSM8K：

| KV cache / max seqs | Flexible exact match | Invalid responses |
| ------------------- | -------------------- | ----------------- |
| **FP8, 512**        | **96.664%**          | 0.682%            |
| NVFP4, 64           | 6.520%               | 93.177%           |
| NVFP4, 512          | **4.549%**           | **94.920%**       |

报告人排除了「高并发伪影」（两个并发上限都失败，FP8 对照组用同一后端）。**根因至今未定位**：PR #55670 修了一个真实的 scale 转换缺陷，打补丁后仍是 4.776%，说明该缺陷 `is not sufficient to explain this Qwen failure`。

**适用范围**：特定几何（`head_dim=256 / 8 query heads / 1 KV head`）上的问题，不等于所有模型都会挂。但**同族、同后端、对照组正常、根因未定位**，足以构成生产禁用。

**两个记账陷阱**：

- **MLA 的 per-token KV 有两种口径，差 13.9%**：

| 口径                                            | 每 token 每层 | 61 层全模型  |
| ----------------------------------------------- | ------------- | ------------ |
| BF16 未量化                                     | 1152 B        | 70,272 B     |
| FP8 朴素口径（`kv_lora_rank 512 + qk_rope 64`） | 576 B         | 35,136 B     |
| **FP8 vLLM 实际打包 `fp8_ds_mla`**              | **656 B**     | **40,016 B** |

NVIDIA + SGLang 的 GB300 长文用的是朴素 576；vLLM 实际是 512 B FP8 NoPE + 16 B scales + **128 B BF16 RoPE**。**按 576 算容量会少算 14%。**

- **`--cpu-offload-gb` 是权重卸载，不是 KV 卸载**；**`--swap-space` 已从 vLLM 移除**。

### 2.3 分层：HiCache 的实际配置

#### SGLang HiCache

**先记住一条官方定性**【SGLang】：

> L1 和 L2 是**单实例私有**的；只有 L3 能共享。`Host memory cannot be pooled across instances or across hosts, not even for two instances on the same node.`

所以想要跨实例复用，**必须配 `--hicache-storage-backend`**。`file` 后端默认落在节点本地 `/tmp/hicache`；`mooncake` / `hf3fs` / `nixl` / `aibrix` 在共享同一 namespace 时可达集群级。

**核心参数与官方推荐值**【SGLang】：

```bash
--page-size 64                        # 缓存管理的页大小
--enable-hierarchical-cache           # 启用 HiCache
--hicache-ratio 2                     # host 内存为 GPU 显存的 2 倍
--hicache-size 100                    # 直接给 GB 数，会覆盖上面的 ratio
--hicache-io-backend kernel           # CPU↔GPU 搬运的 I/O 后端
--hicache-write-policy write_through  # GPU→CPU 的写策略
--hicache-storage-backend             # 可选：hf3fs / mooncake / nixl / aibrix
```

**内存布局的兼容性（官方）**：

| 布局                | 兼容性                                                             |
| ------------------- | ------------------------------------------------------------------ |
| `page_first`        | **只兼容 `kernel` I/O 后端**；用 `direct` 会自动切到 `layer_first` |
| `page_first_direct` | 专为 `direct` 后端设计，兼容 fa3，零拷贝性能与 `page_first` 相同   |
| `layer_first`       | —                                                                  |

**预取策略三选一（官方）**：

| 策略            | 语义                         |
| --------------- | ---------------------------- |
| `best_effort`   | 需要时终止预取               |
| `wait_complete` | 保证完整预取，缓存复用率更高 |
| `timeout`       | 两者折中                     |

**与 PD 分离的两种官方组合**：

1. **仅 Prefill 开 HiCache**——让 Prefill 实例之间共享 KV（适合 SystemPrompt 场景）
2. **Prefill 开 HiCache + Decode 开异步卸载**——让 Prefill 能复用 Decode 节点的 KV（适合多轮对话）

第二种的 Decode 侧多一个 flag：`--disaggregation-decode-enable-offload-kvcache`。

**异构 TP 支持**：不同部署用不同 TP（如 tp=4 和 tp=8）共享同一存储时，用 `--hicache-storage-backend-extra-config '{"tp_lcm_size": 8}'`，值是所有 TP size 的**最小公倍数**。

**现成的 Mooncake 部署样例（官方）**：

```bash
export MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata"
export MOONCAKE_GLOBAL_SEGMENT_SIZE=816043786240
export MOONCAKE_PROTOCOL="rdma"
export MOONCAKE_DEVICE="$DEVICE_LIST"
export MOONCAKE_MASTER=127.0.0.1:50051

python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 8 --page-size 64 \
  --enable-hierarchical-cache --hicache-ratio 2 \
  --hicache-mem-layout page_first_direct --hicache-io-backend direct \
  --hicache-storage-backend mooncake --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy timeout
```

**一条容易踩的坑**（K3 的 DCP recipe）：**host 层还没完全 DCP-aware**——

- L3 **总是**丢掉 DCP flag
- L1+L2 **开着 Spec Decode 时**也丢；关掉 Spec Decode 才保留
- 丢掉 DCP 之后，MLA KV 退回 TP 复制，**每请求的 KV 容量相应缩水**

#### vLLM KV 卸载

**两套 spec**，由 `kv_connector_extra_config` 的 `spec_name` 选【vLLM】：

- `CPUOffloadingSpec`（默认）：单 CPU 层，完成的 GPU block 拷进 pinned host memory
- `TieringOffloadingSpec`：多级，CPU 主层 + 一个或多个二级层

**关键约束（官方）**：`Only the CPU primary tier has direct GPU access. Secondary tiers cannot read from or write to GPU memory; all GPU↔secondary transfers are staged through the CPU primary tier.`

**单层（纯 CPU）最小配置**：

```bash
vllm serve <model> \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "block_size": 64,
      "cpu_bytes_to_use": 1000000000
    }
  }'
```

**多层（CPU + 文件系统）**：

```bash
vllm serve <model> \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "TieringOffloadingSpec",
      "cpu_bytes_to_use": 10737418240,
      "block_size": 16,
      "eviction_policy": "lru",
      "secondary_tiers": [
        {"type": "fs", "root_dir": "/mnt/kv_cache",
         "n_read_threads": 32, "n_write_threads": 16}
      ]
    }
  }'
```

**官方的 Tuning Tips（原文照译）**【vLLM】：

- `cpu_bytes_to_use` 越大越好——更大的 CPU 层意味着更少去访问更慢的二级层、命中率更高。**这个值是所有 worker 的总和，不是每 worker**
- **单层（纯 CPU）配置时，`cpu_bytes_to_use` 要大于 GPU KV 总量**。因为卸载是即时的，CPU 层比 GPU 小就只是镜像，不提升命中率
- `block_size` / `blocks_per_chunk`：更大的卸载块减少簿记开销，但会加大查找粒度
- **FS 线程数**：`n_read_threads` / `n_write_threads` 按存储能承受的并发调。**读在 prefill 路径上对延迟敏感，prefill 命中率高时多给读线程**
- 共享 `root_dir` 的多实例：模型、`block_size`、并行布局、dtype 都一样才会共用一个 `<digest>` 子目录；改任何一项都会生成新目录，旧的成为孤儿（无害，可删）

**跨实例共享的硬前提**：`PYTHONHASHSEED` 必须在所有实例上设成同一个固定值（如 `0`），否则每个进程的 block 内容哈希种子不同，**同样内容会算出不同文件名**。P2P 层会**强制校验**这一点——没设就启动失败，握手里发现对端值不同会被拒绝。

**卸载的收益与代价（实测数据）**：

| 层级转移        | 实测效果                                                                                    | 口径     |
| --------------- | ------------------------------------------------------------------------------------------- | -------- |
| CPU DRAM        | TTFT 降 2–22×，吞吐最高 9×                                                                  | 【vLLM】 |
| 命中 vs 重算    | 1k tokens 时 2.2×，80k 时 32.8×                                                             | 【实测】 |
| CPU DRAM on/off | 10k/40k/80k 改善 1.30/1.18/1.08×；**1k 时四种配置无差别**                                   | 【实测】 |
| SSD 盈亏平衡    | 8k prompt 需 **77.8%** 前缀复用才回本；80k 时降到 7.8%                                      | 【实测】 |
| SSD 超大规模    | 100k prompt 超 HBM 时 −79% TTFT、+264% 吞吐；但 cache 达 12.6–13.7M tokens 时最好也只有 −3% | 【实测】 |

**核心独立结论**：`a cache hit is not sufficient for caching to be beneficial` / `External KV caching should therefore be treated as a setup specific admission decision`。【实测】

**带宽现实**：GPU↔CPU 在 PCIe 5.0 ×16 上实测 54–56 GB/s（理论的 40–45%）；NVMe 单盘 6–13.5 GB/s。**不要把链路速率当吞吐。**

⚠️ **一条反直觉的实测**：GPUDirect Storage（KvikIO/cuFile）在这个负载上 `was slower than all our other implementations`。

### 2.4 复用：两项已经被实测的收益

**Prefix Caching / RadixAttention**：

Agent 场景的真实命中率 **95.7%**（~4,300 个 Claude Code + Codex session，~350,000 LLM steps）【实测】：

- fresh tokens 只占 append tokens 的 **19.0%**——**约 81% 的 prefill 原则上可命中**
- miss 是**空闲驱动**的：间隔超 5 分钟开始出现低命中，1 小时后几乎全 miss
- **cache 命中占 agent 总成本 59.5%**，append 占 29.2%，output 只占 11.2%

**一个直接可用的调参**：超时从 1 分钟提到 1 小时，命中率 85.4% → 98.6%，但存储比从 R=0.74 涨到 5.07（**约 7 倍**）。**大部分收益是便宜的**——5 分钟时已达 ~94% 命中，R≈1.9。

**官方给的调度策略**【SGLang】：`--schedule-policy lpm`（longest prefix match）会重排请求以提升缓存命中，代价是调度开销增加——共享前缀多的负载用。

**Radix cache 不是永远开着好**：K3 cookbook 明确——**对无前缀的流量（离线批处理、评测）关掉它**，因为一个请求占 4–5 个 state slot，关掉只占 1 个。

**cache-aware 路由**：

| 调度器                 | 输出 tok/s | TTFT p90    |
| ---------------------- | ---------- | ----------- |
| **precise-scheduling** | **8730**   | **0.542 s** |
| approximate            | 6944       | 31.083 s    |
| load-based             | 4429       | 94.865 s    |
| random                 | 4429       | 92.551 s    |

（8 vLLM pod / 16×H100，Qwen3-32B，150 个 B2B 客户 × 6,000-token 共享上下文）【llm-d】

**但有反方观点**（Anyscale，作者含 NVIDIA 人员）：`balancing KV cache reuse with token load leads to better overall serving performance than maximizing KV cache reuse alone`。两个具名失效模式：**request herding** 与 **session-level imbalance**。

**私有化多租户必须配 `cache_salt`**：它注入首个 block 的 hash，保证只有同 salt 的请求能复用 KV block。**不配会跨租户泄露。**

**harness 侧的硬规则**（可直接抄进开发规范）：

- 保持 prompt 前缀稳定——**哪怕一个 token 的差异都会让从该点起的缓存全部失效**
- 系统提示开头放时间戳会直接杀掉命中率
- 上下文保持 append-only
- **序列化必须确定性**——很多库不保证 JSON key 顺序稳定，会静默破坏缓存

### 2.5 容量与并发：实际调过的数字

**同一份硬件上的真实并发**（GB300 NVL72，NVIDIA + SGLang 联署）【实测】：

| 指标                                    | GB300                              | GB200       |
| --------------------------------------- | ---------------------------------- | ----------- |
| `mem_fraction_static = 0.75` 下静态预算 | ≈216 GB                            | ≈144 GB     |
| 权重（DeepSeek-R1 NVFP4, EP16/TP16）    | ≈40 GB                             | ≈40 GB      |
| **KV 池**                               | **≈176 GB**                        | ≈104 GB     |
| 单请求 KV（136K cached tokens）         | ≈4.45 GiB                          | —           |
| 理论上限                                | **≈40 req/GPU**                    | ≈24 req/GPU |
| 按 ~85% 运维目标                        | **36 req/GPU**（DEP16 → 576 并发） | 20 req/GPU  |

⚠️ **该页自身有内部不一致**：TL;DR 按 DEP8 写「288 concurrent」，正文按 DEP16 写 576；加速比出现 1.38X–1.58X / 1.4X–1.6X / 1.4X–1.5X 三种表述。

**Kimi-K3 on 8×B300 的现成配置**【NVIDIA model card】：

```bash
--quantization modelopt_mixed --tensor-parallel-size 8
--moe-backend flashinfer_trtllm --kv-cache-dtype fp8
--max-model-len 196608 --max-num-seqs 32
--attention-backend FLASHINFER_MLA
```

配套限制：`flashinfer_trtllm` **是强制的**——`auto-resolution never triggers the TRT-LLM deferred-finalize path, and flashinfer_cutlass lacks a SiTU kernel for routed experts`；且 `a pip-installed SGLang cannot load this checkpoint`（SGLang 路径需专用镜像）。

**DCP 换并发**【SGLang】：`--dcp-size 8` 去重 attention-TP 组内的 MLA KV——**同等引擎吞吐下并发上限 +72%，代价是 ITL 约 1.8×**。适用于上下文 ≥ ~16K，或每 replica 并发超过 128。

### 2.6 常见误算

**那个 `GPU KV cache size` 日志行不是容量承诺**【源码】：

```python
# vllm/v1/core/kv_cache_utils.py:1871
return int(max_concurrency * max_model_len), max_concurrency
```

它字面上就是 `max_concurrency × max_model_len`，**派生的最坏情况乘积**。maintainer 原话：`That log message simply shows a theoretical upper bound ... KV cache blocks are allocated lazily and incrementally, not all at once.`

**按危害排序的误算清单**：

1. **activation / CUDA graph 显存没算进去**（vLLM 现已默认开启估算并打印等效换算）
2. **profiling 看不到的临时 buffer**——KDA 的 chunked-scan buffer 随 `max_num_batched_tokens` 线性增长、在 forward 内部瞬时分配，**启动 profiling 覆盖不到**；超出某个 chunk size 后引擎会在**服务中途**死掉
3. **把 `max_model_len × 并发` 当 KV 需求**
4. **preemption 在 OOM 之前先毁掉 p99**——读 `vllm:num_preemptions`，不要从日志推断
5. **hybrid attention 模型给滑窗层分配了全上下文 KV**（修复后 SWA 层改用 `SlidingWindowSpec`）
6. **`max_num_seqs` 一职两用，且等待队列无界**（`--max-num-queued-tokens` 默认关闭）
7. **block size 在 hybrid 模型上被逼到病态值**——GLM-5.3-Flash 上曾出现 block size 7808，一个 12-token 的 prompt 占掉 32.9% 的池
8. **按 576 B/token 算 MLA 容量**（实际打包是 656 B，少算 14%）

**一条最好的单变量对照**（RTX 4090 / Qwen3-8B bf16）【提交者自测】：`max_num_batched_tokens` 从 2048 提到 8192，KV 池缩 10%，**p99 TTFT 涨 71%**（23.9s → 40.8s），goodput 从 54.7% 掉到 46.2%，而总吞吐不变。

---

## 三、官方调优方法

### 3.1 SGLang：看日志里的三个数

**启动后看 `available_gpu_mem`**【SGLang】：

```
[2025-08-11 17:17:03] max_total_num_tokens=665690, chunked_prefill_size=8192,
max_prefill_tokens=16384, max_running_requests=4096, context_len=65536, available_gpu_mem=13.50 GB
```

官方判据：

- **5–8 GB = 合适**（留给 activations 和 CUDA graph）
- **10–20 GB = 太高**，调大 `--mem-fraction-static` 把内存给 KV
- **太低 = 有 OOM 风险**，调小

`mem_fraction_static = (模型权重 + KV cache pool) / GPU 显存容量`。官方给的实操法：**以 0.01 为步长往上加，直到你的负载出现 OOM**。

**稳态看 `token usage` 和 `#queue-req`**【SGLang】：

```
Decode batch. #running-req: 233, #token: 370959, token usage: 0.82, cuda graph: True, gen throughput (token/s): 4594.01, #queue-req: 317
```

- `#queue-req` 健康区间 **100–2000**。**频繁看到 0 = 客户端提交太慢**
- **`token usage > 0.9` 才算利用得好**。若 < 0.9 且 `#queue-req > 0`，说明服务端太保守，把 `--schedule-conservativeness` 降到 **0.3**
- 反之若频繁看到 `KV cache pool is full. Retract requests.`，把 `--schedule-conservativeness` 提到 **1.3**。**每分钟约 1 次是可以接受的**

**OOM 的三条处置（官方）**：

- prefill 时 OOM → `--chunked-prefill-size` 降到 **4096 或 2048**（代价：长 prompt 的 prefill 变慢）
- decode 时 OOM → 降 `--max-running-requests`
- 都可以再降 `--mem-fraction-static` 到 **0.8 或 0.7**（代价：限制并发上限、降峰值吞吐）

**CUDA graph 上限**：默认只对小 batch（<160 或 256）开。大 TP 的模型上 CUDA graph 到 512 或 768 仍有用，可调大 `--cuda-graph-max-bs-decode`，但要同时降 `--mem-fraction-static`。

**`--mem-fraction-static` 的实践取值**：SGLang 在 B300 上给 VLM 的保守起点是 **0.82**（`raise it toward 0.85 if startup reports insufficient memory`）；大 MoE 吞吐档用 0.88–0.92。DSv4 的 DSpark 路径明确要 `Keep --mem-fraction-static 0.90 to leave enough headroom for the batch-256 verify graph`。

### 3.2 vLLM：preemption 的四条处置

出现这条 warning 说明 KV 不够【vLLM】：

```
WARNING ... Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode
because there is not enough KV cache space. This can affect the end-to-end
performance. Increase gpu_memory_utilization or tensor_parallel_size to
provide more KV cache memory.
```

**官方给的四个动作**：

1. 提高 `gpu_memory_utilization`
2. 降低 `max_num_seqs` 或 `max_num_batched_tokens`
3. 提高 `tensor_parallel_size`（切分权重腾出显存，但同步开销上升）
4. 提高 `pipeline_parallel_size`（层分到多卡，但有延迟惩罚）

**监控**：`vllm:num_preemptions`。V1 的默认 preemption 模式是 `RECOMPUTE` 而非 `SWAP`。

### 3.3 chunked prefill 的调参

vLLM V1 **默认开启** chunked prefill。官方给的调参方向【vLLM】：

- **调小**（如 2048）→ ITL 更好（prefill 更少打断 decode）
- **调大** → TTFT 更好
- **官方推荐**：`For optimal throughput, we recommend setting max_num_batched_tokens > 8192 especially for smaller models on large GPUs`

---

## 四、上线前必测项

| #   | 项                                                            | 理由                                                                                                                                                                            |
| --- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ①   | **KV 量化的 ≥100K needle retrieval**                          | vLLM #56700 原话：`Wrong interpretation passes short-prompt smoke tests and only fails at ≥100K needle retrieval — silent numerical corruption.` **短 prompt 冒烟测试等于没测** |
| ②   | **物理核数 ≥ `2 + N`**（多 DP 用官方公式）                    | vLLM 官方：CPU 不足会 `significantly degrade throughput and latency`                                                                                                            |
| ③   | **自建镜像显式加入 `10.3` arch**                              | 官方镜像的 `TORCH_CUDA_ARCH_LIST` 不含它                                                                                                                                        |
| ④   | **`--disaggregation-decode-extra-slots` 显式设定**（若配 PD） | 不设则 32 请求以上默认归零                                                                                                                                                      |
| ⑤   | **跨实例共享 KV 时统一 `PYTHONHASHSEED`**                     | vLLM 官方：否则同样内容算出不同文件名                                                                                                                                           |
| ⑥   | **精度验收绑定 kernel backend 版本**                          | 同一份 NVFP4 权重换 backend 差约 1 分 GSM8K                                                                                                                                     |
| ⑦   | **启动后确认 `available_gpu_mem` 落在 5–8 GB**                | SGLang 官方判据                                                                                                                                                                 |
| ⑧   | **长上下文任务的忠实度回归**                                  | 独立论文：INT4 KV 下 `over 90% of faithfulness changes are negative, i.e., accuracy metrics are blind to this regression`                                                       |

**另外两条来自 cookbook 的提醒**：

- **B300 1×8 上只有 `Unified` 的 Low-Latency 与 Balanced 两格是 Verified**，其余全部标 `Final Verification In Progress`——「treat those as starting points to verify」
- **大规模预设没有一个在最终权重上跑过完整 serving round**：`the constants derive from measured single- and dual-node rounds plus a 64-GPU sweep. Validate throughput and accuracy on your workload before committing a fleet.`

---

## 附：三条收口判断

1. **KV 量化只到 FP8。** `fp8_e4m3` 是 SGLang 官方明写的推荐值；FP4 KV 官方标为实验性，且 aime25 上有 0.75 → 0.35 的实测反例。
2. **MLA 模型上用 DP，不用 TP。** 两家官方都写明 TP > 1 时 MLA KV 被复制；SGLang 的 Deep PP 实测差距达 2.75 倍。
3. **第一杠杆是模型选型。** MLA 与 Qwen3-235B 的 GQA 差近 5 倍 per-token KV（40 KB vs 188 KB），量级大于任何量化手段。

---

## 相关阅读

- [KV Cache 技术体系](../kv_cache/README.md)——本文只讲 B300 上的配置实践，压缩、淘汰、卸载的机制原理见该目录
- [显存估算](../memory_calc/README.md)——容量测算的方法与脚本
- [vLLM 助力 DeepSeek 吞吐量飙升 5 倍](../vllm/hardware_optimization/deepseek_blackwell_wide_ep.md)——WideEP、NVFP4/FP8 与 Weight Offloading v2 的原理拆解，本文 §1.5 的并行策略是它在部署侧的另一面
- [把 KV Cache 压缩推到极限：DeepSeek-V4.1-Flash 技术报告精读](../deepseek-v41-flash-kv-compression.md)——模型架构侧的 KV 压缩（CSA2、FP4 main KV），与本文的引擎侧实践互补
- [NVIDIA GB300 NVL72 架构解析](../../01_hardware_architecture/superchips/nvidia_gb300.md)——本文用到的显存口径在那一篇有完整的拓扑与带宽背景
- [核心推理优化技术深度解析](../reference_design/03-核心推理优化技术深度解析.md)——KV Cache、Continuous Batching、量化等技术的原理层梳理
