# 专家并行（EP）深度解析——MoE 时代的第五种并行维度

DeepSeek-V3 有 671B 参数、256 个路由专家。一个 H100 只有 80 GB HBM。如果每张 GPU 都装完整模型，671B × FP16 2 bytes ≈ 1.3 TB——需要 17 张 H100 才能放下，还不算 KV Cache。但好消息是：每个 token 只激活 37B 参数（~74 GB FP16，一张 H100 刚好装下）。

「每个 token 只用少数专家」这个事实，催生了 DP/TP/PP 之后的一个全新并行维度——**EP（Expert Parallelism，专家并行）**。它不是 TP 的变种，不是 DP 的特例，它是为 MoE 的稀疏激活专门设计的第五种策略。

本文从「为什么 DP/TP/PP 解决不了 MoE 的存储问题」出发，拆解 EP 的核心机制、All-to-All 通信模式、EP+DP 耦合关系、EPLB 负载均衡、DeepEP 低延迟后端，以及生产环境的选型决策。

> **前置阅读**：[并行策略总览](parallelism_strategies.md) 中 EP 的概述部分（20 行概要）。本文是该概要的完整展开，建议先建立五种策略的全局认知再深入本文。

---

## 一、DP、TP、PP 为什么都解决不了 MoE？

先看三种经典策略面对 MoE 会发生什么。

**DP（数据并行）**：每张 GPU 有完整模型副本。DeepSeek-V3 完整 FP16 副本 = 1.3 TB。单卡装不下，DP 直接出局。

**TP（张量并行）**：把每层的权重矩阵按 hidden 维度切分。Attention 的 QKV projection 可以切，MoE 层的专家 FFN（gate/up projection shape `[7168, 2048]`）也可以切。但 TP 的问题是：每层每个 token 都有一次 All-Reduce。DeepSeek-V3 共 61 层 Transformer，其中后 58 层为 MoE（前 3 层为 Dense FFN，intermediate_size=18432）——仅 MoE 层就有 58 层 × 每层至少 2 次 All-Reduce（up_proj + down_proj 之后各一次）= 116 次 All-Reduce 每次 decode step，加上前 3 层 Dense FFN 的约 6 次，总计约 122 次。在 NVLink 内的 H100 上，单次 All-Reduce 延迟约 5-10μs——122 次就是 0.6-1.2ms，这比一次 decode step 的计算时间（~2-5ms）占比不低，但对 Dense 模型是可以接受的。问题在于：**TP 切专家 FFN 并没有利用 MoE 的稀疏性**——每个 token 的每个专家 FFN 计算 TP 都要参与 All-Reduce，即使 token 根本不走那个专家。而如果专家没有被切分到每张 GPU（即用 DP 方式保留完整专家），那 TP 就没有解决专家的存储问题。

**PP（流水线并行）**：把 61 层 Transformer（含 58 个 MoE 层）切成几段放到不同 GPU 上。PP 的问题是：它只切层数，不切专家数。每段 GPU 仍然要装下该段内所有层的全部 256 个专家——存储问题没解决，只是换了个维度。

结论：这三种策略都假设「每张 GPU 参与每个 token 的全部计算」，只是分工方式不同。MoE 打破了这一假设——每个 token 只需要少数专家，为什么要让所有 GPU 都参与？

**EP 的思路不是「切分所有 GPU 都要用的东西」，而是「让 token 自己选择要去哪张 GPU」。**

---

## 二、EP 的核心机制：分布、路由、通信

### 2.1 专家如何分布到 GPU 上

EP 的核心操作：把模型的所有路由专家均匀分配到 EP 组的各张 GPU 上。每张 GPU 持有连续的 1/N 专家权重。

```text
EP=8, 256 个路由专家, 共享专家 = 2

GPU 0: Expert   0 -  31  (32 experts)       ← 每张 GPU 还有共享专家的完整副本
GPU 1: Expert  32 -  63  (32 experts)
GPU 2: Expert  64 -  95  (32 experts)
GPU 3: Expert  96 - 127  (32 experts)
GPU 4: Expert 128 - 159  (32 experts)
GPU 5: Expert 160 - 191  (32 experts)
GPU 6: Expert 192 - 223  (32 experts)
GPU 7: Expert 224 - 255  (32 experts)
```

连续分布（连续编号的专家在同一张 GPU 上）是 vLLM 和 SGLang 的默认策略，原因有两个：一是路由层的输出（256 维 logits）可以简单地按分段映射到 GPU ID，不需要一张额外的映射表；二是连续分布让 Router 的 weight 和专家 weight 的索引对应关系保持可预测。

共享专家（shared experts，DeepSeek-V3 中有 2 个）不参与 EP 切分——它们处理的是「所有 token 不经过路由都会经过的专家」，所以每张 GPU 都需要完整副本。相较于每个 MoE 层 256 个路由专家的总权重（~22.5 GB FP16），2 个共享专家的开销（~176 MB）可忽略不计。

### 2.2 一个 token 的完整 EP 旅程

当一个 token 进入 MoE 层时，完整的 EP 执行流程如下：

```text
时间线：一个 token 在 EP=8 的 MoE 层中的旅程

Step 1 — Router 计算（本 GPU ⊗，无通信）
  hidden_states: [1, 7168]
  Router Linear: [7168, 256] → logits: [1, 256]
  Top-K: K=8, 选中 Expert {5, 23, 67, 89, 130, 155, 201, 244}
  映射到 GPU: {GPU 0: Expert 5, 23}, {GPU 2: Expert 67, 89},
             {GPU 4: Expert 130, 155}, {GPU 6: Expert 201}, {GPU 7: Expert 244}

Step 2 — Dispatch All-to-All（跨 GPU 通信）
  本 GPU 需要计算 Expert 5 和 23（在本地），其余 6 个专家分布在其他 GPU 上
  将 token 的 hidden_states 发送到 GPU 2、4、6、7（scatter）
  同时接收其他 GPU 发来的需要本 GPU 专家计算的 token

Step 3 — 专家计算（本 GPU ⊗，各自计算）
  GPU 0: Expert 5(token_A 的 hidden) + Expert 23(token_A 的 hidden) +
         其他 GPU 路由到 GPU 0 的 tokens
  每张 GPU 独立计算自己负责的专家的 FFN 输出

Step 4 — Combine All-to-All（跨 GPU 通信）
  GPU 2、4、6、7 将专家计算结果发回原始 GPU（gather）
  原始 GPU 收到所有 8 个专家的输出

Step 5 — Router 加权求和（本 GPU ⊗，无通信）
  output = Σ softmax(logits[i]) × expert_output[i]  (i ∈ Top-8)
  + shared_expert_output × shared_expert_gate
```

**关键观察**：一个 token 的 EP 旅程中，只有 Step 2 和 Step 4 有跨 GPU 通信。Step 1、3、5 都在本 GPU 完成。而且通信的是 token 的 hidden states（7168 维 × FP16 = ~14 KB），不是专家权重。

### 2.3 Dispatch 和 Combine 的通信量

通信量的精算：

- **Dispatch**：token 被发送到它路由到的专家所在的 GPU。最坏情况每个 token 的 Top-8 分布在 8 张不同的 GPU 上，数据量 = $\text{batch\_size} \times \text{top\_k} \times \text{hidden\_size} \times \text{dtype\_size}$。但实际上，一个 token 的 Top-8 通常集中分布在 2-4 张 GPU 上（因为相邻编号的专家在连续 token 上有相似的激活模式）。
- **Combine**：与 dispatch 对称，从各 GPU 收回专家计算结果，数据量与 dispatch 相同。
- **总通信量（每 token 每 MoE 层）**： $2 \times \text{top\_k} \times \text{hidden\_size} \times 2\,\text{bytes}$（FP16）= $2 \times 8 \times 7168 \times 2 \approx 229\,\text{KB}$。如果使用 FP8 通信，减半至 $\approx 115\,\text{KB}$。

对比 TP 的 All-Reduce：TP=8 的每次 All-Reduce 通信量 $\approx 2 \times \text{hidden\_size} \times 2\,\text{bytes} \times (N-1)/N \approx 2 \times 7168 \times 2 \times 7/8 \approx 25\,\text{KB}$。但 TP 每层有多次 All-Reduce（QKV proj、output proj、FFN up、FFN down），总共约 4-6 次 All-Reduce 每层。

EP 的每次通信量比 TP 大（229 KB vs 25 KB），但 EP 只在 MoE 层通信（attention 层无 EP 通信），且 EP 的 combine 可以和最后一轮专家计算并行。

> **Given**：EP 的通信模式是 All-to-All（一对多 + 多对一）→ **New**：为什么 EP 的通信开销在 MoE 架构下是可接受的——因为它只在需要时才发生，而非每层无差别执行。

---

## 三、EP 和 DP 的耦合：不可分割的并行对

### 3.1 EP 不能独立存在

TP 可以独立配置（TP=8，单节点跑一个模型副本）。PP 可以独立配置。

EP 不行。**EP 需要并发请求驱动多个 rank 同时工作。**原因非常直观：EP 只切了专家的存储和计算，每个 rank 只持有 1/N 的专家——某个请求的 token 路由到了 GPU 2 和 4 的专家，GPU 1 和 3 上如果没有其他请求的 token 在处理，它们就空闲。只有多路请求并发时，各 GPU 上的专家才都有活干，模型才能形成可服务的推理流水线。

这个"并发请求喂饱所有 rank"的能力，在 vLLM 中由 `--data-parallel-size` 提供。但需要区分两种 DP 概念：

|          | 传统多实例 DP                    | vLLM 引擎内 DP（`--data-parallel-size`）            |
| -------- | -------------------------------- | --------------------------------------------------- |
| 概念     | 每个实例有完整模型副本，独立服务 | EP 组内划分的并行维度，rank 间共享专家切分          |
| 通信     | 无需通信                         | 有元数据通信（调度、负载均衡）                      |
| 适用模型 | Dense 和 MoE 均可                | **仅 MoE**（`parallel.py:859` 阻止 Dense 模型使用） |
| 实现方式 | 启动多个 vLLM 实例 + 外部 LB     | `--data-parallel-size N --enable-expert-parallel`   |

本文后续讨论的"DP"均指 vLLM 引擎内 DP——它在 EP 语境下与 `--data-parallel-size` 含义一致。传统多实例 DP 不在 EP 讨论范围内。

### 3.2 EP 组的大小推导

vLLM 中 EP 组的大小由源码 `parallel_state.py` 的初始化逻辑决定，而非由用户直接配置：

```text
vLLM EP 组的 rank 编排（parallel_state.py:1894-1901）

WORLD_SIZE = PP_SIZE × TP_SIZE × DP_SIZE × CP_SIZE
  （WORLD_SIZE 为本次推理作业的总 GPU 数，
   PP 为流水线并行、TP 为张量并行、DP 为数据并行、CP 为 Prefill Context Parallel）

EP 组大小 = DP_SIZE × TP_SIZE × CP_SIZE

简化情况（CP=1, PP=1）：
  EP 组大小 = DP_SIZE × TP_SIZE = WORLD_SIZE
  → 所有 rank 在同一个 EP 组内，专家切分跨全部 WORLD_SIZE
```

`parallel.py:127-128` 的注释明确了这一设计意图：_"MoE layers will be sharded according to the product of the tensor parallel size and data parallel size."_

实际的 EP rank 数 = `TP_SIZE × DP_SIZE`（当 CP=1、PP=1 时就等于 WORLD_SIZE）。每张 GPU 持有的专家数 = 总路由专家数 ÷ (TP_SIZE × DP_SIZE)。TP 也在专家维度上生效——如果用 TP=4、DP=1，每卡持有 256/4 = 64 个专家；如果用 TP=1、DP=8，每卡持有 256/8 = 32 个专家。DP 提供了比 TP 更粗粒度的专家切分，且避免了每层 All-Reduce 的开销。

SGLang 的 EP 关系略有不同：`ep_size × moe_dp_size = tp_size`（见 `server_args.py:5273`）。SGLang 将 TP 视为总并行度，从中分出 EP 和 DP 两个子维度。这与 vLLM 的「EP = TP × DP」在数学上等价，但概念上 SGLang 将 EP 和 DP 作为 TP 的子划分。

典型配置：

```bash
# 8 GPU, DP=8, EP=8（EP 覆盖所有 8 张 GPU，每卡 32 个专家）
vllm serve deepseek-ai/DeepSeek-V3 \
  --data-parallel-size 8 \
  --enable-expert-parallel

# 32 GPU, TP=4, DP=8, EP=32（EP 组 = 32 张 GPU，TP 组 = 4 张 GPU 节点内）
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 4 \
  --data-parallel-size 8 \
  --enable-expert-parallel
```

### 3.3 EP 下的 KV Cache 归属

当一个 token 被 dispatch 到远程 GPU 计算专家后，它的 KV Cache 存放在哪？

**答案：KV Cache 始终驻留在 Router 计算发生的那张 GPU 上（即 dispatch 前的 GPU）——token 的 hidden states 被 dispatch 到远程，但 KV 不搬。**

token 被 dispatch 到远程 GPU 后，仅执行专家 FFN 计算。计算完成后 combine 回到原始 GPU，attention 层的 K/V 在原始 GPU 上存储和读取。这意味着：

1. **EP 不影响 KV Cache 的分布**。KV Cache 的切分仍由 DP 决定——每个 DP rank 持有自己服务的请求的完整 KV Cache。
2. **Attention 计算也在原 GPU 完成**。MLA 的 QKV 计算和 attention scoring 都发生在原始 GPU，不需要跨 EP rank 搬运 K/V。
3. **vLLM 中 EP 组内的 DP rank 各自独立管理 KV Cache**：每个 DP rank 持有自己服务的请求的完整 KV Cache，EP rank 之间的专家计算不涉及 KV Cache 的搬运或共享。源码中 `KVCacheGroupSpec`（`v1/kv_cache_interface.py:905`）用于管理混合注意力架构下不同层组的 KV Cache 规格（如 MHA + MLA 层混合），与 EP 的 rank 分组无关。

这个设计很巧妙：EP 借助了「attention 的计算量和 KV Cache 的大小都远小于专家 FFN 的权重」这个事实。如果 attention 本身也需要跨 GPU 切分（如超长上下文下的 SP），那才需要额外的策略——EP 和 SP 可以叠加使用，但它们在 KV Cache 的管理上是独立的。

> **Given**：DP 是「每张 GPU 有完整模型副本，各处理不同的请求」→ **New**：EP 在 DP 的基础上把专家权重进一步切分，但 attention 和 KV Cache 仍由 DP 管理，EP 和 DP 各有分工、不可互换。

---

## 四、All-to-All 通信：EP 的正反面

### 4.1 All-to-All 与 All-Reduce 的本质区别

TP 用 All-Reduce（all-reduce：每张 GPU 贡献 1/N 结果，求和后广播回每张 GPU）。通信模式是「所有人的结果，所有人都需要」。

EP 用 All-to-All（dispatch + combine）：每张 GPU 把需要远程专家计算的 token 发出去，收回自己需要的专家输出。通信模式是「只有路由到的 GPU 才需要通信」。

|                    | All-Reduce（TP）        | All-to-All（EP）                        |
| ------------------ | ----------------------- | --------------------------------------- |
| 模式               | 全对全归约              | 全对全散射+收集                         |
| 触发条件           | 每次矩阵乘之后          | 仅在 MoE 层                             |
| 频率               | 每层 4-6 次             | 每 MoE 层 2 次                          |
| 通信量（每 token） | ~25 KB × 次数           | ~229 KB × 2（FP16）/ ~115 KB × 2（FP8） |
| 延迟特征           | 与 hidden_size 相关     | 与 top_k × hidden_size 相关             |
| 是否可重叠         | 与下一层重叠（PP 场景） | 与专家计算重叠                          |

EP 的 All-to-All 在数据量上打不过 TP 的 All-Reduce——每次通信量更大。但它的优势在于**稀疏触发**：不是每层都通信（MoE 模型约 50% 的层是 MoE 层，其余是 attention + shared expert），且每次都经过的 attention 层和 LayerNorm 无 EP 通信。

### 4.2 NCCL 标准 All-to-All 在 EP 场景下的瓶颈

NCCL 的 All-to-All 实现是为通用场景设计的。在 EP 场景下暴露了两个特定瓶颈：

**瓶颈 1：decode 阶段的小批量高频率**。在 decode 阶段，每个 step 只有 1 个 token（per request），batch 通常在几个到几十个。每次 All-to-All 的数据量 = `batch_size × hidden_size / num_gpus`。当 batch_size=8、hidden_size=7168、num_gpus=8 时，每张 GPU 的发送量仅约 14 KB。在这个量级，延迟由 kernel launch 开销（~5-10μs）主导，而非带宽。而 DeepSeek-V3 有 58 个 MoE 层，每个 decode step 触发 58 × 2 = 116 次 All-to-All——即使每次仅 5μs 的 launch 开销，累积也达 0.58ms，成为不可忽视的延迟项。

**瓶颈 2：dispatch 和 combine 之间的 SM 空转**。NCCL 的标准实现是「先完成全部 dispatch，再做专家计算，再等全部 combine 完成」。在此模式中，第一轮 dispatch 的数据到达后 GPU 就开始等待，直到所有 dispatch 都完成。专家计算结束后又等待 combine 完成。SM 利用率远不到 100%。

---

## 五、DeepEP：低延迟 All-to-All 的专用答案

针对 §4.2 的两个瓶颈，DeepSeek 开源了 DeepEP——专门为 MoE EP 场景设计的低延迟 All-to-All 通信库。

### 5.1 绕过 NCCL 的专用路径

DeepEP 直接基于 NVLink（节点内）和 RDMA（节点间）构建通信原语，完全绕过 NCCL 的通用调度层。核心优化：

1. **专用通信原语**：针对 EP 的 dispatch-combine 模式定制，不携带 NCCL 为通用集合通信引入的额外元数据和调度开销。
2. **Persistent kernel**：将多次 All-to-All 调用的 kernel 在 GPU 上持久驻留（不退出），消除每次调用的 CUDA kernel launch 开销。这对 decode 阶段的频繁小批量通信尤其有效。
3. **FP8 通信路径**：dispatch 和 combine 的数据保持 FP8 格式在线路传输，接收端直接以 FP8 格式消费（不 dequant），将通信量减半。在 H100 的 NVLink 4.0（900 GB/s 双向）上，7 KB 的 FP8 token 传输延迟约 8ns。
4. **计算-通信重叠**：第一个 dispatch 的数据到达后立即开始专家计算（不等全部 dispatch 完成），最后一轮专家计算与 combine 并行执行。这是 DeepEP 最大的延迟收益来源。

### 5.2 两种模式：normal vs low_latency

DeepEP 提供两种运行时模式：

|                   | Normal（高吞吐）                           | Low Latency（低延迟）                  |
| ----------------- | ------------------------------------------ | -------------------------------------- |
| 目标              | Prefill 阶段大 batch                       | Decode 阶段小 batch                    |
| Batch 特征        | 数百-数千 tokens                           | 几个-几十个 tokens                     |
| 优化重点          | 带宽利用                                   | Kernel launch 开销消除                 |
| Persistent kernel | 不使用                                     | 使用                                   |
| 计算-通信重叠     | 部分                                       | 最大化                                 |
| vLLM 配置         | `--all2all-backend deepep_high_throughput` | `--all2all-backend deepep_low_latency` |

在 vLLM 中，通过 `--all2all-backend` 参数切换（`parallel.py:185-195` 定义了全部可用后端：`deepep_high_throughput`、`deepep_low_latency`、`mori_high_throughput`、`mori_low_latency`、`nixl_ep`、`flashinfer_nvlink_two_sided` 等）。也可以通过环境变量 `VLLM_ALL2ALL_BACKEND` 设置：

```bash
# 使用 DeepEP 低延迟模式（推荐用于 decode-heavy 场景）
vllm serve deepseek-ai/DeepSeek-V3 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency
```

### 5.3 实际收益

根据 DeepSeek 在 [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP) 中报告的测试数据，在 8×H100 EP=8 的 DeepSeek-V3 部署中，`deepep_low_latency` 相比标准 NCCL All-to-All 取得了以下典型收益（实际数值取决于 batch size 和序列长度）：

- **Decode 延迟降低**：约 15-25%（小 batch 场景收益最大）
- **Prefill 吞吐提升**：约 5-10%
- **GPU SM 利用率提升**：decode 阶段从 ~60% 提升到 ~75%

收益主要来自 kernel launch 开销的消除（decode 主导）和计算-通信重叠。

> **Given**：标准 NCCL All-to-All 在 EP 场景下的延迟瓶颈 → **New**：DeepEP 如何通过专用化、persistent kernel、FP8 通信和计算-通信重叠来突破这些瓶颈。

---

## 六、负载均衡：当专家们的工作量不一样

### 6.1 推理负载天然不均衡

训练中可以用 auxiliary load balancing loss 强制专家均衡——损失函数里加一项惩罚专家负载方差。推理中没法这么做——请求的 token 路由分布完全由输入内容决定。

根据 DeepSeek-V3 技术报告中路由器饱和度的分析，256 个专家在推理时呈长尾激活分布——少数热门专家承载了大部分 token，但绝大多数专家仍然被激活（覆盖面广），极少出现「某专家完全不被使用」的情况。以经验值估算：约 30% 的专家处理了 60% 的 token，其余 70% 的专家处理 40% 的 token。

真正的问题是**空间局部性**：如果 GPU 3 上的 32 个专家中恰好有 8 个是热门专家，GPU 3 会成为整个 EP 组的瓶颈——其他 GPU 都在等它的专家计算完成。

### 6.2 EPLB：不停机地迁移专家

SGLang 率先实现了 **EPLB（Expert Parallel Load Balancer）**，vLLM 后续跟进：

```text
EPLB 工作流程：

1. 监控（持续进行）
   滑动窗口统计每个专家的 token 到达率（如过去 100ms 内）
   计算每张 GPU 的负载总和 = Σ(该 GPU 上各专家的 token 到达率)

2. 检测（周期触发）
   IF max(GPU 负载) / avg(GPU 负载) > 阈值（典型 1.2）:
       触发重平衡

3. 迁移（在线执行）
   - 选择过载 GPU 上最热门的 1-2 个专家
   - 在目标空闲 GPU 上创建这些专家的权重副本
   - 等待副本同步完成
   - 原子性更新 Router 映射表：后续 token 立即路由到新位置
   - 原 GPU 上的专家权重保留（仍在处理路由表中的旧 token）

4. 清理
   旧映射表中的请求全部处理完毕后，可以选择回收原位置的专家副本
   （可选，取决于显存压力）
```

关键设计决策：**迁移期间不中断推理**。这通过双缓冲路由表实现——Router 维护两套映射表，迁移期间的旧请求仍使用旧表，新请求使用新表。类似于 RCU（Read-Copy-Update）。

### 6.3 冗余专家：以空间换均衡

当迁移频率太高（某些专家持续热门）时，EPLB 的收益被迁移开销吃掉。另一种互补策略是**冗余专家**——允许特定热门专家在多个 GPU 上有副本。

```bash
# vLLM 冗余专家配置
vllm serve deepseek-ai/DeepSeek-V3 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --num-redundant-experts 8     # 8 个额外专家副本
```

256 个路由专家 + 8 个冗余 = 264 个专家分布在 8 张 GPU 上。每卡 33 个专家（比原来多 1 个），但 8 个热门专家被复制到其他 GPU 上——Router 可以将 token 随机路由到副本中的任意一个，分散负载。

多余显存开销：每个冗余专家 ≈ 专家 FFN 权重大小（DeepSeek-V3 中约 88 MB FP16），8 个副本 ≈ 700 MB——对于每个 MoE 层 ~22.5 GB 的专家总权重来说，是 ~3% 的额外开销。

> **Given**：EP 的静态分布（256 专家均匀分到 8 GPU）无法适应推理负载的不均衡 → **New**：EPLB 动态迁移和冗余专家两种策略如何在延迟和显存之间做权衡。

---

## 七、EP 的边界：什么时候该用，什么时候不该用

### 7.1 EP 的前提条件

EP 不是万能药。它的适用边界非常清晰：

| 条件                      | EP 收益       | 原因                       |
| ------------------------- | ------------- | -------------------------- |
| MoE 模型，专家数 ≥ 64     | ✅ 强烈推荐   | 专家存储瓶颈，EP 是最优解  |
| MoE 模型，专家数 8-64     | ⚠️ 视情况     | 小专家数下 EP 通信占比高   |
| MoE 模型，专家数 < 8      | ❌ 不推荐     | All-to-All 开销 > 存储收益 |
| Dense 模型（LLaMA, Qwen） | ❌ 不需要     | 没有专家，EP 无意义        |
| 单卡能装下所有专家        | ❌ 通常不需要 | EP 不会带来吞吐收益        |

一个反直觉的结论：EP 的价值不在于「让推理更快」，而在于「让 MoE 模型能在多卡上跑」。如果只有 1 张 A100 但 Qwen2.5-72B（Dense 模型）装不下——用 TP（或量化）。如果有 8 张 H100 跑 DeepSeek-V3（MoE 模型）——首选 EP。两者的应用场景完全不同。

> **Dense 模型需要多副本怎么办？** vLLM 的 `--data-parallel-size` 仅用于 MoE + EP 场景（§3.1 中已区分两种 DP）。Dense 模型（LLaMA/Qwen）的多副本部署推荐多实例 DP：启动多个独立的 vLLM 实例，由外部负载均衡器分发请求。各实例之间无通信，不使用 `--data-parallel-*` 参数。

### 7.2 EP vs TP 在 MoE 场景下的决策框架

MoE 模型部署中最常见的困惑：「我用 TP 也能切权重，为什么还要学 EP？」

| 决策维度     | TP                      | EP                               |
| ------------ | ----------------------- | -------------------------------- |
| 切分对象     | 所有层的权重矩阵        | 仅 MoE 层的专家 FFN              |
| Attention 层 | 切（head 维度）         | 不切                             |
| 每层通信     | 是                      | 仅 MoE 层                        |
| 单层通信量   | 低（多次小 All-Reduce） | 高（2 次大 All-to-All）          |
| 总通信量     | 与层数成比例            | 与 MoE 层数成比例                |
| 存储效率     | 降至 1/N（N=TP size）   | 专家权重降至 1/N，attention 不变 |
| 关键约束     | 跨 NVLink 延迟          | All-to-All 延迟                  |

**经验法则**：

- **MoE 模型优先考虑 EP**：先用 EP 解决专家存储（EP=8, 每卡 32 专家），再看 attention 层是否需要 TP（通常 MLA 的 attention 权重很小，单卡可装，不需要 TP）。
- **Dense 模型只能用 TP/PP**：没有专家，不存在 EP 选项。
- **超大 MoE 模型 EP + TP 组合**：当 attention 层的隐藏维度太大以至于单卡装不下 attention 权重时（如将来的 DeepSeek 模型 hidden_size 扩展到 12288 以上），EP + TP 叠加。TP 切 attention，EP 切专家。

### 7.3 典型配置速查

```bash
# 场景 1：单节点 8×H100，DeepSeek-V3（推荐配置）
# TP=1, EP=8, 每卡 32 专家
vllm serve deepseek-ai/DeepSeek-V3 \
  --data-parallel-size 8 \
  --enable-expert-parallel

# 场景 2：单节点 8×H100，DeepSeek-V3，低延迟优化
# TP=1, EP=8, 每卡 32 专家, DeepEP 低延迟 All-to-All
vllm serve deepseek-ai/DeepSeek-V3 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_low_latency

# 场景 3：4 节点 32×H100，DeepSeek-V3（TP+EP 组合）
# TP=4, EP=32, 每卡 8 专家
# TP=4 用于 attention 层的 NVLink 内切分，EP 覆盖全部 32 张 GPU
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 4 \
  --data-parallel-size 8 \
  --enable-expert-parallel

# 场景 4：单节点 8×H100，DeepSeek-V3，开启冗余专家
vllm serve deepseek-ai/DeepSeek-V3 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --num-redundant-experts 8
```

> **Given**：读者完整理解了 EP 的机制、通信、负载均衡 → **New**：如何根据模型特性、硬件规模和延迟需求，在 EP 和 TP 之间做出具体选择。

---

## 八、总结

EP 不是「比 TP 更好的并行策略」。它是在 MoE 成为主流模型架构之后才成为必需品的第五个并行维度——在 Dense 模型时代，DP+TP+PP 覆盖了所有部署场景；MoE 的 256 专家 × 每 token 只用 8 个专家的稀疏激活模式，彻底改变了并行切分的基本假设。

EP 的核心取舍：

- **收益**：让 MoE 模型能在多卡上跑，专家权重存储降低到 1/N。对于 DeepSeek-V3 的 256 专家，EP=8 意味着每 GPU 只需存储 32 个专家（而非全部 256 个）。
- **代价**：每次 MoE 层引入两次 All-to-All 通信（dispatch + combine），每次 ~115-229 KB per token。负载不均衡导致部分 GPU 过载。
- **应对**：DeepEP 用专用化、persistent kernel、FP8 通信、计算-通信重叠来压延迟。EPLB 用动态迁移和冗余副本解决负载均衡。
- **边界**：EP 永远与 DP 耦合——没有 DP 的并行请求，EP 无法工作。Dense 模型不需要 EP。

向前看：随着 MoE 模型专家数从 256 增长到 512、1024，EP 的 All-to-All 延迟瓶颈将被进一步放大。FP4 通信（Blackwell 支持）、更激进的计算-通信全重叠、以及专家分组（先按组路由再按专家路由）是下一代 EP 优化的探索方向。

---

## 延伸阅读

- [并行策略总览：DP、TP、PP、EP、SP](parallelism_strategies.md) — 五种策略的对比性入门
- [DeepSeek-V3 MoE vLLM 部署方案](../deployment/deepseek_v3_moe_vllm_h20_deployment.md) — EP=32 的生产环境配置与 SLO 验证
- [vLLM WideEP 架构解析](../vllm/hardware_optimization/deepseek_blackwell_wide_ep.md) — EP + DP 在 Blackwell 上的专门优化
- [MLA TP KV Cache 冗余分析](../vllm/module_analysis/mla_tp_kv_redundancy.md) — EP 下 KV Cache 归属问题的深入探讨
- [vLLM 官方 EP 部署文档](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment.html)
- [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP)
