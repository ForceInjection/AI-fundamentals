# 大模型推理并行策略——DP、TP、PP、EP 到底在切什么

70B 的模型塞不进一张 80GB 的 H100——这是所有推理工程师都会遇到的第一道墙。解法不是换更大的 GPU，而是把模型拆开、把数据拆开、把计算拆开，让多张 GPU 协作完成推理。

「拆」的方式有五种：**DP（数据并行）、TP（张量并行）、PP（流水线并行）、EP（专家并行）、SP（序列并行）**。它们的名字听起来相似，但切的东西完全不同——有的切权重，有的切数据，有的切 KV Cache。本文从「切的到底是什么」出发，用统一的图示把五种策略讲清楚。

---

## 一、总览：五种策略，切三种东西

|             | DP         | TP         | PP       | EP       | SP       |
| ----------- | ---------- | ---------- | -------- | -------- | -------- |
| 切的维度    | batch      | hidden     | layers   | experts  | seq_len  |
| 每张 GPU 有 | 完整模型   | 1/N 权重   | 1/N 层   | 1/N 专家 | 完整模型 |
| KV Cache    | 独立       | 分片       | 独立     | 独立     | 分片     |
| 通信量      | 高（梯度） | 极高       | 中       | 低       | 中       |
| 典型场景    | 训练       | 大模型推理 | 超大模型 | MoE 推理 | 长上下文 |

---

## 二、数据并行（DP）：每人一份完整模型

**切什么**：batch。把一批请求分成 N 组，每组发给一张 GPU，每张 GPU 上有完整的模型副本。

```text
GPU 0: [模型副本 0] ← 请求 1,2      GPU 1: [模型副本 1] ← 请求 3,4
        ├─ 权重: 完整                     ├─ 权重: 完整
        ├─ KV Cache: 独立                  ├─ KV Cache: 独立
        └─ 计算: 1/N batch                 └─ 计算: 1/N batch

训练时: GPU 间 AllReduce 梯度。推理时: 无需 GPU 间通信（实例独立服务）。
```

DP 是训练中最基本的并行策略——每张 GPU 独立跑前向+反向，然后 AllReduce 梯度。推理中 DP 的用法更简单：每个推理实例独立服务一部分请求，**实例间不需要通信**。因此它也是推理部署的默认策略——vLLM 的 TP=1 部署本质就是 DP。

**DP 的 KV Cache**：完全不共享。每个实例维护自己那批请求的 KV Cache，一个实例的 Cache 对其他实例不可见。这也是为什么跨实例的 KV Cache 共享需要额外的方案（如 LMCache 的磁盘共享、Mooncake 的 RDMA 传输）。

---

## 三、张量并行（TP）：把权重矩阵切开

**切什么**：权重矩阵。将每一层的 Q/K/V 投影矩阵和 FFN 矩阵按列（或行）切成 N 份，N 张 GPU 各自持有一份，计算时各自算自己的切片，然后通信合并结果。

```text
GPU 0: [W_Q 第0-7头]   GPU 1: [W_Q 第8-15头]  GPU 2: [W_Q 第16-23头] GPU 3: [W_Q 第24-31头]
       [W_K 第0-7头]          [W_K 第8-15头]         [W_K 第16-23头]        [W_K 第24-31头]
       [W_V 第0-7头]          [W_V 第8-15头]         [W_V 第16-23头]        [W_V 第24-31头]
       [FFN 第1/4]            [FFN 第2/4]            [FFN 第3/4]            [FFN 第4/4]

       每张 GPU 的权重分片是唯一的——不存在重复。每层计算后 AllReduce 合并结果。
```

TP 的核心特征是 **每张 GPU 只有 1/N 的权重**——这正是它可以突破单卡显存上限的原因。代价是每层计算后都需要一次通信（AllReduce 或 AllGather），以 NVLink 的速度（~450 GB/s）这是可行的，但跨节点时延迟会急剧上升。因此 TP 通常**只在节点内使用**，TP size 不超过单节点的 GPU 数。

**TP 的 KV Cache**：KV Cache 也随权重一起被切分。每个 TP rank 只存储自己负责的那部分 K 和 V（按注意力头切分），推理时各 rank 各自计算自己那部分注意力的 QK 乘积，然后通过 AllReduce 合并。KV Cache 总量不变，但分散在 N 张 GPU 上，单卡显存压力降为 1/N。

**KV Cache offloading 时，TP=8 怎么存到外部存储？** 取决于注意力架构——LMCache 和 SGLang HiCache 的源码逻辑完全一致：

| 模型架构                    | 存储行为                                     | 物理文件                                              |
| --------------------------- | -------------------------------------------- | ----------------------------------------------------- |
| 标准 MHA/GQA（LLaMA、Qwen） | 8 个 rank 各自独立写自己的 shard             | `rank_0/block_N.bin` … `rank_7/block_N.bin`，8 份分片 |
| MLA（DeepSeek V2/V3）       | 只有 rank 0 写，其余 rank 被动等待 broadcast | `block_N.bin`，1 份完整文件                           |

原因：MLA 将 KV 压缩为 latent vector 后，KV Cache 体积仅为标准 MHA 的 1/4–1/8，一份完整文件在存储层不是瓶颈。标准 MHA 下每个 rank 独立 I/O，利用各 GPU 的 PCIe 带宽并行传输。LMCache 通过 `save_only_first_rank` 标志（`cache_engine.py` L113-115）、SGLang 通过 `is_mla_model` 分支（`hicache_storage.py` L335-336）控制这一差异。

**一个容易混淆的点**：如果 8 张 H100 做两组 TP=4（而非一组 TP=8），那这两组 TP 就是两个独立的推理实例——各有各的权重分片和 KV Cache，互不共享。TP 组内是「同生共死」的——8 张 GPU 组成一个 TP=8 实例，只有这一个实例的 KV Cache。如果挂了一张 GPU，整个 TP 组的 KV Cache 都不可用。这也是为什么生产环境通常用多组较小的 TP（如 2×TP=4）而非一组大 TP（1×TP=8）——前者虽然单组吞吐低，但故障域更小、调度更灵活。

---

## 四、流水线并行（PP）：把层切开

**切什么**：模型层。GPU 0 负责前 1/N 层，GPU 1 负责中间 1/N 层，以此类推。每张 GPU 上的权重是完整的，只是层数少。

```text
GPU 0: Layer 0-15 ──→ GPU 1: Layer 16-31 ──→ GPU 2: Layer 32-47 ──→ GPU 3: Layer 48-63
        ├─ 权重: 16 层               ├─ 权重: 16 层               ├─ 权重: 16 层
        ├─ KV Cache: 16 层            ├─ KV Cache: 16 层            ├─ KV Cache: 16 层
        └─ 激活: 传给下一张 GPU        └─ 激活: 传给下一张 GPU        └─ 激活: 传给下一张 GPU
```

PP 的通信模式是「接力」——每张 GPU 算完自己负责的层后，把激活值传给下一张 GPU。通信量比 TP 低（只传激活值，不传权重梯度），但存在**流水线气泡（Pipeline Bubble）**——第一张 GPU 在处理 token t+1 时，后面的 GPU 还在处理 token t，导致部分 GPU 空闲等待。

在推理场景中，PP 还面临一个特殊问题：**延迟放大**。一个 token 必须串行经过所有 PP stage 才能生成完毕，PP stage 越多，每个 token 的端到端延迟越大。因此纯 PP 在在线推理中不常用，更常见的是 PP+TP 混合：节点内 TP，节点间 PP。

**PP 的 KV Cache**：每张 GPU 只存自己负责的层的 KV Cache。KV Cache 总量不变，分散存储，单卡压力减小。

---

## 五、专家并行（EP）：MoE 的专属策略

**切什么**：专家。MoE 模型有多个专家（如 DeepSeek-V3 有 256 个路由专家），每次前向只激活少数几个（如 8 个）。EP 把专家分散到不同 GPU 上，每个 token 的前向计算只涉及少数 GPU。

```text
GPU 0: Expert 0-63       GPU 1: Expert 64-127     GPU 2: Expert 128-191    GPU 3: Expert 192-255
       ↑                        ↑                        ↑                        ↑
       └── token A 路由到 Expert 5  └── token B 路由到 Expert 80 └── token C 路由到 Expert 150

       每个 token 只激活 Top-K 专家，跨 GPU 通信只在有 token 路由过去时发生
```

EP 的核心优势是 **通信量天然低**——每个 token 只路由到少数专家，大部分 GPU 之间不需要通信。这也是为什么 DeepSeek-V3 可以在 671B 总参数下（每次激活 37B）以较低成本运行。

**EP 的 KV Cache**：与 TP 不同，EP 不切分注意力头——每个专家内部的注意力计算是完整的。KV Cache 通常与专家同址存储（每个 GPU 上的专家负责自己的 KV），但 DeepSeek 的 MLA（Multi-Head Latent Attention）将 KV 压缩为潜在向量后，KV Cache 的归属与专家计算分离，这部分涉及更复杂的 engineering。

---

## 六、序列并行（SP）：把序列长度切开

**切什么**：seq_len。将一个长序列切成 N 段，N 张 GPU 各处理一段，注意力计算跨 GPU 进行。

```text
GPU 0: token 0-1023  ─┐                  GPU 0: token 0-1023 ─┐
GPU 1: token 1024-2047 ┤ Ring Attention   GPU 1: token 1024-2047 ┤ 各自计算 +
GPU 2: token 2048-3071 ┤ (环状传递 K,V)   GPU 2: token 2048-3071 ┤ 跨 GPU 通信
GPU 3: token 3072-4095 ─┘                  GPU 3: token 3072-4095 ─┘
```

SP 主要用于**超长上下文**场景（如 128K+ tokens）。当单张 GPU 的显存放不下整个序列的 KV Cache 时，SP 将序列分段存储到多张 GPU。代价是注意力计算需要跨 GPU 通信——Ring Attention 是其中最具代表性的实现，它通过环状传递 K 和 V 实现在多 GPU 间计算完整注意力。

**SP 的 KV Cache**：KV Cache 按序列长度切分，每张 GPU 存一段。总 KV Cache 量不变，但分散后单卡可支持更长的上下文。

---

## 七、混合策略：真实部署都是组合拳

生产环境很少有纯单一策略。常见的组合：

```text
vLLM TP=4 部署 (单节点 8×H100):
  TP=4 在一个节点内切分权重 → 可以装下 70B 模型
  剩下的 4 张 GPU 做另一组 TP=4 → 两组实例做 DP

DeepSeek-V3 推理:
  节点内: TP=1 (MLA 架构单卡可跑) + EP=8 (256 个专家分散到 8 GPU)
  节点间: DP=多实例

70B 训练 (2 节点 × 8 GPU):
  节点内: TP=4 (权重切分，NVLink 通信)
  节点间: PP=2 (层切成两段，IB 通信)
  数据: DP (所有 GPU 并行处理不同 batch)
```

---

## 八、一张图看懂：谁切权重，谁切 KV Cache

> 可交互版本见 [并行策略可视化](parallelism_visual.html)——点击每种策略查看权重和 KV Cache 的切分示意。

---

## 九、相关资源

- [KV Cache 原理简介](../kv_cache/01_concepts/basic/kv_cache_原理简介.md) — TP 和 SP 对 KV Cache 的影响机制
- [Prefill 与 Decode 深度拆解](../kv_cache/01_concepts/basic/prefill_decode_qkv_calculation.md) — 理解 TP 通信量为什么必须低的计算背景
- [NCCL 通信路径逐层压测](../../03_ai_cluster_ops/03_nccl/06_nccl_path_benchmark.md) — TP 依赖的 NVLink 带宽实测
- [GPU 调度——拓扑感知](../../03_ai_cluster_ops/04_gpu_scheduling/03_topology_aware_scheduling.md) — TP 组为什么必须在同一节点
