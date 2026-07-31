# 大模型推理并行策略——DP、TP、PP、EP 到底在切什么

70B 的模型塞不进一张 80GB 的 H100——这是所有推理工程师都会遇到的第一道墙。解法不是换更大的 GPU，而是把模型拆开、把数据拆开、把计算拆开，让多张 GPU 协作完成推理。

「拆」的方式有五种：**DP（数据并行）、TP（张量并行）、PP（流水线并行）、EP（专家并行）、SP（序列并行）**。它们的名字听起来相似，但切的东西完全不同——有的切权重，有的切数据，有的切 KV Cache。本文从「切的到底是什么」出发，用统一的图示把五种策略讲清楚。

---

## 一、总览：五种策略，切三种东西

|             | DP        | TP         | PP       | EP       | SP       |
| ----------- | --------- | ---------- | -------- | -------- | -------- |
| 切的维度    | batch     | hidden     | layers   | experts  | seq_len  |
| 每张 GPU 有 | 完整模型  | 1/N 权重   | 1/N 层   | 1/N 专家 | 完整模型 |
| KV Cache    | 独立      | 分片       | 独立     | 独立     | 分片     |
| 通信量      | 无¹ / 高² | 极高       | 中       | 低       | 中       |
| 典型场景    | 训练      | 大模型推理 | 超大模型 | MoE 推理 | 长上下文 |

> ¹ 多实例推理 DP：每个实例是独立进程，无通信。<br>
> ² 训练 DP / vLLM 引擎内 DP：rank 间有梯度同步或元数据通信。<br>
> **可交互版本**：[并行策略可视化](parallelism_visual.html)——点击每种策略查看权重、KV Cache 和数据在各 GPU 上的切分方式。

---

## 二、数据并行（DP）

DP 在不同语境下含义不同，先说清楚这三种场景，再聚焦本文的核心——多实例推理 DP。

| 语境               | 模型副本                         | GPU 间通信          | 调度方式                | 典型用法                               |
| ------------------ | -------------------------------- | ------------------- | ----------------------- | -------------------------------------- |
| **训练 DP**        | 完全相同                         | 每步 AllReduce 梯度 | 框架（DDP/FSDP）        | 所有分布式训练                         |
| **多实例推理 DP**  | 完全相同                         | **无**              | 外部负载均衡器          | LLaMA/Qwen 单卡部署                    |
| **vLLM 引擎内 DP** | 可不同（MoE 下各 rank 专家不同） | 元数据同步          | vLLM scheduler 显式分配 | `--data-parallel-size N`，多与 EP 联用 |

本文聚焦**多实例推理 DP**——这是推理部署中最常见的 DP 形式。vLLM 引擎内 DP（`--data-parallel-size`）将在下文中单独说明。

### 多实例推理 DP

**场景**：你有 4 张 H100，要跑 4 个 LLaMA-70B 的推理实例，每个实例独立服务不同的用户请求。所有 4 个实例用同一份模型权重，不需要互相通信——这是最简单的并行方式，也是推理部署的默认策略。

**问题**：单 GPU 能装下整个模型，但一个实例的吞吐有限——同一时间只能服务有限数量的请求。你需要把请求分发到多个实例上，但每个实例的 KV Cache 是隔离的：实例 A 算过的 prompt，实例 B 不知道，如果 B 后来遇到同样的 prompt 还得重算。

```text
GPU 0: [模型副本 0] ← 请求 1,2      GPU 1: [模型副本 1] ← 请求 3,4
        ├─ 权重: 完整                     ├─ 权重: 完整
        ├─ KV Cache: 独立                  ├─ KV Cache: 独立
        └─ 计算: 1/N batch                 └─ 计算: 1/N batch

训练时: GPU 间 AllReduce 梯度。推理时: 无需 GPU 间通信。
```

**方案**：起多个独立的 vLLM 或 SGLang 服务进程，每个绑定不同的 GPU（如 `CUDA_VISIBLE_DEVICES=0`），前面挂一个负载均衡器（如 Nginx、Envoy 或 Ray Serve）。每个实例是完全自治的——独立加载权重、独立管理 KV Cache、独立处理请求。不需要任何 GPU 间通信，也不需要框架层面的 DP 感知。

**权衡**：KV Cache 不共享是多实例 DP 最大的代价。实例 A 算过的 prompt，实例 B 不知道——如果 B 后来遇到同样的 prompt，Prefill 还得重做。跨实例的 Cache 复用需要额外的基础设施——LMCache 用磁盘做共享存储，Mooncake 用 RDMA 传输，HiCache 用 Mooncake 后端。这些方案的复杂度远高于单实例部署。

### vLLM 引擎内 DP（`--data-parallel-size`）

另一种 DP 形态：引擎内原生 DP。与多实例推理 DP 不同，多个 DP rank 共享同一个 scheduler 进程，请求被显式分配到不同 rank，rank 间通过 `_synchronize_dp_ranks()` 同步 batch 元数据（ubatch 大小、cudagraph 模式）。多节点时通过 `--data-parallel-backend ray` 用 Ray placement group 编排。SGLang 目前没有引擎内 DP。

```text
Dense 模型（LLaMA/Qwen）：
  --data-parallel-size 4 --tensor-parallel-size 2 (单节点 8 GPU)
  → 4 个 DP rank，每个 rank 是 TP=2 的模型副本
  → 每 rank 完整模型权重，rank 间共享同一份 KV Cache pool
  → 当前主要用于 testing/development，生产环境更常用多实例推理 DP

MoE 模型（DeepSeek V4 Pro）：
  --data-parallel-size 8 --enable-expert-parallel
  → 8 个 DP rank，每个 rank 的 attention 权重相同，expert 分片不同
  → `--enable-expert-parallel` 是关键：专家分布在 DP rank 间，路由协同
```

引擎内 DP 的 DP rank 间**存在元数据通信**——`_synchronize_dp_ranks()` 同步 ubatch 大小、cudagraph 模式等。这与多实例推理 DP 的"零通信"不同。当前引擎内 DP 在 Dense 模型上主要用于 testing/development，生产级 Dense 模型部署仍以多实例推理 DP 为主；在 MoE 模型上则是与 EP 联用的生产级方案。

---

## 三、张量并行（TP）

**场景**：70B 的 LLaMA-2 在 FP16 下需要 ~140GB 显存，一张 80GB 的 H100 装不下。DP 不能解决这个问题——DP 只复制模型，每张卡还是需要装完整的 140GB。

**问题**：你需要把单层权重矩阵切成小块，让每张 GPU 只持有 1/N，从而突破单卡显存上限。但这引入了新的问题——每张 GPU 只有部分权重，如何拼出完整的前向计算结果？

```text
GPU 0: [W_Q 第0-7头]   GPU 1: [W_Q 第8-15头]  GPU 2: [W_Q 第16-23头] GPU 3: [W_Q 第24-31头]
       [W_K 第0-7头]          [W_K 第8-15头]         [W_K 第16-23头]        [W_K 第24-31头]
       [W_V 第0-7头]          [W_V 第8-15头]         [W_V 第16-23头]        [W_V 第24-31头]
       [FFN 第1/4]            [FFN 第2/4]            [FFN 第3/4]            [FFN 第4/4]

       每张 GPU 的权重分片是唯一的——不存在重复。
```

**方案**：Megatron-style TP 是业界标准——每层 Attention 和 FFN 被拆成一对 column-parallel + row-parallel 矩阵。列切（column-parallel）的矩阵各 GPU 输入相同（X 被复制到所有 rank），各自算完产生分片输出，不需要通信；紧接着的行切（row-parallel）矩阵接收上一层分片后的输入，各自算完本地结果后 **AllReduce 求和合并**，产生完整输出传给下一层。vLLM 的 `--tensor-parallel-size 4` 和 SGLang 的 `--tp-size 4` 都遵循此实现。

KV Cache 也随权重一起按 head 切分：每个 TP rank 只存自己负责的那部分 head 的 K 和 V。总量不变，但分散后单卡显存压力降为 1/N。

**权衡**：TP 是突破单卡显存上限最直接的方式，代价是**通信量极高**——每层 Attention 和 FFN 之后都需要 AllReduce。在 NVLink 互联的同一节点内这可以接受（~450 GB/s，延迟 ~5μs），但跨节点时延迟上升 5-10 倍，性能不可接受。因此 TP 的硬约束是：TP size ≤ 单节点 GPU 数，TP 组内所有 GPU 必须在同一 NVSwitch 域内。

另一个权衡是故障域：TP 组内任何一张 GPU 挂掉，整个组的 KV Cache 都不可用。生产环境通常用多组较小的 TP（如 2×TP=4）而非一组大 TP（1×TP=8），牺牲单组吞吐换取更小的故障域。

**KV Cache offloading 在 TP 下的行为**（LMCache 与 SGLang HiCache 源码逻辑一致）：

| 模型架构                    | 存储行为                                 | 物理文件                                    |
| --------------------------- | ---------------------------------------- | ------------------------------------------- |
| 标准 MHA/GQA（LLaMA、Qwen） | N 个 rank 各自独立写自己的 shard         | `rank_0/block_N.bin` … `rank_N/block_N.bin` |
| MLA（DeepSeek V2/V3）       | 只有 rank 0 写，其余 rank 被动 broadcast | `block_N.bin`，1 份完整文件                 |

MLA 将 KV 压缩后体积仅为标准 MHA 的 1/4–1/8，一份完整文件不是瓶颈。LMCache 通过 `save_only_first_rank`（`cache_engine.py` L113）、SGLang 通过 `is_mla_model`（`hicache_storage.py` L335）控制此差异。

---

## 四、流水线并行（PP）

**场景**：模型有 64 层，单节点 8 张 GPU 做了 TP=4 已经装下了权重，但还有 4 张 GPU 闲着。或者模型大到 TP=8 也不够——需要跨节点了。

**问题**：跨节点的 TP 不可行（NVLink 只在节点内）。你需要一种通信量更低的并行方式，能在跨节点时将模型拆开，而不需要每层都 AllReduce。

```text
GPU 0: Layer 0-15 ──→ GPU 1: Layer 16-31 ──→ GPU 2: Layer 32-47 ──→ GPU 3: Layer 48-63
        ├─ 权重: 16 层               ├─ 权重: 16 层               ├─ 权重: 16 层
        ├─ KV Cache: 16 层            ├─ KV Cache: 16 层            ├─ KV Cache: 16 层
        └─ 激活: 传给下一张 GPU        └─ 激活: 传给下一张 GPU        └─ 激活: 传给下一张 GPU
```

**方案**：PP 按层切分——每张 GPU 负责一段连续的层，每段内的权重完整。通信只在层边界发生：当前 GPU 算完最后层的激活值，传给下一张 GPU 的第一层。通信量远低于 TP（只传激活值，不传权重梯度）。vLLM 通过 `--pipeline-parallel-size 2`、SGLang 通过 `--pp-size 2` 启用。

**权衡**：PP 的主要代价是**流水线气泡和延迟放大**。气泡来自串行依赖——GPU 0 在处理 token t+1 时，GPU 1 还在处理 token t，越靠后的 GPU 等待越久。推理中延迟随 stage 数线性增长——每个 token 必须串行经过所有 PP stage。因此纯 PP 在在线推理中不常用，更常见的是 PP+TP 混合：节点内 TP=4 突破单卡显存，节点间 PP=2 突破单节点显存。

KV Cache 按层分段——PP stage 0 的 GPU 不存 stage 1 的 KV。读取历史 KV 时可能触发跨 stage 查询，需要框架的 block table 支持跨 stage 寻址。

---

## 五、专家并行（EP）

**场景**：DeepSeek-V3 有 671B 总参数、256 个路由专家。如果所有专家都装在一张 GPU 上，显存会瞬间爆炸。但它的 MLA 架构让单 token 的计算量很小（每次只激活 37B），问题不在计算而在存储。

**问题**：你需要把 256 个专家分散到多张 GPU 上，每个 token 的前向计算只经过少数被路由到的专家所在的 GPU。但这些专家的权重怎么分布？token 到专家的路由怎么跨 GPU 通信？

```text
GPU 0: Expert 0-63       GPU 1: Expert 64-127     GPU 2: Expert 128-191    GPU 3: Expert 192-255
       ↑                        ↑                        ↑                        ↑
       └── token A→Expert 5     └── token B→Expert 80   └── token C→Expert 150

       每个 token 只激活 Top-K 专家，跨 GPU 通信只在有 token 路由过去时发生
```

**方案**：每个 GPU 固定持有 1/N 的专家权重。路由层计算 token 到专家的匹配分数后，通过 All-to-All 通信将 token 的隐藏状态发送到对应专家所在的 GPU，专家计算完成后 All-to-All 返回。DeepSeek-V3 在 vLLM 和 SGLang 中均默认 EP=8（256 专家 ÷ 8 GPU = 每卡 32 专家）。

**权衡**：EP 的最大优势是通信量天然低——每个 token 只路由到 Top-K 专家（K=8），大部分 GPU 之间不需要通信。代价是负载均衡：如果某几个专家特别热门（token 大量涌向 Expert 5），对应的 GPU 会过载。SGLang 的 EPLB（Expert Parallel Load Balancer）动态监控各专家负载，将热门专家副本迁移到空闲 GPU 来解决。vLLM 通过 `KVCacheGroupSpec` 管理 EP 下的 KV 分布。

EP 是 MoE 模型的专属策略，Dense 模型（LLaMA/Qwen）不需要。KV Cache 与专家同址存储——token 的 attention 计算和 KV 留在被路由到的 GPU 上，不需要跨 GPU 搬运。

---

## 六、序列并行（SP）

**场景**：上下文窗口从 4K 扩展到 128K，单张 GPU 的 HBM 放不下一个请求的完整 KV Cache。模型权重能装下，但 KV Cache 爆了。

**问题**：你需要把序列长度切成 N 段，每段放到一张 GPU 上，让多张 GPU 协作完成一个超长请求的注意力计算。但注意力需要每个 token 看到所有其他 token——KV 被切分在不同 GPU 上，如何计算完整的 softmax？

```text
GPU 0: token 0-1023  ─┐                  GPU 0: token 0-1023 ─┐
GPU 1: token 1024-2047 ┤ Ring Attention   GPU 1: token 1024-2047 ┤ 各自计算 +
GPU 2: token 2048-3071 ┤ (环状传递 K,V)   GPU 2: token 2048-3071 ┤ 跨 GPU 通信
GPU 3: token 3072-4095 ─┘                  GPU 3: token 3072-4095 ─┘
```

**方案**：Ring Attention 是当前主流的 SP 实现。GPU 间按环状传递 K 和 V 块：每步每个 GPU 把自己的 K/V 块发给下一个 GPU，同时接收上一个 GPU 发来的块，用新收到的块更新本地的 softmax 累加器。经过 N-1 步环传递后，每张 GPU 都看到了完整的 K/V，注意力计算等价于单 GPU 版本。

**权衡**：SP 的通信开销随序列长度线性增长，在短上下文下不值得。它通常作为 TP 的补充——TP 已经切了 head 维度（减少单卡权重），SP 再切 seq_len 维度（减少单卡 KV Cache）。两者叠加时 KV Cache 在 head 和 seq_len 两个维度上同时被切分。

SP 在 vLLM 和 SGLang 中没有独立的 `--sp-size` 参数，通常以特定 attention 后端的形式出现（如 flashinfer），或通过 DeepSpeed Ulysses 等三方库实现。

---

## 七、混合策略：真实部署都是组合拳

生产环境很少有纯单一策略——每种策略解决一个维度的瓶颈，真实部署是把它们叠起来用。

### 7.1 组合原则

**TP 是基础层**：如果单卡装不下模型权重，先用 TP 在节点内切分。TP 的通信只在 NVLink 内可行，所以 TP 组必须在同一节点。

**PP 是 TP 不够时的扩展**：当模型大到单节点塞不下（如 405B+ LLaMA-3 在 8×H100 上也装不下），用 PP 将层切开跨节点部署。PP 的通信量比跨节点 TP 低一个数量级。

**DP 是吞吐放大器**：在 TP/PP 把模型装下之后，用多组实例（多实例推理 DP）并行服务更多请求。多实例推理 DP 不需要 GPU 间通信，是最廉价的横向扩展。注意与 vLLM 引擎内 DP（`--data-parallel-size`）的区别——后者 rank 间存在元数据通信，主要用于 MoE+EP 场景。

**EP 是 MoE 的专属维度**：与 TP/PP 正交——专家分散不影响 attention 层的计算，两者可以独立配置。

### 7.2 真实部署案例

**案例一：LLaMA-70B on 8×H100（高吞吐推理）：**

```text
硬件: 1 节点 × 8×H100 (80GB)
策略: 2×TP=4 + DP

  TP 组 A (GPU 0-3):  权重各 1/4 → 单卡 ~35GB，可以装下
  TP 组 B (GPU 4-7):  同上

  两组之间做 DP → 两个独立推理实例，各自服务请求
  总吞吐 ≈ 2× 单组吞吐，KV Cache 各自独立
```

为什么不用 TP=8？TP=8 的单组吞吐略高于 2×TP=4，但故障域大（一张 GPU 挂全组停）、调度不灵活（8 卡必须同时空闲）。在推理服务中，2×TP=4 + DP 是更常见的选择。

**案例二：DeepSeek-V3 on 8×H100：**

```text
硬件: 1 节点 × 8×H100 (80GB)
策略: TP=1 + EP=8

  TP=1: MLA 架构下单卡可装下权重（KV 被压缩为 latent vector）
  EP=8: 256 专家 → 每 GPU 32 个专家，token 路由到 Top-8

  节点间可再加 DP，形成多组 EP=8 实例
```

DeepSeek-V3 是少数不需要 TP 的模型——MLA 的 KV 压缩效果远超传统 MHA，单卡即可容纳。这也是为什么它的推理成本远低于同规模的 Dense 模型。

**案例三：LLaMA-405B on 2 nodes × 8×H100（训练）：**

```text
硬件: 2 节点 × 8×H100 (80GB)
策略: TP=4 + PP=2 + DP

  每节点内: TP=4 (NVLink) → 4 GPU 组成一个 TP 组
  跨节点: PP=2 → 节点 0 负责前 63 层，节点 1 负责后 63 层
  数据: DP 在所有 GPU 上并行处理不同 batch

  总 GPU 利用率 ≈ 16 张卡，训练吞吐取决于 PP 气泡大小
```

### 7.3 按瓶颈选策略

**单卡装不下权重** → 先试 TP。TP 把每层的权重矩阵切成 N 份，让单卡只需要 1/N 的显存放权重。这是解决显存瓶颈最直接的方式。如果模型大到单节点全部 GPU 做 TP 也装不下（如 LLaMA-405B 在 8×H100 上仍然超了），才需要 PP 做跨节点层切分——但 PP 的延迟放大在推理中尤其明显（每个 token 串行经过所有 stage），能不用就不用。

**MoE 模型专家太多** → EP 是唯一解。DeepSeek-V3 的 256 个专家必须分散到多张 GPU，每卡固定持有 1/N 的专家权重。EP 与 TP 正交——EP 切的是专家，TP 切的是注意力层的权重，两者可以独立配置。Dense 模型（LLaMA/Qwen）不需要 EP。

**单实例吞吐不够** → 先加 DP。DP 不需要 GPU 间通信，是最廉价的横向扩展——多起几个实例，前面挂个负载均衡器即可。如果单 GPU 显存本身紧张（模型较大），再考虑 DP+TP：用 TP 扩大单个实例的容量（让单个实例能装下更大的模型），用 DP 增加实例数量（让更多请求能并行处理）。

**超长上下文 KV 放不下** → SP。当上下文窗口超过单卡 HBM 容量，TP 帮不上忙——TP 切的是权重，不是 KV。Ring Attention 把序列长度切成 N 段分散到 N 张 GPU，每张只需要存 1/N 的 KV。但在短上下文（< 32K）下 Ring Attention 的通信开销不值得——先从长上下文是否真的需要开始确认。TP+SP 叠加时 KV 在 head 和 seq_len 两个维度上同时切分，单卡压力进一步减小。

**热门专家 GPU 不均衡** → 这是 MoE 模型特有的负载均衡问题。如果 EP=8 时某个专家（如 Expert 5）接收了远超平均的 token 量，它所在的 GPU 成为瓶颈。SGLang 的 EPLB 会自动监控并将热门专家副本迁移到空闲 GPU，vLLM 目前需要手动调整 EP size 或专家分布。这个问题在 Dense 模型中不存在——Dense 模型每层所有参数都参与计算，天然负载均衡。

---

## 八、相关资源

- [KV Cache 原理简介](../kv_cache/01_concepts/basic/kv_cache_basics.md) — TP 和 SP 对 KV Cache 的影响机制
- [Prefill 与 Decode 深度拆解](../prefill_decode/prefill_decode_qkv_calculation.md) — 理解 TP 通信量为什么必须低的计算背景
- [NCCL 通信路径逐层压测](../../03_ai_cluster_ops/03_nccl/06_nccl_path_benchmark.md) — TP 依赖的 NVLink 带宽实测
- [GPU 调度——拓扑感知](../../03_ai_cluster_ops/04_gpu_scheduling/03_topology_aware_scheduling.md) — TP 组为什么必须在同一节点
