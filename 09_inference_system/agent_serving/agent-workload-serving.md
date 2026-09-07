# 当 Agent 流量成为推理系统的主要负载

> 过去推理系统服务的是「人打字」：请求短、无状态、分钟级会话。Agent 把这件事倒了过来：一次任务由程序连续发起 10–20 次调用，token 耗用是普通对话的 5–30 倍（Gartner 预测口径）。当主要流量从人变成 Agent，推理系统赖以设计的四个隐含假设逐条失效。
>
> 2026-09 | 基于 vLLM（`43d691ec6b`，2026-08-07）与 SGLang（`f7101b0ae6`，2026-08-18）源码验证；vLLM Router 部分转引本仓库 [router.md](../vllm/routing/router.md)（Router 仓库未本地核验，已标注）
>
> **性质说明**：机制与行为均经源码验证；涉及成本/耗时的数字为量级示意，用于建立直觉，落地前需按自有硬件实测；外部文献数据按原文口径转引。

---

## 一、负载在换人：Agent 流量长什么样

Chat 时代，推理系统对负载有一组很少被写下来的隐含假设。它们如此自然，以至于散落在所有调度、缓存代码的默认行为里：

| 隐含假设                   | Chat 负载的现实                  | Agent 负载的现实                                     |
| -------------------------- | -------------------------------- | ---------------------------------------------------- |
| 请求相互独立、无状态       | 每条消息独立到达，无跨请求约束   | 一次任务 = 有状态的调用序列，每轮都要带上全部历史    |
| KV 生命周期 = 请求生命周期 | 请求结束，KV 即可释放            | 会话存活贯穿数十次调用，中间夹着工具执行的「空窗」   |
| 上下文短且形态稳定         | 几百到几千 token，系统提示词为主 | 上下文单调增长，工具结果（网页、代码、日志）大量涌入 |
| 容量按并发数规划           | QPS × 平均时长，前缀命中率稳定   | 命中率随任务类型剧烈波动，重试与并行分支放大流量     |

数据侧，这个趋势已经可以量化：Agent 完成一件事要调用模型 10–20 次，token 耗用是普通对话的 5–30 倍（Gartner 预测，转引自 [Token Factory 素材 05](../../99_misc/token_factory_talk/references/05-jevons-demand-future.md)）；HF Hub 的 Agent 流量监测显示，2026 年 4 月仅 Claude Code 一类编码 Agent 就占流量的 67.8%，7 月降至 44.4%，份额下降不是 Agent 退潮，而是十多个新客户端涌现带来的生态分化【素材 08】。驱动力已经从「人」切换到「机器」。

微观形态上，AgentSysBench 对 10 个代表性 Agent 应用、6 万余次模型调用与 11 万余次工具调用的测量给出了第一份系统画像（[From LLM Inference to Agentic Workloads](https://arxiv.org/abs/2608.15127)，2026-08）：会话中位活跃计算时间仅 20%，其余时间「活着但空闲」；空闲期导致的 KV 重算（re-prefill）占 cache 相关 token 的 55.9%、总成本的 31.5%；工具定义与系统指令等控制面元数据最高占上下文窗口的 84.3%；长短请求混批时，队头阻塞可使快请求减速 7.1–35.7 倍、小请求 TPOT 恶化至 +79.7%。这些数字把第二节要展开的四个连锁问题，从推测变成了实测。

Agent 流量不是把 Chat 流量「放大」，而是把它变形。下面四个连锁问题，每一个都能在源码里找到引擎的应对，以及应对不了的部分。

---

## 二、四个被打破的假设，四个连锁问题

### 2.1 KV 生命周期错配：逐出还是保留，从偶发变成常态

Agent 会话的存活时间远长于单次请求。两轮调用之间，系统要做一个 Chat 时代几乎不存在的决定：

- **逐出**：下轮 turn 到来时重算整条轨迹。一个累积了 50K token 历史的 Agent 会话，若 KV 已被逐出，resume 时的 prefill 需要重读全部历史：按单卡 10–20K tok/s 的 prefill 量级估算，就是一次 2.5–5 秒的 TTFT 尖刺（量级示意，按硬件实测为准）。对一个每轮都在等待工具返回的交互式 Agent，这个尖刺出现在每一轮。
- **保留**：空闲会话的 KV 持续占用显存。并发会话数一多，「看不见的会话」吃掉的显存比活跃请求还多，挤压的是正在 decode 的 batch。

Agent 会话的 KV 命中率直接决定 TTFT 形态：Chat 场景下 prefix cache miss 的代价是几百毫秒，Agent 场景下是数秒。

### 2.2 调度语义失真：队列里混着两种完全不同的东西

Agent 流量进入队列后，调度器看到的是两类请求的混合体：

1. **活跃请求**：正在 prefill/decode，对 TTFT/TPOT 敏感；
2. **休眠会话**：工具执行中，下一轮几秒到几分钟后才回来；但它的 KV 若还占着显存，就在持续支付「租金」。

更麻烦的是重试带来的长 prefill 冲击：一次 50K token 的重试 prefill，足以让同 batch 的所有 decode 请求的 TPOT 抖动一个量级。Chat 时代靠 chunked prefill 缓解的问题，在 Agent 时代被放大，因为 Agent 的 prefill 特别长，重试又频繁。

### 2.3 路由的粘性需求：会话要「记住」实例

多实例部署下，Agent 会话的第 N 轮请求若被路由到一个没见过它的实例，前缀缓存全部失效，2.1 的 TTFT 尖刺直接兑现。Chat 时代负载均衡追求「均匀」，Agent 时代要追求「亲和」，而这两个目标在容量紧张时互相冲突。

### 2.4 容量规划失真：QPS 公式失效

Chat 容量公式：`并发数 = QPS × 平均请求时长`。Agent 场景下，决定显存的不只是并发请求，还有并发会话数 × 会话 KV 增长速率 × 命中率。命中率取决于任务类型组合：一批编码 Agent 和一批客服 Agent 的 KV 增长曲线完全不同，按平均数规划的容量会在任务类型切换时震荡。

---

## 三、引擎现状：四个问题，源码里已有什么答案

以下均基于 vLLM `43d691ec6b`、SGLang `f7101b0ae6` 源码验证。

### 3.1 KV 生命周期：radix tree + 分层存储（SGLang 走得最远）

SGLang 的 RadixCache 是会话复用的核心载体：

- `match_prefix`（`python/sglang/srt/mem_cache/radix_cache.py:376`）按前缀树匹配历史 KV；
- `inc_lock_ref` / `dec_lock_ref`（`radix_cache.py:622`、`radix_cache.py:637`）通过引用计数把「正在使用」的子树排除在逐出之外（`evictable_size_` 相应增减）；
- 逐出按 LRU（`last_access_time`，`radix_cache.py:248`）执行 `evict`（`radix_cache.py:592`）。

这套机制对 Agent 负载的含义：只要下一轮请求的 prompt 前缀还在树里、没有被逐出，resume 就是「免费」的。但 radix tree 默认只活在一个进程的显存里：重启即失，多实例间不共享。

分层缓存把「会话记忆」扩展到显存之外：`--enable-hierarchical-cache`（`python/sglang/srt/server_args.py:2668`，默认关闭）启用 HiCache 后，L1（GPU）逐出的 KV 写回 L2（CPU）/L3（远程存储），下轮命中 L2 的恢复代价远低于重算（架构细节见 [HiCache 深入详解](../sglang/hicache_deep_dive.md)）。Agent 负载下，HiCache 应当作会话记忆层来配置，而不是可开可关的调优项。

vLLM 侧：prefix caching 在 V1 已默认开启（`vllm/config/cache.py:93`，`enable_prefix_caching: bool = True`），复用依据是块级前缀哈希：`get_computed_blocks`（`vllm/v1/core/kv_cache_manager.py:229`）调用 `find_longest_cache_hit`（`kv_cache_manager.py:261`）定位最长命中，请求结束由 `free`（`kv_cache_manager.py:567`）归还块，同前缀的后续请求仍可命中未被覆盖的块。但 vLLM 没有「为某会话保留 KV」的租约/TTL 原语：命中是统计性的，显存压力下 Agent 会话的历史块随时可能被新请求覆盖。

### 3.2 调度：请求级优先级已有，会话语义没有

先纠正一个常见印象：两个引擎都已具备请求级优先级。

SGLang：

- 调度策略可选 `lpm / random / fcfs / dfs-weight / lof / priority / routing-key`（`server_args.py:828-846`），默认 `fcfs`；对前缀复用最友好的 `lpm`（最长前缀优先）需要显式开启；
- `--enable-priority-scheduling` 开启后按请求 priority 字段调度，且此时 `schedule_policy` 必须是 `fcfs` 或 `lof`（`server_args.py:9246-9249`）；
- 显存不足时 `retract_decode`（`python/sglang/srt/managers/schedule_batch.py:2816`）逐个收回 decode 请求直至内存可容纳，若仍不够，则中止最后一个请求而不是让调度器崩溃；收回顺序由 `_get_decode_retraction_order`（`schedule_batch.py:2869` 起）决定，默认保留「已生成最少输出、输入最长」的请求，`retraction_policy=priority` 时优先保护高优先级请求；
- `schedule_conservativeness`（`server_args.py:886`，默认 1.0）是防频繁 retract 的调节阀，调大即更保守地准入。

vLLM V1：抢占受害者按 `(priority, arrival_time)` 最小者选中（`vllm/v1/core/sched/scheduler.py:592-594`），被抢占请求重新入队等待重算。

缺的是什么：以上所有原语都以「请求」为单位。没有任何机制回答「这个请求属于哪个会话」「这个会话已经花了多少预算」「这批互为分支的请求应该同进同退」。引擎外围已出现会话语义的雏形：NVIDIA Dynamo 的 KV 路由支持 priority/latency 两类 agentic hints，让用户面 turn 插队后台任务；llm-d 的 Endpoint Picker 正在社区推进 session-affinity 路由（[issue #177](https://github.com/llm-d/llm-d-inference-payload-processor/issues/177)）。但它们都是网关层的「拼接」，vLLM/SGLang 内核里仍没有会话原语。这条裂缝有多宽，从学术界的反应可见一斑：Autellix 把 agent 程序整体作为一等公民重新调度（同等延迟下吞吐 4–15×），Helium 从数据系统视角重做 workflow-aware serving。调度层被整体重造，说明参数调优弥合不了。

### 3.3 路由：前缀感知已有，会话粘性靠应用层

vLLM Router 的 `cache_aware` 策略在每个 worker 侧维护近似 radix 树追踪已缓存前缀，按匹配率路由（匹配率阈值 `cache_threshold=0.5`，缓存条目 30 秒周期清理；机制详解见 [vLLM Router 架构解析](../vllm/routing/router.md)，Router 为独立 Rust 项目，本文未对源码本地核验）。

对 Agent 负载的启示与缺口：

- `cache_aware` 解决的是**前缀维度的亲和**：系统提示词、工具 schema 这类稳定前缀收益最大；
- **会话维度的粘性**（同一会话的 N 轮请求尽量落同一实例）没有内建：radix 树追踪的是「哪些前缀在哪个节点上」，不是「哪个会话属于哪个节点」。生产路由层的现状是「有意识、无标准」：Dynamo 维护跨 worker 的 KV block 全局索引（Flash Indexer），llm-d 的 EPP 把 session-affinity 路由列为社区议题，但都未成为跨引擎的通用机制。实践中仍靠应用层网关按 session id 做一致性哈希，或接受外置 KV 层来解耦（见 4.3）；
- 多引擎混合部署时，路由层无法感知 LMCache/Mooncake 里的 KV 分布，KV 感知路由目前停留在各自体系内（分布式 KV 层见 [LMCache 架构概览](../kv_cache/02_systems/lmcache/lmcache_overview.md)、[Mooncake 架构详解](../kv_cache/02_systems/mooncake/mooncake_architecture.md)）。

### 3.4 容量：把规划公式从 QPS 换成会话三要素

容量公式的迁移：

```text
Chat：  并发请求数 ≈ QPS × 平均请求时长
Agent： 并发会话成本 ≈ Σ (活跃请求数 × 请求 KV)
        + Σ (休眠会话数 × 会话 KV × 保留概率)
        + Σ (每轮新增 token × (1 − 会话命中率) 的重算峰值)
```

方法论层面的支撑已经齐备：KV 容量推演的框架见 [GLM-5 KV Cache 容量规划](../kv_cache/01_concepts/capacity_planning/glm5_kv_cache_capacity_planning.md)，服务产能与承诺的 Goodput 框架见 [Token 工厂的产能怎么算？](../../99_misc/token_factory_talk/cluster-capacity-measurement.md)。新的工作是把 Goodput 的「成功」从请求级改成 turn 级：一个工具调用失败重试 3 次的 turn，Goodput 该怎么记？现有框架没有答案。

---

## 四、三个没有标准答案的权衡

### 4.1 粘性 vs 均衡

亲和路由提升命中率，但把同类会话压到同一实例，制造热点与长尾。vLLM Router 用 `cache_threshold`（0.5）画了一条线：匹配率过线才按亲和路由，否则回落到负载均衡，本质上是承认亲和是概率性的、均衡是兜底的。Agent 流量下这条线的位置需要按任务类型重调：编码 Agent 的会话内前缀增长率高，过线请求复用价值大；而系统提示同质化的客服 Agent，任何实例命中率都高，亲和的边际价值低。

### 4.2 保留 vs 重算

直觉上的成本模型（量级示意）：

```text
保留成本 = 会话 KV 字节 × 保留时长 × 单位显存租金
重算成本 = 历史 token 数 / prefill 吞吐 × 等待时间价值
```

变量是保留时长。Agent 会话的工具间隔（秒到分钟）恰好落在「显存租金不可忽略、重算又不便宜」的区间。新架构把天平进一步推向重算：CSA 把 KV 压到 1/4，KDA 干脆不产生 token 级 KV（见 [post-kv-cache-era-challenges.md](../post-kv-cache-era-challenges.md)），重算成本在快速通缩。今天按 H100+MHA 口径设计的粘性/外置 KV 方案，换到新架构模型上可能突然不划算。会话基础设施的设计要把「重算变便宜」当作一阶趋势来对冲。

### 4.3 会话状态放哪：三层选型

| 层               | 代表                                                                                                                | 优势                                 | 代价                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ------------------------------------------ |
| 引擎内 radix     | SGLang RadixCache、vLLM prefix hash                                                                                 | 零额外跳数，命中即免 prefill         | 与进程共生：重启即失、多实例不共享、无 TTL |
| 外置 KV 存储     | LMCache（L2/L3）、Mooncake                                                                                          | 跨实例/跨引擎共享，会话记忆持久      | 恢复走网络/盘，多一跳运维复杂度            |
| 应用层上下文重组 | 压缩/摘要（[Claude Code 上下文压缩](../../08_agentic_system/context/claude-code-context-compression.md)）、状态外化 | 把「存」的问题换给「算」，引擎零改造 | 摘要有信息损失，重组本身是一次 prefill     |

选型不是单选，多数生产系统是「引擎内 radix 保热点 + 外置存储保长尾 + 应用层压缩控总量」的三明治。要避免的是默认态：什么都不配，每轮 resume 都交全额重算税。

---

## 五、怎么度量：三个 Agent 形态指标与缺口清单

现有 benchmark（`benchmark_serving`、`sglang bench`）的流量形态是 Chat 的：独立请求、稳定前缀。Agent 负载需要自己的度量：

1. Turn-resume TTFT：同一会话上一轮结束后 T 秒（模拟工具执行）发起来轮请求，测 TTFT 分布。它同时暴露逐出策略与分层缓存的实效，Chat TTFT 测不出这些。
2. 会话 KV 命中率：按会话聚合的 prefix cache 命中 token 占比，随任务类型/会话长度分布，而不是集群平均值。
3. 每任务 token 成本：含重试与分支放大的、按「任务」而非「请求」聚合的 token 消耗，容量规划与成本归因都该以它为单位。

缺口清单（截至 2026-09，按源码现状）：

- 引擎无会话原语：session id → 亲和/配额/分组抢占的映射，只能应用层拼；
- KV 无租约/TTL：「保留多久」没有声明式接口，逐出策略对所有流量一视同仁；
- KV 感知路由不跨引擎、不跨存储层（Router 只见引擎、LMCache 只见自己）；
- Agent 形态 benchmark 标准化刚起步：MLCommons 已于 2026-07 启动 [MLPerf Agentic Inference](https://mlcommons.org/2026/07/agentic-inference-for-mlperf-inference/)（多轮、上下文持续增长、闭环工作流），学术界已有 AgentSysBench 这样的负载画像工具，但 turn-resume TTFT、会话 KV 命中率尚不在任何标准测法中。

实测待办（本文数字均未实测，为量级示意）：同一 20 轮工具调用 trace（每轮间隔 5–30 s），分别在「radix 开/关」「HiCache 开/关」下测 turn-resume TTFT 与每任务 token 成本。这一步做完，本文才能从机制分析变成容量依据。

---

## 结语

Chat 时代推理系统的一等公民是请求，Agent 时代要换成会话。请求级的机制引擎已备齐：优先级、抢占、前缀复用、分层逐出都有源码级的答案；会话级的原语（粘性、租约、分组、turn 级度量）还全部空缺，应用层网关与 KV 存储层因此在 2026 年拥挤起来。给基础设施团队的建议：把 Agent 平台接上推理集群之前，先决定会话状态放在三层中的哪一层。这个决定做错，后面的调度与容量优化都在为它还债。

---

## 参考资料

1. **负载画像与系统研究**
   - Chaokun Chang et al., [From LLM Inference to Agentic Workloads](https://arxiv.org/abs/2608.15127), arXiv 2608.15127, 2026-08——本文第一节量化数据来源（AgentSysBench：会话中位活跃计算时间 20%、re-prefill 占 cache 相关 token 55.9%、元数据占上下文最高 84.3%）
   - [Autellix: An Efficient Serving Engine for LLM Agents as General Programs](https://arxiv.org/abs/2502.13965), arXiv 2502.13965, 2025（NSDI'26 发表）——程序级调度，同等延迟下吞吐 4–15×
   - [Helium: Efficient LLM Serving for Agentic Workflows — A Data Systems Perspective](https://dl.acm.org/doi/10.1145/3802046), ACM——workflow-aware serving
   - [Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost), USENIX ATC'24——多轮会话 KV 分层复用的先导工作（原 AttentionStore）

2. **生产系统与路由**
   - NVIDIA, [Full-Stack Optimizations for Agentic Inference with NVIDIA Dynamo](https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/)——KV block 全局索引（Flash Indexer）与 agentic hints
   - llm-d, [Agentic Serving Well-Lit Path](https://llm-d.ai/docs/0.8/well-lit-paths/workloads/agentic-serving) 与 [KV Cache Management](https://llm-d.ai/docs/architecture/advanced/kv-management)；session-affinity 路由进展见 [issue #177](https://github.com/llm-d/llm-d-inference-payload-processor/issues/177)
   - Red Hat, [Master KV cache aware routing with llm-d for efficient AI inference](https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference), 2025-10
   - [DualMap: Enabling Both Cache Affinity and Load Balancing](https://arxiv.org/html/2602.06502v1), arXiv 2602.06502——4.1 节「粘性 vs 均衡」的学术跟进

3. **Benchmark**
   - MLCommons, [Agentic Inference for MLPerf Inference](https://mlcommons.org/2026/07/agentic-inference-for-mlperf-inference/), 2026-07——多轮、上下文持续增长、闭环工作流的基准标准化

延伸阅读：LLM 系统演进「请求 → 会话 → 轨迹」的三段叙事，见 [LLMSys PaperList](https://github.com/AmberLJC/LLMSys-PaperList/)。

---

## 源文件索引

| 文件                                            | 关键符号                                                                                                               | 本文引用点                                             |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `python/sglang/srt/mem_cache/radix_cache.py`    | `match_prefix`、`inc_lock_ref`/`dec_lock_ref`、`evict`、`last_access_time`                                             | 前缀匹配与引用计数保护（:376、:622、:637、:592、:248） |
| `python/sglang/srt/managers/schedule_batch.py`  | `retract_decode`、`_get_decode_retraction_order`                                                                       | 显存不足时的收回顺序与最后请求中止（:2816、:2869）     |
| `python/sglang/srt/managers/schedule_policy.py` | `PrefillAdder`、`rem_chunk_tokens`、`budget_state`                                                                     | 准入预算（:504、:512-538、:664、:834）                 |
| `python/sglang/srt/managers/scheduler.py`       | `get_next_batch_to_run`、`event_loop_normal/overlap`                                                                   | 调度主循环（:3015、:1719、:1754）                      |
| `python/sglang/srt/server_args.py`              | `schedule_policy`（默认 fcfs）、`enable_priority_scheduling`、`schedule_conservativeness`、`enable_hierarchical_cache` | 策略选项与默认值（:828-846、:886、:2668）              |
| `vllm/v1/core/sched/scheduler.py`               | `schedule`、抢占受害者选择                                                                                             | 按 (priority, arrival_time) 抢占（:440、:592-594）     |
| `vllm/v1/core/kv_cache_manager.py`              | `get_computed_blocks`、`find_longest_cache_hit`、`free`                                                                | 前缀命中与块归还（:229、:261、:567）                   |
| `vllm/config/cache.py`                          | `enable_prefix_caching`                                                                                                | V1 默认开启（:93）                                     |

_版本：vLLM `43d691ec6b`（2026-08-07）、SGLang `f7101b0ae6`（2026-08-18）。文中未标注实测的耗时/成本数字均为量级示意。_
