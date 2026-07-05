# Attention Sinks 与 KV Cache 淘汰策略：滑动窗口为什么不够？

KV Cache 随序列长度线性膨胀——百万级上下文下，单请求的 KV 就能吃掉几十 GB 显存。PagedAttention 通过按需分配 block 解决了"怎么高效存"，但它不回答"存不下了怎么办"。当 GPU 显存耗尽，推理系统必须选择：**谁的 KV 块被牺牲？**

CPU Cache 淘汰里选错一个 cache line，多等几十个周期而已。KV Cache 淘汰里丢掉一段关键上下文，模型可能给出完全错误的答案。而最直觉的淘汰策略——滑动窗口，只保留最近 N 个 token——在实践中会导致生成质量断崖式下降。

这篇文章从 Attention Sinks 这一基础发现出发，推导为什么"按位置淘汰"行不通，以及业界如何从盲目淘汰演进到用注意力结构做 guide 的精确淘汰。

> **前置阅读**：[KV Cache 原理简介](../basic/kv_cache_原理简介.md) — KV Cache 的工作机制、显存公式；[PagedAttention 原理介绍](../basic/paged_attention.md) — block 级内存管理，本文假设你已了解 block table 和按需分配的基本概念。

---

## 一、为什么淘汰不可避免

### 1.1 KV Cache 的线性膨胀

每生成一个新 token，每层 transformer 就要追加一组 K 和 V。以 LLaMA-2 70B（GQA, 8 KV heads, head_dim=128, 80 layers, FP16）为例：

$$\text{单 token 单层 KV} = 2 \times 8 \times 128 \times 2 \text{ bytes} = 4 \text{ KB}$$

$$\text{128K context, batch=8} = 4 \text{ KB} \times 80 \times 131072 \times 8 \approx \mathbf{320 \text{ GB}}$$

模型权重本身约 140 GB（FP16），KV Cache 已是其 2 倍以上。此为所有请求均达到 128K 序列长度时的理论上限（$4\text{ KB} \times 80\text{ layers} \times 128\text{K tokens} \times 8\text{ requests}$）。实际峰值受 prefix sharing 命中率、请求序列长度参差不齐等因素影响，通常低于此值。

### 1.2 两个缓解方向，各有天花板

面对这个压力，业界沿着两条路径推进：

- **压缩每个 token 的 KV**：GQA/MQA 减少 KV head 数、MLA 低秩压缩、量化（FP8/INT4）。但压缩率有上限——MLA 已将每 token 从 32 KB 压至 ~1.1 KB，继续压缩到 0.1 KB 以下将触及精度红线。
- **减少存储的 token 数**：即淘汰——丢弃部分历史 token 的 KV。只要淘汰策略足够精确，理论上可以压缩到任意小。

**两条路径互补，但淘汰是唯一可以"不受维度限制"持续缩容的手段。** 压缩决定每个 token 存多少，淘汰决定存多少个 token。最终 KV Cache 大小 = 压缩后的单 token 大小 × 保留的 token 数量。

---

## 二、滑动窗口的诱惑：最直觉的答案

### 2.1 "越近越重要"的朴素假设

淘汰策略设计的第一个自然想法是：自回归生成是逐步推进的，最新 token 理应对下一步预测最重要。因此最简单的策略就是**滑动窗口**——只保留最近 W 个 token 的 KV，其余全部丢弃。

这个策略有令人难以拒绝的优点：

- **实现极简**：只需一个环形缓冲区，追加新 token 时覆写最旧的
- **显存占用恒定**：不随序列长度增长，W=4096 时就是固定的 4096 个 token 的 KV
- **工程友好**：无需额外数据结构、无需追踪注意力分数、无需调度器配合

几乎所有工程师在第一次面对 KV Cache OOM 时，第一个想到的方案就是滑动窗口。

### 2.2 实验给出了残酷的否定

2023 年，MIT、Meta、CMU 和 NVIDIA 的联合团队在 StreamingLLM 论文[^1]中进行了系统的消融实验。结论是明确的：

**仅保留最近 W 个 token（无初始 token）时，PPL 在超过预训练长度后呈断崖式上升——不是缓慢退化，而是瞬间崩塌。** 模型输出变得完全不可用。

但实验中有一个更关键的发现：**只要额外保留开头 4 个 token，再加上最近 W 个 token，PPL 就能保持稳定**——即使序列长度远超预训练上限（如用预训练 4K 的 Llama-2 跑到 4M token）。这 4 个开头的 token 甚至不需要有语义——替换成换行符 `\n`，效果一样。

这个现象反直觉：为什么离当前生成位置最远的 4 个 token，比中间的几千个 token 都关键？

---

## 三、Attention Sinks：发现注意力分布的隐藏结构

### 3.1 SoftMax 的"强制消费"

答案藏在注意力的数学结构里。对于序列中位置 $m$ 的 query，它对所有位置 $n \leq m$ 的 key 计算注意力权重：

$$\alpha_{m,n} = \frac{\exp(q_m \cdot k_n / \sqrt{d})}{\sum_{j=0}^{m} \exp(q_m \cdot k_j / \sqrt{d})}$$

分母是位置 0 到 $m$ 的指数和，分子是位置 $n$ 的指数。关键约束：$\sum_{n=0}^{m} \alpha_{m,n} = 1$——SoftMax 强制所有权重之和等于 1。

当 query 与大多数前面的 key 都没有强语义关联时，**SoftMax 不允许注意力"不消费"**。多出来的注意力质量必须有个去处。由于 SoftMax 的指数放大效应——即使所有内积值都很小，归一化后仍然会分配出 100% 的权重——多余的注意力被倾倒给了某个固定目标。

### 3.2 为什么是前几个 token？

因果注意力掩码（causal mask）揭示了答案：位置 0 的 token 对位置 1、2、3...的所有 query 都可见。位置 $m$ 的 token 只对位置 $\geq m$ 的 query 可见。前几个 token 拥有最大的"受众面"——序列中每个后续位置都能 attend 到它们。

这使得前几个 token 成为**天然的注意力接收器（Attention Sink）**：每个位置的 query 都可能把多余的注意力分配给它们，因为它们在所有位置都"可见"。随着序列增长，这种累积效应越来越显著——位置 0 收获了序列中所有后续位置的注意力残差。

论文的实验完美验证了这一点：把开头 token 替换成语义无关的 `\n`，注意力分布几乎不变——说明这个现象与语义无关，纯粹来自数学结构。（StreamingLLM 论文的 Figure 1-3[^1] 展示了注意力权重按序列位置的分布，前几个 token 的权重显著高于中间数千个 token，建议对照阅读以获得直观感受。）

### 3.3 去掉 Sink 后发生了什么？

去掉前几个 Attention Sink token 后，模型并不会"均匀地"把注意力重新分配给剩余 token。由于 SoftMax 的指数特性——即使内积的微小波动，经过指数放大后也会导致注意力高度集中到少数几个 token 上——某个恰好与 query 有微小正内积的 token 会成为新的"伪 Sink"。但这个新 Sink 在序列中的位置不稳定，其语义也并不适合承担注意力接收器的角色，导致信息流混乱，逐层放大后最终输出崩塌。

**滑动窗口失败的根源在此：它不是因为"丢失了重要语义信息"而失败，而是因为"破坏了注意力分布的数值稳定性"而失败。** Attention Sink 像一个锚——去掉它，整个注意力机制就开始漂移。

> **与位置编码无关**：Attention Sinks 是 SoftMax + Causal Mask 数学结构的副产品，与使用 RoPE、ALiBi 还是可学习绝对位置编码无关。位置编码只影响注意力分布的具体形状，不影响 Attention Sink 现象的存在——这是自注意力机制本身的结构属性，不是某种位置编码的副作用。

---

## 四、从发现到策略：Informed Eviction 的三条路线

Attention Sinks 的发现回答了"为什么滑动窗口不行"，但更重要的是，它揭示了一条设计原则：**哪些 token 可以淘汰，不应该由"远近"决定，而应该由"注意力分布的结构"来决定。**

基于这一原则，业界发展出三类"知情淘汰"（Informed Eviction）策略。它们共享同一个底线——Attention Sink 不可丢弃——但通过不同的方式识别和保留其他重要 token。

### 4.1 StreamingLLM：最简可行方案——Sink + Window

既然前几个 token 的注意力吸收功能不依赖语义，那最简单的方案就是：**永久锁定开头的 4 个 token 作为 Attention Sink，同时保留最近 W 个 token 的 Rolling Window。中间的 token 全部丢弃。**[^1]

```text
保留结构：
┌──────────────┬──────────────────────────────┬──────────────────────────────┐
│  Sink tokens │          已淘汰区域            │       Recent window          │
│   (位置 0-3) │      (位置 4 ~ N-W-1)         │     (最后 W 个 token)         │
│   永久锁定    │         直接丢弃               │         持续滚动              │
└──────────────┴──────────────────────────────┴──────────────────────────────┘
```

这是一个**静态策略**——它不需要观察注意力分布，只依赖位置信息就能做决策。核心数据结构只是一个环形缓冲区 + 前 4 个位置的不可驱逐标记。论文进一步发现：如果预训练阶段引入一个专用的可学习 sink token，推理时只需保留这 1 个 token 即可替代 4 个普通 token 的 sink 功能，进一步节省 KV 空间。

**适用场景**：流式对话、多轮聊天——模型需要在对话中持续生成，但不需要"回忆"很远的历史细节。

**局限**：一刀切地假定所有中间 token 都不重要。长文档问答（"合同第三段的金额是多少？"）、代码库分析（"utils/helper.py 中的那个函数..."）等任务中，中间某个 token 可能承载着关键信息——StreamingLLM 会无情地丢弃它。

### 4.2 H2O：让注意力分数做裁判

StreamingLLM 的问题在于它**根本不看内容**——不管是"请记住这个密码：xyz789"还是"嗯嗯好的我知道了"，在中间位置一律平等淘汰。H2O (Heavy Hitter Oracle)[^2] 换了一个思路：**让注意力分数来打分——被注意力机制频繁"关照"的 token，就是重要的 token。**

核心机制：

1. **累积注意力分数**：每个 decode 步骤，当前 query 对所有历史 KV 计算注意力权重。将这些权重按 key 的位置累积——位置 $i$ 的累积重要性 = 所有 decode 步骤中对位置 $i$ 的注意力分数之和。
2. **识别 Heavy Hitters**：累积分数最高的 top-k 个 token 被标记为 Heavy Hitter，不可淘汰。
3. **保留策略**：Sink（前几个 token）+ Heavy Hitters（累积分数 top-k）+ Recent Window（最近 token）。

```python
# H2O 的核心逻辑（伪代码）
# budget: 总保留 token 数（sink + heavy_hitters + recent_window）
for step in decode_steps:
    attn_scores = compute_attention(query, all_keys)  # 当前 query 对所有 key 的注意力权重
    for i in range(seq_len):
        cumulative_score[i] += attn_scores[i]  # 逐 token 累积注意力分数

    # 选出 heavy hitters: 累积分数最高的 (budget - sink_count - window_count) 个
    heavy_hitters = topk(cumulative_score[sink_count : -window_count], k=budget - sink_count - window_count)

    # 保留: sink + heavy_hitters + recent_window
    evict_mask = ones(seq_len)
    evict_mask[sink_indices] = False              # 永远保留
    evict_mask[heavy_hitter_indices] = False      # 注意力高分
    evict_mask[recent_window_indices] = False     # 最近窗口
    evict_tokens(evict_mask)
```

**理论保证**：论文将 KV Cache 淘汰形式化为次模函数最大化问题，并证明了在静态快照下，贪心保留累积注意力最高的 token 具有 $(1-1/e)$ 的近似最优性保证[^2]。需要注意实际推理中 token 集动态增长且评价函数随时间变化，理论保证转移到动态场景需要更强的条件——但实验表明贪心策略在实践中仍然有效。

**收益**：在 OPT-6.7B 和 OPT-30B 上，仅保留 20% token 即可维持与原模型相当的精度，吞吐最高提升 29×（vs DeepSpeed Zero-Inference），延迟降低 1.9×。

**代价**：需要为每个 token 维护累积注意力分数——对于百万级 token 序列，分数本身的存储和更新开销需要额外设计（量化、稀疏化或滑动平均）。

### 4.3 SnapKV：在 Prefill 阶段"预判"一生

H2O 的累积注意力有一个结构性缺陷：它在 decode 过程中逐步积累，前期步骤可能因为信息不足而误淘汰重要 token（因为还没"看到"它们被大量 attend）。SnapKV[^3] 的核心洞察是：

**模型在读完整个 prompt 后，末尾几个 token 的注意力分布已经高度预示了整个生成过程中哪些位置重要。** 不需要等到 decode 逐步累积。

具体流程：

1. **Prefill 结束**：整个 prompt 的 KV 已计算完毕
2. **观察窗口投票**：取 prompt 末尾的 observation window（如最后 32 个 token）的 query 对所有前缀位置的注意力权重
3. **按 head 聚合**：每个 attention head 独立计算——将该 head 的 observation window 内所有 query 对每个前缀位置的注意力权重**求和**（sum pooling），得到每个前缀位置的"重要性投票分"
4. **聚类平滑**：对投票分做 **1D max pooling**（kernel_size=7），使高重要性 token 邻近的 token 也被保留——保留上下文完整性，避免"孤立 token"丢失局部语义
5. **选出重要位置**：每个 head 保留 top-k 个投票最高的位置 + observation window 本身
6. **生成阶段只使用保留的 KV**

```text
Prefill 阶段:                                   Decode 阶段:
┌──────────┬─────────────┬──────┐              ┌─────────────────────────────┐
│  前缀     │ observation │      │              │  仅使用保留的 KV + 逐步新增    │
│ (被投票)  │  window     │  →   │              │  (保留哪些由投票决定)          │
│          │   (投票者)   │      │              └─────────────────────────────┘
└──────────┴─────────────┴──────┘
        ↓              ↓
   sum pooling    选出 top-k
   (按 head)   →  max pooling  →  保留位置
                   (聚类平滑)
```

**收益**：16K token 输入下实现 8.2× 内存压缩，生成速度提升 3.6×，单卡最大 380K token。

> **GQA 下的 KV 共享**：SnapKV 在每个 head 内独立打分并选出 top-k 位置，但由于 GQA 下多个 Q head 共享同一组 KV head，最终保留的 KV 是所有 head 选择的**并集**——任一 head 投票为重要的位置，其完整的 KV（覆盖该组所有 head）都会被保留。这避免了各 head 各自保留不同子集导致的 KV 碎片化。

**关键局限——"休眠 token"问题**：SnapKV 和 H2O 都假设"注意力高的 token = 重要的 token"。但有些 token 在注意力分布中几乎不可见，却至关重要——比如 API key、密码、代码中的常量值。这些 token 在 generation 早期不被 attend，但在特定 query 出现时（如"请输出之前生成的 API key"）才突然变得关键。此时它们早已被淘汰，无法恢复。2025 年 Transactional Attention[^4] 论文在凭据检索类任务上实测 SnapKV 几乎完全失败（准确率接近 0%）——这类"休眠 token"在注意力分布中完全不可见，却是后续生成的关键依赖。

---

## 五、系统视角：当 Token 级判断遇上 Block 级约束

前三节讨论了"算法上哪些 token 重要"——属于注意力层的判断。但推理系统中 KV Cache 以 **block 为单位**管理（PagedAttention 的 `block_size=16` 或 256），一个 block 里可能同时包含重要 token 和不重要 token。淘汰粒度是 block，不是 token。

### 5.1 策略需要降级

StreamingLLM 或 H2O 可以将重要性精确到单个 token，但在块管理系统中，释放一个 block 意味着放弃其中全部 token。工业落地中的折中方案：

- **按 block 内重要 token 占比排序**：重要 token 占比最低的 block 先被淘汰
- **仅淘汰全"冷" block**：只有当 block 内所有 token 都低于重要性阈值时才淘汰——更保守，需要更大的 `block_size` 预算
- **双速 block 池**：将"冷 block"和"热 block"放入不同的分配池，优先从冷池中回收

vLLM 的 RFC #39076[^5] 提出了一种更激进的方案——**Entropy-Gated Online Block Expiration**：在 decode 阶段实时追踪每个 block 的注意力质量，低于阈值的 block 在 decode 过程中被动态标记为 EXPIRED，PagedAttention kernel 在 softmax 前直接将其归零，物理页立即回收。这与前面基于注意力分数的静态淘汰不同——过期判断发生在 decode 的每一步，而非 Prefill 结束时一次性决定。据提案估计可减少 30-60% 的活跃 KV 占用，但该方案仍在提案阶段，尚未在生产环境中大规模验证。

### 5.2 vLLM Preemption：当逐块淘汰不够用时

当显存压力达到极限、无法靠逐块释放缓解时，vLLM 调度器触发 **Preemption**——整个请求的所有 KV block 被牺牲：

| 模式                              | 行为                                                                                       | 代价                  | 适用场景                     |
| --------------------------------- | ------------------------------------------------------------------------------------------ | --------------------- | ---------------------------- |
| **Recompute**（V1 唯一模式）      | 释放所有 block，请求返回等待队列，需重新 Prefill（共享前缀的 token 可能命中 prefix cache） | 丢弃未缓存的已计算 KV | 所有请求                     |
| **Swap**（V0，需 CPU Offloading） | Block 移至 CPU 内存，请求进入 swapped 队列，恢复时回传                                     | 需要 CPU↔GPU 带宽     | V0 多序列请求（Beam Search） |

V1 只有 Recompute 一种模式——SWAP 在 V1 中被有意移除（`num_cpu_blocks = 0`），设计假设是 Prefix Caching 可以让恢复时的大部分 Prefill token 命中缓存，从而降低重算成本[^6]。但需要注意：**并非总能完全命中**——如果 Preemption 发生在 decode 中段且请求没有与其他请求共享前缀，恢复时仍需从头完整 Prefill。被 Preempt 的请求通过 `_preempt_request()` 插入 `self.waiting` 队列最前面，保证不会长期饥饿。Preemption 是"核选项"：整个请求的所有 KV block 被释放，`num_computed_tokens` 归零，缓存完全命中前仍需完整走一遍 block 分配路径。

**Preemption 触发条件**：由调度器 `schedule()` 中 `allocate_slots()` 返回 `None`（KV Cache 耗尽）触发——不是基于水位线的主动预防，而是分配失败的被动响应。KVCacheManager 内部有一个 watermark 参数用于为等待/被抢占请求预留 block，但它不驱动 Preemption 决策本身，只影响新请求的接纳时机[^6]。

### 5.3 两层逻辑，一条链路

```text
算法层（Token/Block 重要性）
  StreamingLLM / H2O / SnapKV
  → 精确识别"哪些值得留"
  → 减少内存压力，降低 Preemption 触发频率
        ↓
系统层（请求级 Preemption）
  vLLM Scheduler + allocate_slots 失败时触发
  → 当算法层压缩仍不够时，做"最后一道防线"
  → 牺牲整个请求，通过 Prefix Caching + 重算恢复
```

两层互补：算法层通过精细化淘汰减少 Preemption 的发生；系统层 Preemption 保证在最极端的内存压力下系统不会 OOM 崩溃。

---

## 六、取舍：没有免费的晚餐

### 6.1 六种策略的三轴对比

| 策略                             | 决策依据           | 质量保持  |  额外开销  | 复杂度 |      典型保留比例       | 典型场景           |
| -------------------------------- | ------------------ | :-------: | :--------: | :----: | :---------------------: | ------------------ |
| **Sliding Window**（仅最近窗口） | 位置（最近）       |  ❌ 崩塌  |     零     |  极低  |          W / N          | —（不可用）        |
| **StreamingLLM**（Sink+Window）  | 位置（开头+最近）  |  ✅ 稳定  |     零     |   低   |        (4+W) / N        | 流式对话           |
| **H2O**（累积注意力）            | 注意力分数（累积） |  ✅ 优秀  |     中     |   中   |       ~20% token        | 需 recall 中间信息 |
| **SnapKV**（观察窗口投票）       | 注意力分数（快照） |  ✅ 优秀  |     低     |   中   | ~12% token（8.2× 压缩） | 长 prompt 处理     |
| **Entropy-Gated Exp.**           | 注意力质量（实时） |  ✅ 优秀  |     中     |   高   |  40-70%（据提案估计）   | 超长 decode        |
| **Preemption**（系统层）         | 调度策略           | ✅ 可恢复 | 高（重算） |   低   |    0%（全请求释放）     | 内存压力极限       |

Sliding Window 不加入生产比较——它是其他所有策略的"反面教材"，也是理解为什么需要 Informed Eviction 的起点。

> 注："额外开销"指相对于纯位置策略（Sliding Window）新增的注意力分数追踪、存储或计算成本。"复杂度"指在 PagedAttention block 管理体系中的集成难度。

### 6.2 三个开放问题

**注意力分数的存储成本**。H2O 要求为每个 KV token 维护累积注意力分数——对于 1M token 的单请求，朴素存储 FP32 累计分数需约 4 MB，单个请求看似不大，但 batch=32 时即达 ~128 MB，且需每步更新，访存开销不可忽略。工业落地通常对分数做 FP16 或 INT8 量化，或用滑动平均替代精确累积。若分数存储瓶颈未能解决，H2O 在超长序列场景下的有效保留比例将被迫降低，退化为近似 StreamingLLM——仅保留 sink + 少量疑似 heavy hitter + window。

**"休眠 token"的检测与保护**。凭据类、数值类 token 在注意力分布中几乎不可见，却在特定 query 下突然变得至关重要。完全依赖注意力分数的淘汰策略（H2O、SnapKV）存在系统性的漏判风险——这是注意力引导策略的阿喀琉斯之踵。需要注意，回退到 StreamingLLM 并不能解决这个问题（它同样会丢弃中间 token）。可行的方向包括引入语义层保护——如 Transactional Attention[^4] 的锚定词赞助机制——或在已知存在关键信息的区间强制保留对应 block（通过用户标注或 prompt 结构推断）。

**训练时预防 vs 推理时修补**。StreamingLLM 的 sink token 预训练代表了另一种范式：在训练阶段解决问题（"你只有一个 sink token"），而非在推理阶段修补问题（"我们帮你保留前 4 个"）。这引出了一个更深层的问题：是否可以训练模型将注意力更多地分配给语义相关的 token，而非固定地倾泻给前几个位置（减小 Attention Sink 效应）？如果重要 token 能因此获得更高的相对注意力分数，H2O 和 SnapKV 的淘汰精度将大幅提升——它们正是依赖分数差异来区分关键与非关键 token。反之，若注意力分布变得均匀而缺乏区分度，top-k 选择反而不稳定。目前这是一个 open research direction，也是连接训练与推理优化的一个交叉点。

### 6.3 一句话总结

**从 Sliding Window 到 Attention Sinks 到 Heavy Hitters，KV Cache 淘汰的本质是从"盲目按位置淘汰"走向"用注意力结构做 guide 的精确淘汰"——代价是越来越多的 bookkeeping。选择哪个策略，取决于你的任务对"丢失中间关键信息"有多敏感，以及你愿意为此付出多少额外显存和计算。**

如果你的场景是多轮对话——StreamingLLM 的零开销方案足够。如果需要精确 recall（RAG、长文档分析）——H2O 或 SnapKV 的注意力引导淘汰值得投入。如果你在构建推理平台——系统层 Preemption + 算法层压缩两者都需要。没有哪种方案是"最好的"，只有最匹配你的任务对"精确性 vs 成本"权衡点的那个。

---

## 相关阅读

- [KV Cache 原理简介](../basic/kv_cache_原理简介.md) — KV Cache 的工作机制和显存公式
- [PagedAttention 原理介绍](../basic/paged_attention.md) — Block 级内存管理：block table、按需分配、碎片率
- [KV Cache 压缩技术详解](../compression/kv_cache_compression.md) — 四维冗余模型 + 压缩全景
- [RoPE 与 Prefix Caching](../prefix_caching/rope_and_prefix_caching.md) — 位置编码如何影响缓存复用
- [vLLM KV Offloading Connector 与 LMCacheConnector 深度对比](../advanced/kv_offloading_analysis.md) — Preemption 的 offloading 视角

[^1]: Xiao et al., "Efficient Streaming Language Models with Attention Sinks," ICLR 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) — 提出 Attention Sinks 现象和 StreamingLLM 框架。实验跨度：Llama-2、MPT、Falcon、Pythia，最高测试 4M+ token。

[^2]: Zhang et al., "H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models," NeurIPS 2023. [arXiv:2306.14048](https://arxiv.org/abs/2306.14048) — 将 KV Cache 淘汰形式化为次模函数最大化问题，证明在静态快照下贪心保留累积注意力最高的 token 具有 $(1-1/e)$ 的近似最优性。在 OPT-6.7B/30B 上，保留 20% token 即可维持精度，吞吐最高提升 29×。

[^3]: Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation," NeurIPS 2024. [arXiv:2404.14469](https://arxiv.org/abs/2404.14469) — 观察窗口投票机制，16K 输入实现 8.2× 压缩，单卡支持 380K token。已知局限：对凭据类"休眠 token"系统性漏判。

[^4]: Basu, "Transactional Attention: Semantic Sponsorship for KV-Cache Retention," Apr 2025. [arXiv:2604.11288](https://arxiv.org/abs/2604.11288) — 指出注意力不可见的"休眠 token"问题，提出通过锚定词的结构赞助保护机制。

[^5]: vLLM RFC #39076, "Entropy-Gated Online KV Block Expiration During Active Decode." — 在线 decode 阶段基于注意力质量的 block 动态淘汰方案，预期减少 30-60% 活跃 KV 占用。

[^6]: vLLM V1 调度器源码：[`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) — `_preempt_request()` 将请求插入 `self.waiting` 最前面，`num_computed_tokens` 归零后走完整 prefix cache 查找路径。关于 V1 移除 SWAP 的设计决策，参见 [Discussion #11082](https://github.com/vllm-project/vllm/discussions/11082)。
