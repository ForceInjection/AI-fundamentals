# RoPE 与 Prefix Caching：位置编码如何决定 KV 能否复用

想象你运营着一个 AI 客服系统。每通对话都以同样的 System Prompt 开头——"你是专业客服，回答要礼貌准确"。按照 Prefix Caching 的逻辑，这段 System Prompt 的 KV Cache 只需算一次，后续所有请求直接复用——省时省算力。但工程实践中很快发现：**复用经常失败**。

原因藏在位置编码里。RoPE（Rotary Position Embedding）将每个 token 的绝对位置"旋转"进了 K 向量。同一段 System Prompt，在第一通对话里占据位置 0-99，在另一通追加了历史记录的对话里可能占据位置 50-149——位置变了，K 向量的数值就变了，hash 对不上，缓存复用失败。

这就是 Prefix Caching 从理论到实践之间最容易被跳过的一步：**位置编码打破了"相同内容 = 相同表示"的假设**。

> **前置阅读**：[Prefix Caching 原理分析](prefix_caching.md) — 本文假设你已了解 block 级 KV 复用的基本机制。

---

## 一、位置为什么会变——从 AI Agent 的视角看

### 1.1 一个典型的多轮对话

AI Agent 的一次任务执行通常需要多个推理轮次。每轮推理时，模型需要看到完整的上下文——之前的对话、工具调用结果、中间思考过程——全部拼接在一起作为新的 prompt。

以客服 Agent 为例。第一轮，用户问"帮我查订单 #12345"，此时 prompt 只有 System Prompt（约 100 token）加上这短短一句用户问题。Agent 调用订单查询工具，拿到"已发货"的结果，组织成回复——这进入了第二轮推理：System Prompt、用户原始问题、工具调用记录、工具返回结果，全部拼接在一起，总计约 400 token。用户追问"那预计什么时候到？"——第三轮推理，prompt 又变长了，但只是往尾部追加了新内容。

**关键观察**：在这三轮推理中，System Prompt 始终在最开头，占据位置 0-99。每轮新增的对话和工具结果只是往尾部追加，前缀纹丝不动。KV Cache 的前缀 block 可以直接复用——这正是 vLLM 的 Automatic Prefix Caching (APC) 最擅长的场景。

### 1.2 位置什么时候会偏移

但 Agent 场景并非总是这么规整。考虑以下情况：

**情况一：对话太长，触发截断：**

当对话轮次累积到几千 token 时，模型上下文窗口有限（比如 4096 token），最早的内容会被截断。System Prompt 可能被部分或全部截掉，后续内容的绝对位置整体前移——原来在位置 2000 的 token，截断后可能变成了位置 500。所有缓存失效。

**情况二：不同用户有着不同的前缀：**

用户 A 发起了一次标准对话——System Prompt 在最前面，后面接用户问题。用户 B 则先上传了一份 500 token 的销售报告要求分析，System Prompt 被放在了报告之后。同一段 System Prompt，在用户 A 的请求里占据位置 0-99，在用户 B 的请求里被推到了位置 501-600。RoPE 编码的位置角完全不同——缓存命中失败。

**情况三：多 Agent 协作：**

在 multi-agent 系统中，不同 Agent 的 System Prompt 和上下文各不相同。一个专门做代码审查的 Agent 和另一个做测试生成的 Agent 可能共享同一个"基础指令块"，但这个指令块在不同 Agent 的完整 prompt 中处于不同位置。

### 1.3 问题的本质：content-identical ≠ cache-identical

以上所有情况的共同根源是：**两个请求的 token 序列内容相同，但起始的绝对位置不同**。位置差异导致 RoPE 编码不同，RoPE 编码不同导致 K 向量不同，K 向量不同导致 block hash 不同——最终缓存查找失败。

这意味着 **content-identical ≠ cache-identical**。位置信息已经"烧进"了 K 的数值表示里。在 Agent 场景中，由于上下文结构的高度动态性，这个问题尤为突出。

---

## 二、RoPE 做了什么——从旋转角度到缓存障碍

上一节讲了"位置偏移导致缓存失败"的现象。这一节解释 RoPE **为什么**会让位置影响 K 的数值。

### 2.1 注意力需要位置信息

自注意力机制本身是**置换不变的**——"猫追狗"和"狗追猫"的 token 集合相同，不加位置信息就无法区分。位置编码的任务是在计算注意力之前注入"第几个 token"的信息。

### 2.2 RoPE 的数学直觉

RoPE 的做法是：对 Q 和 K 向量的每对相邻维度 $(2i, 2i+1)$，施加一个旋转矩阵，旋转角与 token 位置 $m$ 成正比：

$$
\begin{pmatrix} q_{m,2i} \\ q_{m,2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{m,2i} \\ x_{m,2i+1} \end{pmatrix}
$$

其中 $\theta_i = 10000^{-2i/d}$。直观理解：每个维度对以不同的频率旋转——低频维度转得快（精细区分相邻位置），高频维度转得慢（粗粒度区分远近）。

这种设计有一个优雅的性质：两个位置的 Q 和 K 做内积时，旋转角相减，结果只依赖位置差 $(m-n)$：

$$
(Q_m R_m) \cdot (K_n R_n)^T = Q_m R_{m-n} K_n^T
$$

**相对位置信息被自动嵌入到注意力分数中**，无需额外计算。这是 RoPE 被几乎所有现代 LLM 采用的核心理由。

但这同时意味着**绝对位置 $m$ 也被不可逆地编码进了 K 向量**——这正是 §一 中 AI Agent 场景里缓存失败的数学根源。

---

## 三、三种绕过方案

既然绝对位置阻碍了 block 级复用，工程上有三条路径。共同目标：**缓存查找对位置不敏感，但注意力计算对位置保持正确**。

### 3.1 方案 A：限制匹配范围——最简单，也最受限

**思路**：只允许在相同绝对位置前缀上匹配。从位置 0 开始的 System Prompt block，只和另一个也从位置 0 开始的 System Prompt block 共享。

**实现**：vLLM 的 Automatic Prefix Caching (APC) 通过 **hash chaining** 隐式编码位置——每个 block 的 hash 不仅依赖当前 token 内容，还依赖前一个 block 的 hash（`hash_block_tokens(prev_block_hash_value, curr_tokens, ...)`[^1]）。链的长度本身对应绝对位置——链上第 0 个 block 和第 50 个 block 的 hash 永远不同，即使 token 内容相同。

**优点**：实现极简，无额外计算开销。

**缺点**：回到 AI Agent 场景——如果用户 A 的请求以 System Prompt 开头（命中），用户 B 的请求前面有 500 token 的文件内容（System Prompt 在位置 501，不命中），用户 B 完全无法复用用户 A 已经算好的缓存。

### 3.2 方案 B：剥离 RoPE，重算位置——命中率换计算

**思路**：缓存时不存完整的 K，只存 K 的非位置部分（nope，即施加 RoPE **之前**的内容部分）。复用时，用目标位置重新施加 RoPE。

**工作流程**：

1. **缓存时**：提取 K 的 nope 部分，基于纯内容做 hash（位置无关）
2. **命中时**：从缓存读取 nope + 目标位置 $m$ → 施加 RoPE → 得到目标位置的完整 K
3. **V 不参与**：V 不受 RoPE 影响，直接复用

**额外开销**：每个命中的 block 需要一次额外的 RoPE 旋转。对大模型来说，这个开销远小于完整的 Prefill 重算。

**命中率**：理论上限接近 100%——只要内容相同就能命中，与绝对位置无关。

MLA（DeepSeek-V2/V3）架构天然解耦了内容和位置，**为方案 B 提供了理想的基础**：KV latent 不包含位置信息，可直接对其做内容哈希实现位置无关的缓存查找；命中后配合目标位置重新施加 RoPE key，即可完成跨位置复用。不过，这只是架构层面的便利——要真正实现跨位置复用，仍需推理框架在缓存查找策略、RoPE 重算路径等环节做专门设计。

### 3.3 方案 C：Rotary Correction——不重算，只旋转差值

**思路**：缓存完整的 K（含 RoPE），复用前通过数学变换"校正"到目标位置。利用 RoPE 旋转矩阵的正交性和可加性（$R_m \cdot R_n = R_{m+n}$）：

$$
K_{m'} = K_m \cdot R_{m'-m}
$$

从位置 $m$ 的缓存 K 变换到位置 $m'$，只需施加一个 $(m'-m)$ 的旋转差矩阵——单次矩阵乘法。

**相比方案 B 的优劣**：

- **优势**：不需要缓存两份数据（nope + rope），不需要重新计算完整 RoPE forward，计算量更低
- **劣势**：实现复杂——attention kernel 需要支持"复用 + 校正"混合路径；低精度（FP8/INT8）下旋转校正可能引入额外量化误差

LMCache 的 `CacheBlend` 机制（EuroSys 2025 最佳论文，已集成入 [vLLM Production Stack](https://github.com/vllm-project/production-stack)）采用了这一思路，在 block 命中后对 K 做位置旋转校正，并支持 vLLM、SGLang、NVIDIA Dynamo 等多引擎间的跨引擎 KV 复用。[^2]

> **验证**：旋转可加性 $R_{m'} = R_{m'-m} \cdot R_m$ 在数学上精确成立（见 [`rope_verify.py`](rope_verify.py) 的矩阵验证），但直接用 PyTorch 叠加两次 `apply_rotary_emb` 会在高维上产生浮点累积噪声。工业实现通常直接用 CUDA kernel 计算 $R_{m'-m}$，避免两次旋转。

### 3.4 方案对比

| 方案                     | 命中率 |    额外计算    | 实现复杂度 | 代表实现                |
| ------------------------ | :----: | :------------: | :--------: | ----------------------- |
| A: 位置限制匹配          |   低   |       无       |     低     | vLLM APC, SGLang (默认) |
| B: nope 哈希 + 重算 RoPE |   高   | 中 (重做 RoPE) |     中     | MLA 架构天然支持        |
| **C: Rotary Correction** |   高   | 低 (单次旋转)  |     高     | LMCache CacheBlend      |

方案 A 和 C 在实际系统中往往**共存**：位置对齐时走 A 的快速路径直接命中，位置不对齐时走 C 做校正后复用——兼顾命中率和效率。

> _注：SGLang 默认的 RadixAttention 通过 token 序列前缀做 Radix Tree 匹配，要求序列前缀对齐，属于方案 A。跨引擎的跨位置复用可通过集成 LMCache 的 CacheBlend 来获得方案 C 的能力。_

---

## 四、从现象到方案

Prefix Caching 的核心挑战不是"怎么存储"——block table 和 hash 匹配是成熟的工程方案。真正的难点是**什么时候可以复用**：两个看似相同的 KV block，可能因为位置编码不同而并不等价。

AI Agent 的复杂性放大了这个问题——动态前缀、多轮拼接、文件预处理，让同一段文本频繁出现在不同的绝对位置。RoPE 的相对位置内嵌特性是 attention 质量的基石，但绝对位置编码的副作用是缓存复用的障碍。

三种绕过方案代表了三种权衡：

> 方案 A — 放弃跨位置复用，换取实现简单
> 方案 B — 剥离后重算，用计算换命中率
> 方案 C — 数学校正，在最低开销下实现跨位置复用

理解这个限制，才能真正理解 Prefix Caching 的命中率上限从何而来——以及 vLLM、SGLang、LMCache 在 Agent 场景下做出不同工程选择的原因。

---

## 相关阅读

- [Prefix Caching 原理分析](prefix_caching.md) — block 级 KV 复用的基本机制
- [RadixAttention 原理](radix_attention.md) — SGLang 的跨位置 block 复用方案
- [LMCache CacheBlend](../../02_systems/lmcache/cache_blend.md) — RoPE 校正的工业实现
- [不同注意力类型的 KV Cache 到底长什么样](../basic/attention_kv_cache_formats.md) — MLA 为什么天然支持 nope 哈希
- [RoPE 原始论文: RoFormer — Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [vLLM APC 实现](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py) — `hash_block_tokens` 的 hash chaining 源码
- [验证脚本](rope_verify.py) — 三个断言的 Python 可复现验证

[^1]: vLLM `kv_cache_utils.py` 中 `hash_block_tokens(prev_block_hash_value, content_token_ids, extra_keys)` → 返回 `hash((parent_hash, block_tokens, extra_keys))`。每个 block 的 hash 依赖父 block 的 hash，链长隐式编码绝对位置。

[^2]: SGLang `RadixCache.match_prefix` 在 [`radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) 中纯粹基于 token ID 序列做 Radix Tree 匹配，不包含 RoPE 位置校正逻辑。SGLang 默认和 vLLM APC 一样要求前缀从序列起始对齐。
