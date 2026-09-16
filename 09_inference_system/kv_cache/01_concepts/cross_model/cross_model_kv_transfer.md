# 把 14B 的 KV 交给 32B：跨模型复用的可能与代价

本文依据 NVIDIA 论文 _Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse_（arXiv:2608.03893v1，2026-08-04），以及 Realtime AI Lab 的中文解读（2026-09-09）。文中所有数字与论文 v1 正文、表格逐条核对，出处标在文末。

---

## 一、为什么会有这个问题

### 1.1 一条 32K 上下文的 prefill，要 7 秒

论文的延迟实验里有一个数字：在单台 8×H100 上，把 32,768 token 送进 Qwen3-32B 做一次 prefill，耗时 **6,975 ms**。接近七秒。

这七秒里，模型把每个 token 的隐状态逐层投影成 Key 和 Value，存下来备用。这些 K/V 就是 KV Cache。之后每生成一个 token，模型读这份缓存做 attention，不再重算历史。缓存存在的意义，就是让这段计算只发生一次。

问题出在"只发生一次"的前提上：**它只对同一个模型成立。**

### 1.2 prefix caching 救不了跨模型

今天的推理引擎都有 prefix caching。同一段 system prompt、同一份文档、同一段对话历史，第二次进来直接命中缓存，prefill 跳过。RadixAttention、PagedAttention、多级缓存卸载，过去几年围绕这件事做了大量工程。

但这些机制有一个共同前提：缓存的产生者和消费者是**同一个模型**。换了模型，缓存全部作废。

### 1.3 被废掉的是一整层共享缓存

一次 prefill 七秒，听起来还能忍。但这只算了单次请求的账。实际部署里更贵的是另一块，论文没有展开：L3 共享缓存。

今天的推理系统普遍用分层缓存：L1 在 GPU 显存，L2 在主机内存，**L3 落在外部存储**（NVMe、分布式存储、对象存储），越往下容量越大、单位成本越低。三层的分工里有一条关键约束：**L3 是唯一可以跨实例共享的**。SGLang 官方文档写得很直白：

> L1 和 L2 是单实例私有的；只有 L3 能共享。`Host memory cannot be pooled across instances or across hosts, not even for two instances on the same node.`

所以 L3 承载的不是某个实例的热数据，而是**整个部署共用的冷数据池**。规模是 TB 量级：本仓的 GLM-5 容量推演中，2,000 session 的场景需要约 **5.92 TB** NVMe 空间来存全量副本；SGLang 的 Mooncake 部署样例里，全局共享段配的是 **816 GB**。

**L3 里的每一个字节，都是用 GPU 算力换来的。**

要重建 L3 里的 KV，只有一条路：把对应的文本重新送进模型做一次 prefill。磁盘备份可以拷贝、重放，KV 不行。一个 TB 级的缓存层，等于把开头那七秒重复了成千上万次，而这笔算力已经花掉了，是沉没成本。

换模型的那一刻，这些缓存全部失效。没有报错，没有告警，文件还躺在存储里，只是再也命中不了。接下来要付两笔账：重建期间命中率回到零点，以及重新花一遍那笔算力。

这个损失有多大，agent 负载上的实测给了一个参照：在约 4,300 个 Claude Code 与 Codex session、35 万步推理的样本里，**fresh token 只占 append token 的 19.0%**。正常情况下约 **81% 的 prefill 由缓存承担**，缓存一旦失效，这部分全部回到 GPU。

### 1.4 三个场景

论文开头点出了两个正在放大的趋势：**长 agentic 会话**（上下文跨多轮累积，越滚越长）和**多模型编排**。两条叠在一起，prefill 成本被放大得很厉害：会话越长，prompt 越长；而每次切换模型，接收方都要对整段累积的上下文**重新 prefill 一遍**。

论文列举了三种真实场景：

- **成本-质量级联**：大部分请求交给小模型，少数难题升级给大模型
- **会话中途换模**：长会话跑到一半切换模型
- **路由**：按请求特征分发到不同模型

这三种做法的共同点是，实践里会在**同一个 model family** 内切换不同规模的成员。Qwen3 的 8B 和 32B，Llama 3.1 的 8B 和 70B，共享 tokenizer、相似架构，切换起来的工程代价最小。

而每一次切换，都意味着一遍 7 秒。

### 1.5 根因：KV 是各自训练出来的表示

为什么缓存不能直接搬？论文指出的难点在于，源模型和目标模型可能在**层数、隐藏维度和 KV head 配置**上都不同。

表示层面同理。模型 A 和模型 B 各有自己的 $W_K$、$W_V$ 和内部特征，即使 tokenizer 和输入 token 相同，两边算出来的 K/V 也处在各自的表示空间里。KV 是**模型自己算出来的一份中间表示**：从缓存里读不出原始文本，也读不出 token ID，它只对生成它的那个模型有意义。

所以问题的形状是：**能不能把 Source 模型的 KV，转换成 Target 模型期待的样子，从而省掉 Target 那次 prefill？**

---

## 二、为什么"线性映射"是对的函数形式

这一节是全文的地基。理解它，后面所有设计选择才有解释。

### 2.1 KV 不神秘，它就是 X·W

在 Self-Attention 里，每一层把 token 的隐状态投影成 Query、Key、Value：

$$
K = X \cdot W_K, \qquad V = X \cdot W_V
$$

$X$ 是进入该层 attention 的隐状态（残差流经过归一化和上游若干层的累积结果），$W_K$ 和 $W_V$ 是该层的权重矩阵。

**KV 的来源是隐状态的一个线性投影。**这一点是后面全部推导的起点。

### 2.2 推导：线性 ∘ 线性 ∘ 线性

论文对"为什么用线性"给的是经验理由：跨模型 KV 之间表现出显著的线性结构。但**为什么线性是一个自然的选择、而不是随手试的函数形式**，论文没有展开。下面这段推导是本文补的。

现在假设两个模型的隐状态之间存在近似线性关系：

$$
X_B \approx X_A \cdot T
$$

$T$ 是一个固定的变换矩阵。代入 $K_B$ 的定义：

$$
\begin{aligned}
K_B &= X_B \cdot W_K^B \\
    &\approx (X_A \cdot T) \cdot W_K^B \\
    &= X_A \cdot (T \cdot W_K^B)
\end{aligned}
$$

而 $K_A = X_A \cdot W_K^A$，所以只要 $W_K^A$ 列满秩，就有 $X_A = K_A \cdot (W_K^A)^{+}$（伪逆）。代回去：

$$
\begin{aligned}
K_B &\approx K_A \cdot (W_K^A)^{+} \cdot T \cdot W_K^B \\
    &= K_A \cdot M
\end{aligned}
$$

**$M$ 是一个固定的矩阵。**

也就是说：**线性投影 ∘ 线性映射 ∘ 线性投影 = 线性。**

论文选 closed-form 的 ridge 回归而不是训练神经网络，方向与此一致：**只要 $T$ 存在，线性就是正确的函数形式。**表示空间如果真的近似线性相关，KV 之间的关系必然是线性的，不需要用非线性模型去逼近。

### 2.3 于是问题收敛成一个可证伪的假设

整篇论文的成败，压在 $X_B \approx X_A \cdot T$ 这一条上。而这个假设：

- **不是理论保证**，两个模型各自独立训练，没有任何机制强制它们的残差流线性相关
- **可以测量**：给定源和目标，拟合一个线性回归，看能解释多少方差

论文测出来的答案是：**部分成立，且不总是成立。**

Qwen3 14B → 32B 上，用**单个源层**的 KV 去预测某个目标层的 KV，线性回归能解释目标 Key 方差的 **56%**、Value 的 **32%**。

一半多一点。后面所有工程处理的都是这个差距。

---

## 三、方法：三个组件

论文的方法由三部分组成，消融实验显示它们的贡献并不平均。

### 3.1 Cross-layer selection：贡献最大的一项

第一个问题很实际：**层数不同怎么办？**

Qwen3-14B 和 32B 层数不一样，Source Layer 10 未必对应 Target Layer 10。同一个 layer index 在两个模型里可能处于完全不同的深度。

论文的做法是：**不假设层间对应，而是为每个目标层单独挑源层。**

具体地，对每个目标层 $l$，测试多个候选源层，按预测能力（head 平均 R²）排名，取 Top-k，把它们的 KV 特征**拼接**起来作为输入：

$$
X_K^l = \left[\bar{K}_s^{l_1} \,\|\, \bar{K}_s^{l_2} \,\|\, \cdots \,\|\, \bar{K}_s^{l_k}\right]
$$

同一目标层内的所有 head 共享这组选中的源层，论文的说法是"enabling cross-head information flow"。

**k 是 per-pair 的超参数。**Qwen3 14B → 32B 的最终配置用 **k = 8**。

如果层间存在干净的对应关系，k=1 就够用了。**需要 8 个源层才能预测 1 个目标层，说明单个源层承载不了目标层需要的信息。**

效果对照（Qwen3 14B → 32B，样本内 R²）：

| 配置    | K      | V      |
| ------- | ------ | ------ |
| k = 1   | 0.5572 | 0.3249 |
| k = 8   | 0.7914 | 0.6541 |
| k = all | 0.8451 | 0.7645 |

k 从 1 提到 8，K 的 R² 从 0.56 涨到 0.79。论文的消融实验里，**cross-layer selection 是三个组件中贡献最大的一项**。

从 k=1 到 k=8 吃掉大部分增益，k=6 时已经达到 k=all 的 92.3%（K）和 87.7%（V），这个池子不需要取满。

### 3.2 RoPE factoring：不做的话会崩

Key 里含有位置信息。RoPE（旋转位置编码）把位置 $p$ 编码成一次旋转：

$$
K_{\text{rope}}(p) = R(p) \cdot K_{\text{content}}
$$

同样一段内容，放在 position 10 和 position 10,000，旋转后的 Key 完全不同。

如果直接对带 RoPE 的 Key 做拟合，映射会**同时拟合"两个模型的表示差异"和"校准数据里的位置分布"**。校准序列只有 1,024 token，而推理时可能用到 32,768 token，位置分布对不上。

论文的解法是把位置因素剥离出来：

$$
\hat{K}_t = \left(K_s \cdot R_{\Theta_s}^{-1}(t) \cdot W_K + b_K\right) \cdot R_{\Theta_t}(t)
$$

三步：**先逆旋转掉源模型的 RoPE，在 content space 里做线性映射，再套上目标模型的 RoPE。**映射权重完全在无位置空间里拟合，因此不绑死在校准时的位置分布上。$R_{\Theta}$ 是正交矩阵，求逆精确且开销可忽略。

Value 不含位置编码，直接映射。

消融实验的数据很能说明这步的分量（Qwen3 14B → 32B）：

| 配置                           | ARC-C | HellaSwag | MMLU      | GSM8K    |
| ------------------------------ | ----- | --------- | --------- | -------- |
| 完整方案                       | 61.60 | 80.70     | 78.09     | 90.98    |
| **只在推理时省略 Target RoPE** | 44.97 | 75.39     | **25.79** | **4.17** |
| 拟合和推理都不做 RoPE          | 61.09 | 80.73     | 77.70     | 90.98    |
| 不做 RoPE 且 k=1               | 27.65 | 44.81     | 26.07     | 0.38     |

第二行是关键：**只在推理阶段省掉目标 RoPE，MMLU 掉到 25.79（五选一，随机水平）、GSM8K 掉到 4.17。**而拟合和推理都不做 RoPE 反而接近完整方案，说明问题不在 RoPE 本身，在于**两侧必须一致**。

最后一行是多重退化的叠加：RoPE 和 cross-layer 都不做、且只取 k=1，GSM8K 剩 0.38。

### 3.3 Per-head ridge：闭式解，不用梯度

有了选定的源层，接下来为**每个目标层的每个 KV Head** 分别建立 K 和 V 的映射：

$$
\hat{Y} = XW + b
$$

K 和 V 各用一套参数，参数不在不同 target head 之间共享。拟合用 Ridge Regression，正则系数 **$\lambda = 0.01$**，去中心化后有闭式解：

$$
W^{*} = (X^{\top}X + \lambda I)^{-1} X^{\top}Y
$$

论文特意说明了为什么不用纯最小二乘：特征维度可达数万，而且 top-k 源层天然相关，$X^{\top}X$ 接近奇异。Tikhonov 正则项用来稳定求逆，带来的拟合偏差可以忽略。$\lambda = 1$ 时性能会崩（HellaSwag 掉 15.79 分），但 0.01 附近很稳。

**整个过程不使用反向传播或梯度下降。**论文称之为 closed-form、training-free fit。

不过"training-free"这个词需要补一句：**它仍然需要校准数据和计算资源。**校准集是 500 条 FineWeb-Edu 序列、每条 1,024 token，stride-4 采样后每个目标 head 约有 128K 个 token 级观测。论文报告每个模型对的 mapper 在一台 8×H100 节点上拟合**约 47–87 分钟**。

---

## 四、做到了什么程度

### 4.1 六组模型对

论文在 Qwen3、Llama 3.1、Ministral 3 三个家族上测了六组 small-to-large 模型对。**Retention** 的定义是 `transfer accuracy / target standalone accuracy`，即转换后模型的准确率与目标模型独立 prefill 后准确率的比值。

| 模型对               | k   | 平均 retention | GSM8K |
| -------------------- | --- | -------------- | ----- |
| Qwen3 14B → 32B      | 8   | **97.6%**      | 95.6% |
| Qwen3 8B → 32B       | 12  | **87.5%**      | 68.8% |
| Ministral 3 3B → 8B  | all | **76.2%**      | 36.6% |
| Llama 3.1 8B → 70B   | 20  | **72.8%**      | 18.2% |
| Ministral 3 3B → 14B | 20  | 44.2%          | 3.2%  |
| Ministral 3 8B → 14B | 12  | 41.6%          | 1.6%  |

四组落在 73–98%，两组掉到 42–44%。论文把它们分成两个 tier。考虑随机基线后，两组失败 pair 的 floor-normalized 平均只剩 **14.7%** 和 **11.1%**。

论文报告里 Qwen3 14B → 32B 的 ARC-C retention 是 101.0%，超过 100 是因为这次 transfer accuracy 略高于 standalone 基线。**单项超过 100% 不足以支持"转换后模型普遍优于原模型"的结论**：这是一次评测的波动，不是系统性增益。

### 4.2 平均分掩盖的东西：GSM8K 的断层

4.1 那张表的最后一列是 GSM8K。14B → 32B 保住 95.6%，看起来很好。但四组"可用"的 pair 里，有三组掉到 70% 以下：8B → 32B 是 **68.8%**，3B → 8B 是 **36.6%**，Llama 8B → 70B 只剩 **18.2%**。两组失败 pair 更直接，3.2% 和 1.6%。

**知识与推理的退化不是一个量级。**

知识型任务（ARC、HellaSwag、WinoGrande、MMLU）的情况好得多：四组"可用"的 pair 在这四项上的 16 个单元格里，14 个在 87% 以上；两组失败 pair 落在 32%–74%。

平均 retention 把这两类任务的差距抹平了。对一个要跑 agent 或做数学推理的场景，**"平均 87.5%"是一个会误导人的数字**。

### 4.3 一个反直觉的发现：R² 不能预测 retention

按理说，拟合质量好（R² 高）应该意味着转换效果好。论文的数据否定了这一点。

两个 R² 相同的案例，表现完全相反（"小→大"即论文的 S→L，"大→小"即 L→S）：

| 模型对              | R²_K | 表现                                       |
| ------------------- | ---- | ------------------------------------------ |
| Llama 3.1 8B → 70B  | 0.84 | 小→大 保 94% HellaSwag，**大→小 只剩 37%** |
| Ministral 3 3B → 8B | 0.84 | 双向都保 93%                               |

拟合质量一模一样，结果一个方向好一个方向差。

论文进一步算了相关性（12 个 matched-KV pair，三个家族）：

- **Attention-output cosine** 与 HellaSwag retention 的 Pearson r = **+0.57**
- **R²_K** 与 retention 的 r = **−0.20**

R² 衡量的是"能不能重建出目标 KV"，但下游表现取决于"重建出来的 KV 送进 attention 后，输出对不对"。两者不是一回事。

**工程含义很实际：你在拟合之前无法预判哪个模型对能用。**只能先花 47–87 分钟拟合出来，跑一遍 benchmark，才知道结果。论文把"可预测的 transferability"列进了 future work，并指出目前缺一个**拟合前就能筛查的信号**。

### 4.4 非线性映射能救一部分

论文试过用 MLP 替换 ridge：两个 1,024-unit 隐层，Adam，lr 1e-3，20 epochs。

| 模型对               | ridge | MLP       | 变化         |
| -------------------- | ----- | --------- | ------------ |
| Qwen3 14B → 32B      | 97.6% | 97.3%     | −0.3 pp      |
| Ministral 3 3B → 8B  | 93.3% | 91.8%     | −1.5 pp      |
| Ministral 3 3B → 14B | 68.0% | **92.3%** | **+24.3 pp** |
| Ministral 3 8B → 14B | 58.7% | **95.5%** | **+36.8 pp** |

（表内为 HellaSwag retention）

两组失败的 pair 被 MLP 大幅救回，而本来表现好的两组反而略降。

这个结果本身就是证据：**在那两组 pair 上，$X_B \approx X_A \cdot T$ 这个前提不成立。**要用非线性才能拟合出可用的映射。论文的分析显示，MLP 把拟合误差从"attention 敏感的子空间"移开了（K-concentration 平均降约 2.5，cosine 平均升约 0.45）。

---

## 五、部署这笔账

论文的延迟收益很醒目：**mapper 比 target re-prefill 快 2.7–25.1×**，70 个测试单元里 mapper 全部更快。32K context、Qwen3 14B → 32B 这一项，mapper 用时 **278 ms**，re-prefill 用时 **6,975 ms**。

但要用起来，还有几笔账要算。

### 5.1 Mapper 本身的体积与方向性

| 模型对              | k   | Mapper 参数量 | 存储  |
| ------------------- | --- | ------------- | ----- |
| Qwen3 14B → 32B     | 8   | 1.07 B        | 4 GB  |
| Qwen3 8B → 32B      | 12  | 1.61 B        | 6 GB  |
| Llama 3.1 8B → 70B  | 20  | 3.36 B        | 12 GB |
| Ministral 3 3B → 8B | all | 1.85 B        | 7 GB  |

**1–3.4 B 参数、4–12 GB 存储**，几乎等于再养一个小模型。

而且**映射是有方向的**。小→大和大→小各自需要独立的 mapper，P 个模型最多需要 **P(P−1)** 个有序对。论文估算 3/4/5 个模型的 fleet 分别需要约 **39 / 79 / 131 GB** 的 mapper 存储。

加载开销论文按 25–50 GB/s 主机到设备链路上估算为 **80–480 ms**，但这是计算值不是实测值。好在 mapper 不需要常驻 GPU。

### 5.2 校准成本与敏感性

校准集是 500 条 FineWeb-Edu 序列，每对拟合 47–87 分钟。论文测了敏感性：

| 变量         | 设置           | HellaSwag 变化 |
| ------------ | -------------- | -------------- |
| $\lambda$    | 1              | **−15.79 pp**  |
| $\lambda$    | 0.1            | −0.98 pp       |
| 样本量 N     | 50             | −1.64 pp       |
| 样本量 N     | 1000           | +0.16 pp       |
| **校准语料** | **CodeAlpaca** | **−5.24 pp**   |
| 校准语料     | Wikipedia      | −1.05 pp       |

$\lambda$ 和样本量都比较鲁棒，但**校准语料的影响很大**：把 FineWeb-Edu 换成 CodeAlpaca，HellaSwag 掉 5.24 分。

论文自己把这一条列进 limitations，并指出附录只在一个 pair 上测了语料替换，"Neither substitution separates subject matter from register"：**无法界定把校准限定在医学、法律这类单一领域时会发生什么**。

### 5.3 延迟收益的正确口径

**2.7–25.1× 是 mapper application 对比 target re-prefill 的加速，不是端到端推理加速。**

论文附录明确说明了两点：延迟实验用的是 synthetic inputs，只为隔离 compute cost；**把映射后的 cache 送到目标进程的端到端传输成本没有测量**。

另外这个加速是**不对称的**（Qwen3 14B ↔ 32B）：

| 方向    | 32K mapper | 32K re-prefill | 加速      |
| ------- | ---------- | -------------- | --------- |
| 小 → 大 | 277.6 ms   | 6,975.3 ms     | **25.1×** |
| 大 → 小 | 427.1 ms   | 2,952.7 ms     | **7×**    |

小→大省得多，因为大模型的 prefill 本来就贵。

### 5.4 多轮 drift

长会话是论文自己列的动机，所以多轮表现很关键。论文用 CoQA（100 个约 15 轮的对话）测了 handoff 的累积误差：

- 小→大的差距从第 1 轮到第 10 轮扩大 **1.7 pp**
- 大→小的 drift 以 **0.33 pp/轮** 线性增长

论文的判断是两个斜率在十轮内都不至于级联失败，但**明确指出了大→小的线性 drift 在极长会话中会累积**。

对一个几百轮的长 agent 会话，这个斜率意味着什么，论文没有给答案。

---

## 六、边界，和一个判断

### 6.1 论文自己划的四条边界

| 边界                     | 论文原文                                                                 | 含义                                                                                                            |
| ------------------------ | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| **同族**                 | 限于 within-family transfer                                              | Qwen→Qwen、Llama→Llama、Ministral→Ministral。Qwen→Llama 列为 future work                                        |
| **matched-KV**           | 六个 pair 按构造都是 matched-KV                                          | KV Head 数和 per-head 维度都相同（8→8，128→128）。闭式 mapper **对维度不做结构性要求**，但 mismatched-KV 未测试 |
| **dense full-attention** | Sliding-window、local attention 和 attention-recurrent hybrid 不在范围内 | —                                                                                                               |
| **k 的选择**             | k 在报告所用的同一批 log-likelihood benchmark 上选出                     | 论文自陈这不等同于完整的 out-of-sample selection；附录 H 量化其影响最多 2.49 pp，且变动时总是向下               |

这四条来自论文 §5 Limitations，不是外部质疑。

### 6.2 架构排他性

第三条边界排除了当前长上下文的主力架构。论文只把它列为范围外，没有展开原因；下面的分析是本文补的。

**MLA**（DeepSeek 系列）：KV 是压缩后的 latent 向量，不存在"per-head K/V"可以逐头映射。前面那套推导（$K = X \cdot W_K$）在 MLA 上根本不成立，因为缓存里存的不是投影结果，而是一个低秩表示。

**混合注意力**（Kimi K3 的 69 层 KDA + 24 层 MLA）：不仅要映射 KV，还要搬 recurrent state。而 recurrent state 是**递归算出来的状态**，压根不是残差流的线性投影，第二节的推导对它完全失效。

论文在 future work 里把这条列了出来，明确点名了 attention-recurrent 架构（Nemotron 3）。

结果是：**方法覆盖的是上一代 dense 架构，而排除了正在扩张的那些。**

### 6.3 一个判断

比 97.6% 更能说明问题的是单层 R²：**0.56 / 0.32**。要靠拼 **8 层**才能到 0.79。

**这个数字度量的是：两个独立训练出来的模型，表示空间差得有多远。**

这个距离不是必然的。今天的训练目标里，没有任何一项鼓励同族模型之间保持 KV 空间兼容。两个模型各自优化自己的 next-token prediction，表示空间长成什么样是副产物。

换句话说：**如果模型族在设计阶段就约定一个共享的 prefix 接口（对齐的层深度、一致的 KV 几何、约定好的位置编码边界），这个问题根本不会存在。**KV 天然可迁移，不需要事后拟合，不需要校准数据，不需要 8×H100 上跑一小时，也不需要为 P(P−1) 个方向各存一份 mapper。

这篇论文做的是**事后补救**：用统计方法拟合一个本可以在架构设计阶段消除的差异。

回到 §1 的那笔账：一整层 TB 级的共享缓存，换模型时被废掉。这套方法能在同族、dense、且 $X_B \approx X_A \cdot T$ 恰好成立的模型对上，把它捞回一部分，而四组"可用"的 pair 里还有三组的 GSM8K 已经掉出可用区间。剩下的部分，仍然要从零重算。

这不否定它的价值。它第一次给出了"不同模型的 KV 空间之间到底有多少可利用结构"的系统性数据，划清了边界，还诚实列出了反例和失败模式。这种工作比宣称通用的更有用。

但方向感要说清楚：**它优化的那个问题，值得用更好的方式在更早的地方解决。**而且随着架构继续分化（MLA 已是主流、混合注意力在扩张），事后补丁的适用面只会继续收窄。

如果要在生产里用它，我的建议是把它当**特定形状下的工具**：同族、dense、长上下文、已有源模型 KV 的级联或迁移场景。用之前先接受两个事实：mapper 有体积（4–12 GB/方向），以及你在拟合完跑 benchmark 之前，不知道这对模型能不能用。

---

## 相关阅读

- [Prefix Caching](../prefix_caching/prefix_caching.md)——同模型内的 prefill 复用，本文讨论的跨模型问题是它的边界之外
- [RoPE 与前缀缓存](../prefix_caching/rope_and_prefix_caching.md)——RoPE 与缓存复用的关系，本文 §3.2 的 RoPE factoring 是同一问题的跨模型版本
- [PD 分离中的 KV 传输](../pd_transfer/01_disaggregated_prefill_kv_transfer.md)——同模型跨节点的 KV 搬运，与本文的跨模型搬运是两条正交的轴
- [KV Cache 量化](../compression/kv_cache_quantization.md)——本文的 mapper 在 bf16 下拟合，量化 KV 上的表现论文未涉及
- [KV Cache 容量规划：GLM-5 推演](../capacity_planning/glm5_kv_cache_capacity_planning.md)——§1 的 L3 容量推演（5.92 TB / 2,000 session）出处
- [B300 上的模型部署与 KV Cache](../../../deployment/b300-deployment-and-kv-cache.md)——§1 的 L3 共享约束与 agent 负载命中率数据出处
- [KV Cache 技术体系](../../README.md)——本目录的完整导航

## 参考资料与源文件索引

1. **Heo et al., _Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse_, arXiv:2608.03893v1, 2026-08-04（NVIDIA）**

   全部方法、实验与数字。§2 推导对应论文 §3；§3.1 对应 §3.2（cross-layer selection）与 Table 7；§3.2 对应 §3.3（content-space mapping）与 Table 2 消融；§3.3 对应 §3.1（per-head ridge）；§4.1 对应 Table 1；§4.3 对应 §4 的 R² 与 cosine 相关性分析；§4.4 对应 Table 3、Table 4；§5.1 对应 Table 12（附录 D）；§5.2 对应 Table 8、Table 9、附录 C；§5.3 对应 Table 5、Table 16、附录 G；§5.4 对应多轮 handoff 实验；§6.1 对应 §5 Limitations

2. **Realtime AI Lab《NVIDIA 论文解读：KV Cache 跨模型转换》，2026-09-09**

   中文解读，本文写作时作为交叉校验。其报告的数字与论文 v1 一致；本文补充了它未涉及的 mapper 体积、方向性、GSM8K 分项、R² 与 retention 的相关性分析

3. **本仓 [GLM-5 KV Cache 容量规划](../capacity_planning/glm5_kv_cache_capacity_planning.md)**

   §1 的 L3 容量推演（约 5.92 TB / 2,000 session）。**本仓自建推演，非实测**，口径为 GLM-5 INT4、30 KB/token、200 用户 × 10 session、30% prefix 命中、10% 碎片

4. **本仓 [B300 上的模型部署与 KV Cache](../../../deployment/b300-deployment-and-kv-cache.md)**

   §1 的 L3 共享约束（引自 SGLang 官方文档）、Mooncake 部署样例的 816 GB 全局共享段、agent 负载 95.7% 命中率与 fresh token 占比 19.0%（实测，来源为约 4,300 个 Claude Code 与 Codex session）

**时效性说明**：论文为 v1（2026-08-04），尚未见后续版本。文中所有 benchmark 数字均为论文报告值，本文未独立复现。`training-free` 一词在论文中特指不使用梯度训练，仍需要校准数据与拟合算力，引用时建议保留这一限定。
