# MTP 深度解析：把投机能力训进模型里

> Multi-Token Prediction（MTP）在训练阶段就为模型植入「一次预测 N 个未来 token」的能力，推理时利用这些内置预测头替代独立 draft model，在单次 forward 中同时完成草拟与验证。与投机解码相比，MTP 消除了独立 draft forward 的开销，但代价是无法对已发布模型追加——它不是推理技巧，而是训练承诺。

---

## 一、自回归的墙，投机解码的梯子

一个 70B 模型在 decode 阶段，单次 forward 的 compute-to-memory 比不到 20%——GPU 的大部分 SM 在等待 HBM 把下一条 KV cache 行送进来，而不是在计算。这不是硬件能力问题，而是**自回归协议**的固有约束：每步只能生成 1 个 token，下一个 token 依赖上一个 token 的 KV 状态，串行依赖链不可打断。

### 1.1 投机解码的答案：用另一个模型猜

[投机解码](illustrated-speculative-decoding.md) 提供了一个巧妙的绕行方案：让一个更快、更小的 **draft model** 先行草拟 K 个候选 token，然后让 target model 一次性批量验证。从接收率 $\alpha$ 和草拟-验证耗时比 $\rho$ 推导，当 $\alpha K > 1 + \rho K$ 时，投机解码带来净加速。在理想条件下，decode 延迟可以降低 2–3 倍。

但这个方案绑定了一个前提：**需要两次 forward**。Draft model 一次，target model 一次。两次 forward 意味着两倍的 KV cache 查询、两倍的 kernel launch 开销、两倍的调度器往返。

### 1.2 投机解码没回答的问题

投机的核心矛盾在于：草拟能力来自模型外部的独立模块（draft model），草拟质量越高，draft model 越接近 target model——直至极端情况 $\rho \to 1$，投机解码的收益归零。

这引出一个更根本的问题：**能不能把「预测未来 token」的能力，从外部 draft model 搬到模型内部？** 让模型在训练阶段就学会预测 token i+2、i+3、i+4——推理时，这些内置预测头自然地充当草拟者，不再需要独立的 draft model。

这个问题的答案，就是 MTP。

---

## 二、核心问题：训练一种新的预测能力

MTP 改变了两个层面的设计：

- **训练侧**：如何在 causal attention 的约束下，让模型学会从位置 i 同时预测 token i+1（常规）、i+2、i+3...？这需要新的架构模块和 loss 设计。
- **推理侧**：单次 forward 产出的多个候选 token，如何高效验证？验证失败后如何回退？

这两个问题不是独立的——训练时的架构选择直接决定了推理时的效率上限。

---

## 三、训练视角：MTP 头的诞生

### 3.1 标准因果 LM：只看一个方向

标准的 causal language model 在每个位置只做一件事：

$$\mathcal{L} = -\sum_{i}\log P_{\theta}(x_{i+1} \mid x_{\leq i})$$

模型的 hidden state 在位置 i 被投影到 vocab 维度，输出 token i+1 的概率分布。所有的参数——self-attention、FFN、layer norm——都在为「猜下一个」这一个目标优化。即使模型内部隐式地学到了一些「下文会是什么」的表示，也没有显式的训练信号要求它预测更远的 token。

### 3.2 MTP：级联的预测模块

MTP 在主模型的最后一层 hidden state 之上，级联 N 个预测模块。每个模块负责预测一个特定 offset 的 token：

```text
位置 i 的 hidden state h_i
  │
  ├── 主预测头 → P(x_{i+1} | x_≤i)     ← 标准 next-token
  │
  ├── MTP 模块 1 → P(x_{i+2} | x_≤i)   ← 多出来的预测
  ├── MTP 模块 2 → P(x_{i+3} | x_≤i)
  │   ...
  └── MTP 模块 N → P(x_{i+N+1} | x_≤i)
```

每个 MTP 模块是一个独立的 transformer block，接收两个输入：主模型的 hidden state $h_i$，以及前一个 MTP 模块的输出（级联连接）。在 DeepSeek V3 的实践中，MTP 模块共享主模型的 embedding 层和输出投影矩阵，从而大幅减少额外参数量——独立的 MTP 组件只有一个 transformer block + 一个线性投影。

### 3.3 Loss 设计：等权求和的信号

每个 MTP 头独立计算 cross-entropy loss：

$$\mathcal{L}_{\text{MTP}} = \mathcal{L}_{\text{main}} + \sum_{k=1}^{N} \mathcal{L}_{\text{MTP}_k}$$

DeepSeek V3 对每个 MTP 头赋予了相等的权重。这个选择背后有设计考量：预测 token i+2 和预测 token i+3 对模型表征能力的训练价值是等价的——都不是近距离的语法级预测，而是需要更全局的语义理解才能完成的跨越式预测。等权求和避免了对某一步预测的过度聚焦。

### 3.4 参数量代价：14B 换来的能力

DeepSeek V3 主模型 671B，MTP 模块增加约 14B 参数（~2%）。在 V3 中，MTP 深度 N=1——只预测一个额外的未来 token。14B 看起来不小，但相对于主模型的 671B 来说占比很低。更关键的是，MTP 不是一个可选插件——它在训练时就和主模型一起优化，推理时也必须加载进显存。这是 MTP 与投机解码最根本的差异：投机解码的 draft model 可以随意替换甚至关闭，MTP 是模型结构的一部分。

---

## 四、推理视角：一次 forward 等于一次草拟

### 4.1 单次 forward 的 N+1 个产出

标准的 decode 每步只输出 1 个 token 的概率分布。而在 MTP 模式下，一次 forward 同时产生：

- 主预测分布 $P(x_{i+1} \mid x_{\leq i})$
- MTP 预测分布 $P(x_{i+2} \mid x_{\leq i})$、$P(x_{i+3} \mid x_{\leq i})$、...

所有这些分布共享同一次 hidden state 计算。attention 层的 K、V 矩阵在整个 forward 中只计算一次——MTP 模块只消费已经存在的 hidden state，不产生额外的 attention 计算。

### 4.2 Self-Speculation：草拟与验证共享一个 forward

投机解码需要两步：draft model forward → target model forward。MTP 做的是 **self-speculation**——草拟和验证来自同一个模型、同一次 forward。

具体来说，推理时的 MTP 加速流程如下：

```text
Step T:
  Forward: hidden state h_T 经过主预测头 → 采样得到 token t_{T+1}
           h_T 同时经过 MTP 模块 → 采样得到 token t_{T+2}

  Token t_{T+1} 被确认输出。Token t_{T+2} 作为投机标记，暂不输出。

Step T+1:
  Forward: 输入序列 [...t_T, t_{T+1}]
           主预测头 → 采样得到 token t_{T+2}'

  验证: 比较 t_{T+2}' 与上一轮的 MTP 预测 t_{T+2}
    - 一致: t_{T+2} 被确认，本轮 MTP 预测 t_{T+3} 继续投机
    - 不一致: 接受主预测结果 t_{T+2}'，丢弃 MTP 预测，从分歧点继续
```

这个过程的核心好处是：**不需要第二次 forward 来验证**。验证发生在下一步的主预测中——这是自然的、零额外开销的。MTP 预测只是「提前」给出了一个候选，下一步的主预测自然验证它。

### 4.3 接受率：不是预测准确度，是分布一致性

MTP 的接受率 $\alpha$ 衡量的是：MTP 头在上一轮预测的 token，在下一轮被主模型「独立确认」的比例。影响 $\alpha$ 的关键因素：

- **温度（temperature）**：与投机解码相同，低温度意味着更确定性、更高接受率。贪心解码下 $\alpha$ 可达 85–95%；temperature=1.0 时降到 50–70%。
- **任务类型**：代码补全的 $\alpha$ 最高（语法和模式的规律性强），翻译次之（语义等价空间相对封闭），创意写作最低（开放式的下一词选择）。
- **MTP 步长**：每个额外的预测步长都会降低该步的独立接受率。第一个 MTP 头（预测 i+2）的接受率始终是最高的。

### 4.4 与因果注意力兼容：位置 i 如何「预测未来」？

一个自然的问题是：位置 i 的 hidden state 只能看到 token 0...i，它凭什么预测 i+2 的 token？

答案在训练阶段。MTP 模块在训练中接收两个信号：位置 i 的 hidden state $h_i$，以及 token i+1 的 embedding（来自前一个 MTP 模块的输出或主模型的预测）。通过跨位置的信息级联，MTP 模块学会了「在已知 token i+1 的条件下，i+2 最可能是什么」。在推理时，token i+1 被主模型在当前 step 预测出来，MTP 模块基于此推测 i+2。

这是 MTP 的关键架构洞察：**不是凭空预测更远的 token，而是基于「主模型已经预测出来的中间 token」做链式推测。**

---

## 五、对比：MTP 的边界在哪里

### 5.1 不是投机解码的替代品

将 MTP 理解为「投机解码的升级版」是一种常见的误解。他们解决同一个问题（加速 decode），但机制和适用场景截然不同：

| 维度             | MTP                            | 投机解码                       |
| ---------------- | ------------------------------ | ------------------------------ |
| **预测来源**     | 模型自身的 MTP 模块            | 独立 draft model 或 Eagle head |
| **何时决定**     | 训练阶段（模型设计的决定）     | 部署阶段（运维的选择）         |
| **能否追加**     | 否（训练时嵌入）               | 是（任何已有模型可接入）       |
| **模型质量影响** | 训练时提升表征能力             | 无影响（纯推理技巧）           |
| **Forward 次数** | 1 次（草拟和验证共享）         | 2 次（draft + verify）         |
| **额外显存**     | MTP 模块参数（~2%）            | Draft model 参数 + KV cache    |
| **灵活性**       | 低（绑定模型，MTP 头不可替换） | 高（可更换 draft、可关闭）     |
| **接受率**       | 与温度高度相关的确定性         | 取决于 draft-target 分布一致性 |
| **代表案例**     | DeepSeek V3 / V4               | LLaMA 3 70B + 8B draft         |

### 5.2 MTP 真正的优势

MTP 的核心优势只有一个字：**省**——省掉了一次 forward。在 decode 阶段，一次 forward 的耗时是可观的（即使有 KV cache，模型仍需要处理 batch_size × num_layers 的 attention 和 FFN）。省掉 draft forward 意味着 MTP 的 wall-clock 加速比天然高于等条件配置的投机解码。

但这不意味着 MTP 在所有场景下都优于投机解码。当使用一个极小的 draft model（如 0.5B 的 Qwen2 对 72B 的 target）时，draft forward 的开销极低（$\rho \approx 0.01$），投机解码的额外开销几乎可以忽略，而 MTP 的 2% 参数始终需要加载。

### 5.3 选择 MTP 还是投机解码？

用一条规则概括：

- 如果你是**模型发布者**（训练了模型）→ MTP。训练时的一次性投入，所有使用者永久受益，且不需要额外的 draft model 配置。
- 如果你是**推理服务方**（使用他人模型）→ 投机解码。对已发布模型无法追加 MTP，但可以在部署层面选择合适的 draft model 组合。
- 如果你在**使用 DeepSeek 系列**→ MTP 已经内置。V3/V4 的 MTP 模块是模型结构的一部分，vLLM 通过 `deepseek_mtp` 方法将其直接作为草拟源——用户只需配置 `num_speculative_tokens`，无需额外指定 draft model。

---

## 六、工程落地：vLLM 中的 MTP

### 6.1 专用 speculative 方法：`deepseek_mtp`

vLLM V1 为 DeepSeek V4 提供了专门的 `deepseek_mtp` speculative decoding 方法，而非简单地复用通用 Eagle 路径。启用方式：

```bash
--speculative-config='{"method": "deepseek_mtp", "num_speculative_tokens": 1}'
```

架构上，MTP 遵循与 Eagle 相同的 self-speculation 模式——草拟与验证共享同一个 base model。但 V4 的实现有两个关键差异：

**Target model 侧**：`DeepseekV4ForCausalLM` 暴露 `get_mtp_target_hidden_states()` 方法，从主模型最后一层 hidden state 中提取 MTP 草拟所需的表示。这个 hidden state 是 pre-`hc_head` 的残差流缓冲——MTP 模块不是在 attention 输出之后再独立计算，而是直接消费主模型已经算好的中间结果。

**Draft model 侧**：`DeepSeekV4MTP`（继承 `nn.Module`）接收三个输入：`input_ids`（token embedding）、`positions`（位置编码）、以及 target model 传来的 `hidden_states`。它的 `forward()` 返回 draft hidden states，`compute_logits()` 应用 hypercompressed head 将其投影到 vocab 维度。

这与通用 Eagle 路径的区别在于：`deepseek_mtp` 利用了对 V4 模型结构的精确知识——知道哪些层产出的 hidden state 最适合 MTP 草拟、知道 hypercompressed head 的内部维度——因此不需要像通用 Eagle 那样做适配层的额外映射。

### 6.2 V4 MTP 模块的架构细节

对比 V3 的 MTP 实现，V4 在 `DeepSeekV4MTP` 内部引入了几个 V4 专属的架构组件：

**`e_proj` / `h_proj` 分离投影**。V3 的 MTP 使用融合的 `eh_proj` 做 hidden state 和 embedding 的联合投影。V4 将这两者拆分为独立的 `e_proj` 和 `h_proj`，各自带 FP8 线性量化——这不仅减少了参数耦合，也允许 embedding 路径和 hidden state 路径使用不同的量化精度。

**`hc_head` 超压缩词表投影**。V4 的 MTP 不直接使用主模型的 `lm_head` 做 logits 计算，而是通过一个独立的 `hc_head`（hypercompressed head）将 MTP hidden state 映射为 logits。这个 head 的维度远小于 `lm_head`（超压缩），只在 MTP 草拟阶段使用——主模型的 `lm_head` 仍然负责最终的 token 预测。

**`DeepseekV4DecoderLayer` 与 aux-stream 管理**。MTP 模块内部的 transformer 层使用 V4 专属的 decoder layer，自带 aux-stream（辅助流）管理——这是 V4 注意力架构（CSA/HCA）在 MTP 模块中的对应实现。aux-stream 在 MTP 草拟时提供额外的上下文信息，不参与主模型的 attention 计算。

**权重复写与量化检测**。`load_weights()` 执行多层映射：checkpoint 中的 `mtp.{i}.*` 被重写为 `model.layers.{num_hidden_layers + i}.*`，使得 vLLM 的 spec layer 识别逻辑能正确标记 MTP 层。同时，`_mtp_block_is_quantized_on_disk()` 扫描 `model.safetensors.index.json` 中 `mtp.*` 键的量化 scale 后缀——如果未检测到量化参数，说明 MTP 权重以 BF16 存储，则跳过 quant_config，避免因精度不匹配导致的 KeyError 或 AttributeError。加载完成后，还会验证所有 MTP spec layer 都有至少一个参数被成功加载，缺失则抛出 `ValueError`。

### 6.3 与 FlashMLA / FlashAttention 3 的共存

MTP 的 forward 不会与 attention 后端产生冲突。因为 MTP 模块消费的是主模型 attention 层产出的 hidden state（通过 `get_mtp_target_hidden_states()`），而非参与 attention 计算本身，它天然地与 FlashMLA（MLA 专用后端）和 FlashAttention 3（通用后端）共存。在一个 FlashMLA 管理的 decode forward 中，MTP 模块接收的是已经完成的 attention 结果，做额外的 token 预测——不修改 attention 流程，不增加 attention 的显存读写。

### 6.4 实际表现

在 AIME 2024 基准上，DeepSeek V4-Flash 开启 MTP（NVFP4-FP8 量化，`num_speculative_tokens=1`）的表现：

- 草拟 token 接受率 **81.6%**（concurrency=8）
- 接受率在 c=1 到 c=16 之间保持 ~88% 平坦——批处理不降低草拟质量
- Wall-clock 加速比约 **2.95×**（相对不开 MTP 的等硬件配置）

接受率不随并发增加而衰减，说明 MTP 的草拟质量不受 batch 内其他请求的 KV cache 状态干扰——这是 self-speculation（不依赖外部 draft model）的天然优势。

---

## 七、取舍与边界

### 7.1 三个不可回避的局限

**训练耦合**。MTP 是一个训练时决定，不是一个部署时配置。对已发布模型，无法「开启 MTP」——只能在使用该模型时被动受益于训练阶段植入的 MTP 能力。这不是一个「接入」方案，而是一个「架构」方案。

**接受率对多样性的妥协**。与所有投机机制一样，MTP 的接受率随输出多样性的增加而下降。在 temperature > 1.0 的场景下，MTP 的加速收益可能归零甚至为负（回退开销 > 预测收益）。对于创意写作、对话生成等需要多样性的场景，MTP 的收益自然低于代码生成等确定性场景。

**首 token 延迟无改善**。MTP 只加速 decode 阶段——它是一个关于「每步生成几个 token」的优化，与 prefill 阶段的首 token 延迟（TTFT）无关。如果延迟瓶颈在 prefill 阶段（长 prompt、Chunked Prefill 未开启），MTP 不会带来显著改善。

### 7.2 组合空间

MTP 与其他加速技术的组合可能性：

| 组合                 | 可行性      | 效果                                                                       |
| -------------------- | ----------- | -------------------------------------------------------------------------- |
| MTP + Prefix Caching | ✅ 已支持   | 互补——各自加速不同阶段（Prefix Caching 加速 prefill，MTP 加速 decode）     |
| MTP + 量化           | ✅ 已支持   | MTP 模块同样可量化（V3 支持 FP8），但接受率对精度损失更敏感                |
| MTP + 投机解码       | ⚠️ 理论可行 | MTP 头 + 外部 Eagle draft 可形成两层草拟，但调度复杂性陡增，当前无成熟实现 |
| MTP + CUDA Graph     | ✅ 已支持   | vLLM 中 MTP 的 forward 可被捕获进 CUDA Graph                               |
| MTP + PD 分离        | ✅ 已支持   | MTP 只影响 decode 节点，与 PD 分离架构正交                                 |

### 7.3 未来方向

DeepSeek V3 使用 1 个 MTP 模块（深度 N=1）。V4 延续了 MTP 路线，在 Flash 版本中 MTP 与稀疏注意力（NSA/DSA）和压缩表示（CSA/HCA）深度整合。未来的趋势可能是：**MTP 深度的自适应调节**——根据实时接受率动态调整使用几个 MTP 头的预测；以及 **MTP + MoE 的更深融合**——MTP 模块本身采用 MoE 结构，进一步降低额外参数的计算开销。

另一个值得关注的方向是 MTP 与 contrastive decoding 的结合。MTP 头的预测分布可以被视为一种「弱版本」的未来分布——将其与主模型的当前分布做对比，有助于在不需要额外模型的情况下实现对比解码的效果。

---

## 八、一句话总结

MTP 把「预测未来 token」的能力从外部 draft model 搬到模型内部——训练时植入，推理时激活。它不是投机解码的替代品，而是在「模型发布者有能力修改训练流程」这个前提下，提供了一条更彻底的优化路径：省掉 draft forward，让 main forward 既是草拟者也是验证者。

> 相关阅读：
>
> - [图解投机解码](illustrated-speculative-decoding.md) — 投机解码的完整原理与性能模型，本文多处对比的基础
> - [投机解码与 KV Cache 交互](../kv_cache/01_concepts/scheduling/02_vllm_spec_decode.md) — Placeholder、回滚、草稿 KV 的工程细节
> - [DeepSeek V4 长上下文注意力支持解析](../vllm/module_analysis/deepseek_v4.md) — V4 的注意力架构与 MTP 的 vLLM 集成
> - [DeepSeek 注意力架构进化：从 MLA 到 CSA/HCA](../vllm/module_analysis/deepseek_attention_evolution_mla_to_csa_hca.md) — 注意力机制的演进脉络，MTP 运行其上的注意力基础
