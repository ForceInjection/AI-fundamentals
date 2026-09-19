# 没有 KV Cache 的模型：delta-rule 线性注意力的推理挑战

> 标准 Transformer 的推理系统为一个事实而建：模型要为每个 token 存 KV Cache，于是有了 PagedAttention、前缀复用、分层存储、容量推演这一整套技术栈。delta-rule 线性注意力把这个事实改掉了：历史被压进一个固定大小的循环状态，不存在 token 序列的 KV。这不是某一个模型的选择，而是一整条正在成型的谱系：Qwen3-Next/3.5/3.8 的 Gated DeltaNet、Kimi Linear/K3 的 KDA，连混合比都约定俗成地取 3:1。
>
> 2026-09 | 基于 vLLM（`43d691ec6b`，2026-08-07）与 SGLang（`f7101b0ae6`，2026-08-18）源码验证；架构结论引自 Kimi Linear（arXiv:2510.26692）、Kimi K3（arXiv:2607.24653）技术报告
>
> **性质说明**：机制与行为均经源码验证；性能数字（6.3×、75% 等）为各技术报告口径；涉及耗时的数字为量级示意。本文是 [post-KV-cache 新挑战](../kv_compression/01-post-kv-cache-era.md) §3/§5 两块「需要解决」的纵深展开。

---

## 一、谱系、门控与 3:1 公约数

谱系先交代一句：DeltaNet 是学术起点，Gated DeltaNet 把门控加成标量级并随 Qwen3-Next 进入主力产品线，KDA 再把门控细化到通道级、随 Kimi 系落地。这里只列落地情况：

| 模型                  | 线性注意力层              | 门控粒度 | 引擎支持                      |
| --------------------- | ------------------------- | -------- | ----------------------------- |
| Qwen3-Next（80B-A3B） | Gated DeltaNet            | 标量门控 | vLLM、SGLang（Day-0，2025-09） |
| Kimi Linear（48B-A3B）| KDA（细粒度通道级门控）   | 通道级   | vLLM、SGLang                  |
| Kimi K3（2.8T）       | KDA + Gated MLA + AttnRes | 通道级   | vLLM、SGLang                  |
| Qwen3.5               | 延续混合路线（GDN）       | 标量门控 | vLLM、SGLang（Day-0，2026-02） |
| Qwen3.8（2.4T-A95B）  | 69 层 GDN                 | 标量门控 | SGLang（Day-0，2026-08）       |

共同点有三条：3:1 的混合比（3 层线性层配 1 层全注意力/MLA 层，线性层省成本，少数全注意力层保全局交互能力）；循环状态替代 KV 序列（历史压缩进与序列长度无关的状态矩阵）；位置信息走衰减门（不需要 RoPE）。公开口径下，K3 在 1M 上下文的 decode 提速 6.3×、KV 体积降至全注意力基线的零头（技术报告口径，见 [post-KV-cache 篇 §1](../kv_compression/01-post-kv-cache-era.md) 的核验记录）。

对推理系统来说，适配已经落进了两个主流引擎：vLLM 合入了 `GDNAttentionBackend`、`mamba_attn` 后端与 `MambaSpec`，SGLang 合入了 `hybrid_linear_attn_backend` 与自研 KDA PTX kernel（`layers/attention/linear/kda_ptx.py`）。至于适配的成色，从第二节开始逐层拆。

---

## 二、三个被打破的系统假设

### 2.1 prefill 并行性：串行账

标准 attention 的 prefill 所有 token 一起算；线性层的推理要走 chunkwise 循环：chunk 内并行，chunk 间串行传递状态。SGLang 的 chunk 大小取自 FLA 库的 `FLA_CHUNK_SIZE`（缺省 64，`server_args.py:9092-9103` 有引擎侧的取舍逻辑），1M（10⁶）上下文就是约 15,625 个串行步。

这笔账的两个性质（[post-KV-cache 篇 §3](../kv_compression/01-post-kv-cache-era.md) 已推导，此处只列结论）：tensor 并行不减少串行步数，只能缩短每步延迟；TTFT 的下限由「步数 × 每步延迟」决定。K3 报告的应对是三件套：给衰减门加下界（scaled sigmoid，让 chunk 内的对角块能上 Tensor Core）、fused kernel 把 chunk 内计算合成一次、以及 KDA Context Parallelism（跨设备切序列分摊串行开销，报告 §5.1.2）。

### 2.2 前缀缓存语义重写：从「token 匹配」到「状态检查点」

标准前缀缓存回答的问题是「这个请求和已有请求共享多少个 token」。线性层没有 token 级 KV，问题改成「这个请求能否从某个已缓存的状态检查点续起」。检查点落在 block 边界（每 `block_size` 个 token 存一次状态快照），命中的收益与代价都变了：命中的是一个状态矩阵（大小固定，与命中长度无关），失效的代价是从上一个检查点重放。

vLLM 用一个三档开关管理这件事（`vllm/config/cache.py:38`）：

```python
MambaCacheMode = Literal["all", "align", "none"]
```

`mamba_cache_mode` 默认 `"none"`（前缀缓存关闭时）；`"all"` 在每个 block 边界存全部检查点；`"align"` 只在每个调度步的最后一个 token 处、且恰逢 block 边界时存，是前缀缓存开启时的默认档（`cache.py:134-142`）。缓存对象由 `MambaSpec`（`v1/kv_cache_interface.py:710`）纳入 KV 管理框架。

SGLang 走得更远，把状态检查点做成了独立的缓存子系统：`mem_cache/mamba_radix_cache.py`（状态的 radix 索引）、`mamba_checkpoint_pool.py`（检查点池）、外加一条精细的逐出策略：`mamba_max_states_per_path`（`server_args.py:2544`）限制每条 root-to-tail 路径保留的检查点数，超限时优先淘汰最浅的内部状态，同时保住尾部、分叉点和被引用（locked）的节点。这已经是「状态版的缓存逐出策略」，复杂度对标 KV 侧的 radix tree 管理。

### 2.3 双账本：混合架构的调度税

3:1 交错意味着每个请求同时持有两种生命周期完全不同的资源：循环状态（大小固定、必须整体存在、没有「部分释放」）和 MLA KV（随序列增长、可分页、可部分复用）。显存规划从一本账变两本账：KV 池有压缩与逐出的弹性，状态池是硬容量。

引擎的解法是按层分派：SGLang 的 `HybridLinearAttnBackend`（`hybrid_linear_attn_backend.py:952`）持有全注意力与线性注意力两个后端，按层号分派 forward（线性侧是 `MambaAttnBackendBase` 的子类，:45），对上层屏蔽差异。屏蔽得越好，上层越容易忘记两本账的存在，这正是容量规划最容易踩的坑（见 4.3）。

---

## 三、双引擎实现对照

| 维度       | vLLM                                                       | SGLang                                                         |
| ---------- | ---------------------------------------------------------- | -------------------------------------------------------------- |
| 线性层后端 | `GDNAttentionBackend` + `mamba_attn` / `mamba1_attn`       | `HybridLinearAttnBackend`（:952）+ `MambaAttnBackendBase` 子类（:45） |
| Kernel     | GDN prefill 三档：triton / flashinfer / cutedsl（按硬件自动选，`qwen_gdn_linear_attn.py:88`） | 自研 KDA PTX kernel（`linear/kda_ptx.py`）+ FLA                |
| 检查点模式 | `all / align / none` 三档（`config/cache.py:38`）          | `mamba_radix_cache` + `mamba_checkpoint_pool`                  |
| 检查点逐出 | —（随前缀缓存统一管理）                                    | `mamba_max_states_per_path` 按 path 限额逐出（:2544）          |
| 状态池     | `MambaSpec`（`kv_cache_interface.py:710`）                 | `max_mamba_cache_size`（:2533）、`mamba_ssm_dtype`（:2536）    |
| 投机解码   | `MambaSpecDecodeGPUContext`（`worker/mamba_utils.py:481`） | `mamba_slot_fused` 批量清/拷 conv 槽位（draft 扇出优化）        |
| 模型覆盖   | Qwen3-Next/3.5/MTP、Kimi K25/K3 等                         | kimi_linear / kimi_k3 / qwen3_next_mtp 等                      |

值得单独说的两处：

**vLLM 的 ReplaySSM**（`config/cache.py`，`use_replayssm`，默认关闭）：decode 时把最近的 SSM 输入存进一个长度 B（默认 16）的环形缓冲，攒满才把检查点刷回 HBM，跳过每步的全状态写。这是线性层版的「连续批处理省写」思路，条件是 Triton mamba 后端 + 非投机 decode。

**SGLang 的逐出策略粒度**。`mamba_max_states_per_path` 的说明值得读原文：超限时移除「最浅的合格内部状态」，同时保留尾部、分叉与被锁定的节点。翻译一下：一个会话树的中间检查点可以被淘汰（损失的是从该点分叉的复用），但活跃会话的续点（尾部）和多会话共享的起点（分叉）优先存活。这套策略和 KV 侧 radix tree 的 `lock_ref` 是同构思想在不同对象上的重演。

## 四、三个权衡

**checkpoint 密度 vs 恢复成本**。检查点存得密，任意位置可精确续起，但状态写放大；存得疏，命中后要重放到目标位置。vLLM 的 `all`/`align` 两档、SGLang 的 per-path 限额，都是在调这个旋钮。选择依据是前缀复用率：复用高的负载（Agent 会话）值得密，复用低的负载省着写。

**chunk 大小**。步数与 chunk 内计算量的乘积固定，但延迟特性不同：chunk 小则串行步数多（每步一次状态依赖等待），chunk 大则块内 O(C²) 的因果矩阵乘变大。FLA 缺省 64 是工程折中，而引擎侧还要把模型 chunk 大小与页大小取 max 并要求整除（`server_args.py:9092` 起），说明缓存对齐与计算效率对 chunk 的诉求并不一致。

**状态池硬顶 vs KV 弹性**。KV 池可以靠逐出、压缩、分层腾挪，状态池的并发上限是刚性的：每个活跃会话占一份固定大小的状态，池满就是不能接新会话。容量规划时线性模型的并发上限要按状态池算，不能沿用「KV 可压缩」的直觉。这是对 [Agent 负载一篇 §2.4](../agent_serving/agent-workload-serving.md) 容量公式的补充：混合架构下两本账都要算，且状态这本没有弹性。

---

## 五、三个问题的纵深展开

上面三个系统问题在本目录另有深入拆解：

- **prefill 串行**（§2.1）→ [02 · chunkwise 串行 prefill 的调度](02-chunkwise-scheduling.md)：区分模型级串行与调度级切块。结论是轮间插 decode 两引擎早已解决，真正缺的是三个调度原语：串行权重、完成时限、检查点感知切块。
- **状态检查点**（§2.2）→ [01 · KDA 机制拆解](01-kda-mechanism.md) 讲串行的机制来源，[03 · 状态语义深水区](03-state-semantics.md) 讲检查点在回滚、分叉、重放一致性上的语义代价。
- **双账本与容量**（§2.3、§4.3）→ [03](03-state-semantics.md) 的 COW 关闭与状态独占分析，以及它对 [Agent 负载篇 §2.4](../agent_serving/agent-workload-serving.md) 容量公式的补充。

三篇之后仍然开放的问题：**状态感知的准入**（池满时拒绝新会话还是驱逐旧检查点？SGLang 的 per-path 限额只管「存多少」，不管「谁进来」）；**ReplaySSM 的覆盖面**（缓冲刷写限定 Triton 后端 + 非投机 decode）；**chunk 间隙的调度原语**（见 [02 篇 §四](02-chunkwise-scheduling.md) 的三点设计空间）。

待实测项挂 InferenceX 排期：同一多轮 trace 在「状态检查点开/关」「chunk 大小」两轴下的 turn-resume TTFT。线性层的 resume 成本曲线和 KV 模型形状不同，值得单独测。

---

## 结语

KV 时代，推理系统管理的对象是「序列」；线性注意力时代，对象变成了「状态」。序列可以部分命中、部分释放、按 token 计价，状态只能整体存在、按检查点续起、按会话占位。两引擎的适配代码把能自动迁移的部分迁完了（分派、容量抽象、检查点纳入 KV 框架），剩下的都是迁移不了的部分：调度要理解串行，缓存要重写语义，容量要认硬顶。所以评估一个引擎的线性注意力支持，除了看能不能跑，更要看这三件事各做到了哪一层。

---

## 源文件索引

| 文件                                                               | 关键符号                                                            | 引用点                   |
| ------------------------------------------------------------------ | ------------------------------------------------------------------- | ------------------------ |
| `vllm/config/cache.py`                                             | `MambaCacheMode`、`mamba_cache_mode`、`use_replayssm`               | :38、:134-142、:150 附近 |
| `vllm/v1/kv_cache_interface.py`                                    | `MambaSpec`                                                         | :710                     |
| `vllm/v1/worker/mamba_utils.py`                                    | `MambaSpecDecodeGPUContext`                                         | :481                     |
| `vllm/v1/attention/backends/gdn_attn.py`                           | `GDNAttentionBackend`                                               | —                        |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`     | `_resolve_gdn_prefill_backend`（triton/flashinfer/cutedsl 三档）    | :88                      |
| `vllm/v1/attention/backends/mamba_attn.py`                         | mamba 注意力后端                                                    | —                        |
| `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` | `HybridLinearAttnBackend`、`MambaAttnBackendBase`                   | :952、:45、:832          |
| `python/sglang/srt/layers/attention/linear/kda_ptx.py`             | KDA PTX kernel                                                      | —                        |
| `python/sglang/srt/mem_cache/mamba_radix_cache.py`                 | 状态 radix 索引                                                     | —                        |
| `python/sglang/srt/mem_cache/mamba_checkpoint_pool.py`             | 检查点池                                                            | —                        |
| `python/sglang/srt/server_args.py`                                 | `max_mamba_cache_size`、`mamba_max_states_per_path`、FLA chunk 逻辑 | :2533、:2544、:9092-9103 |

_版本：vLLM `43d691ec6b`（2026-08-07）、SGLang `f7101b0ae6`（2026-08-18）。性能数字为技术报告口径；耗时数字为量级示意。_

---

## 参考资料

- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692), arXiv:2510.26692, 2025——KDA 原始出处
- Kimi Team, [Kimi K3 技术报告](https://arxiv.org/abs/2607.24653), arXiv:2607.24653, 2026——K3 的混合架构与 KDA Context Parallelism
- vLLM, [Qwen3-Next 支持公告](https://vllm.ai/blog/2025-09-11-qwen3-next), 2025-09——Gated DeltaNet 混合架构的引擎落地
- Sebastian Raschka, [Gated DeltaNet for Linear Attention](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/08_deltanet/README.md), LLMs-from-scratch——DeltaNet 谱系的教学级拆解
- LMSYS, [SGLang and Miles Add Day-0 Support for Qwen3.8](https://www.lmsys.org/blog/2026-08-12-qwen3-8-day0-support), 2026-08——delta-rule 谱系扩张的最新样本（69 层 GDN、2.4T-A95B）
