# 素材：成本技术（物理三维框架 / 模型架构 / KV Cache / 硬件 / 价格历史）

> 用途：支撑演讲第三部分「成本革命——在计算/存储/通信三维上的稀疏化」。
> 统一框架：token 成本 = 计算（FLOPs）+ 存储（容量+带宽）+ 通信（卡内/卡间），三者最终都以**能量**计价（黄仁勋的 Tokens/W 即把三维压缩为一维的工厂 KPI）。技术革命的共同哲学是**稀疏化**——能不算的不算、能不存的不存、能不传的不传；而**能量不可稀疏化**（1GW 的工厂不会变成 2GW），决定了产业终局是"每瓦 token"的竞赛。

## 〇、模型架构革命：三维上的稀疏化（源头层）

### 0.1 计算维：参数越来越多怎么办 → 参数/深度稀疏

- **MoE（混合专家，参数稀疏）**：只激活部分专家。DeepSeek-V3 总参数 671B、每次推理仅激活 37B（5.5%，节省约 87% 算力）；V4-Pro 总参数 1.6 万亿、仅激活 64 个专家（490 亿参数，激活比例约 **3.06%**）。来源：国泰君安、腾讯云 V4 解析。
- **深度稀疏（背景概念）**：学界有层深度路由思路——MoD（Mixture of Depths，DeepMind 2024 年论文）：不是所有 token 都值得走完全部层，每层 router 预测该 token 是否需要本层计算，简单 token 浅层通过，推理 FLOPs 按比例下降。**注意：DeepSeek 官方从未使用 MoD 术语**，其计算稀疏化通过 MoE（参数稀疏）+ DSA（注意力选择）+ Engram（条件记忆查表）实现——DSA 稀疏的是"哪些 token 参与注意力"，Engram 是"记忆-计算分离"，均非层深度路由。演讲中 MoD 仅作背景一句话提及，不得暗示 DeepSeek 采用。
- **Engram（表达稀疏，V4 引入）**：更激进——"多少 token 不需要被理解？"查表替代深层重建，把不必要的深度计算直接省掉。来源：CSDN《深入理解 DeepSeek Sparsity——从 MoE 到 DSA，再到 Engram》。
- 一句话：**智能不需要全部参数，也不需要全部深度——把算力花在最值得的 token 上。**

### 0.2 存储维：上下文越来越长怎么办 → 记忆格式/范围/精度压缩

- **MLA（多头潜注意力，格式压缩）**：DeepSeek-V2（2024-05）首创，缓存前把完整 KV 压缩进低维潜空间（512 维），KV Cache 减少约 **93.3%**（约 16 倍）；"吸收技巧"使 query 直接在潜空间点积，推理无需真正解压。效果：最大生成吞吐提高 5.76 倍，训练成本省 42.5%。被 Kimi K2、GLM-5 等主流模型采纳。来源：百度百科、国泰君安、SemiAnalysis。
- **DSA（DeepSeek 稀疏注意力，注意力选择）**：DeepSeek-V3.2-Exp（2025-09）引入，"先筛选、后计算"——闪电索引器（仅约自注意力 5% 的计算量）给历史 token 打分，细粒度选择器只保留 Top-k（128K 上下文选 2048 个）做完整注意力。复杂度 O(L²) → O(L·k)，注意力计算量减少约 **98%**（2048/131072 ≈ 1.6%，与官方及百度百科口径一致）；128K 上下文处理速度 +1.8 倍、GPU 内存占用 -40%；GLM-5 集成案例 H800 上推理成本降 40–50%、性能损失 <1%。**归属注**：DSA 本质是计算维稀疏（省注意力 FLOPs），其 KV 收益是次要的——在三维框架中挂"存储维"仅因演讲叙事方便，引用时说明其为跨维机制。来源：百度百科、百度千帆。
- **量化（精度压缩）**：4-bit 权重把每 token 权重读取量从 74GB 降至 18.5GB（decode 是 memory-bound，读取量直接决定速度）；KV 量化 FP16→INT8 再省 50–75%。
- **端到端架构结果**（本仓库 `post-kv-cache-era-challenges.md`）：V4 的三大稀疏机制 = MoE（参数）+ Engram（条件记忆）+ DSA（注意力选择），1M 上下文下 V4-Pro 单 token 推理 FLOPs 是 V3.2 的 **27%**、KV 是 **10%**；V4-Flash FLOPs 仅 **10%**、KV 仅 **7%**；相对标准 BF16 GQA8 基线 KV 体积降至约 **2%**（250GB → 5GB）。Kimi K3 走另一条路（KDA 线性注意力，复杂度 O(N) + Gated MLA），1M 上下文 decode 加速 6.3×。**注**：mHC（流形约束超连接）是训练稳定性/层间连接设计，不属于稀疏化机制，勿与三大机制并列。
- 一句话：**记忆不必完整（压缩后仍可恢复），也不必全部看（只读相关的）。**

### 0.3 通信维：模型大到必须拆卡 → 协作成本成为新瓶颈

- MoE 的 all-to-all（每个 token 跨卡找专家）、TP 每层 all-reduce、PD 分离的 KV 跨机搬运——模型越大，通信越接近瓶颈。
- 定量证据（本仓库 `post-kv-cache-era-challenges.md`）：V4-Pro 每 GBps 带宽需对应 **6.1 TFLOP/s** 算力才能隐藏通信；NVLink 4.0 900GB/s 可满足卡内，跨节点 25G/100G 以太网不足。
- 对策：EP 专家并行与专家本地化、拓扑感知调度、PD 分离就近传输（腾讯太极 16×H20 实测 15,800 tok/s，PD 分离端到端 +30–40%）。
- 一句话：**分工产生协作成本——架构选择决定通信拓扑；这是三维中最难稀疏化的一维。**

### 0.4 能量维：不可稀疏化的物理底线

- 计算、存储、通信都能"少做"，但供电与散热是物理硬约束：1GW 的 AI 工厂永远不会变成 2GW（黄仁勋）。
- 因此 Tokens/W 成为工厂 KPI；数据中心选址从"网络枢纽"变"电力枢纽"（西部绿电）；电费占比随规模上升（折旧可摊薄、电力摊不薄：小型集群硬件 60–70%→大型 40–50%，电力 5–10%→10–15%，见 `references/03` §5 旁证）。
- 硬件代际 = 能量维的持续抬升：同一个 1GW 数据中心两年 token 生成速率 200 万 → 7 亿（**350 倍**，同期摩尔定律约 1.5 倍）；Grace Blackwell NVLink72 每瓦 token 比 **Hopper 系（H100/H200）** 高 35–50 倍（黄仁勋先报 35 倍后承认故意保守，SemiAnalysis 实测约 50 倍）；Vera Rubin 官方口径为"每百万 token 成本降至 Blackwell 的 1/10"（"再提 2–10 倍"为厂商口径，未检索到独立测试）；液冷 PUE ≤1.1（白皮书口径）vs 传统 1.5–2.0（GTC 另有 PUE ≤1.2 口径，引用时注明）。

### 0.5 三维稀疏化 × 每瓦产出的完整叙事

模型架构（源头：稀疏化设计）→ 系统引擎（执行：分页/复用/批处理/PD 分离，见下文 §1–§2）→ 硬件载体（物理：每瓦产出）——三层相乘，才有三个数量级的降价空间。**技术革命是乘法不是加法。**

---

## 一、KV Cache：隐形成本之王

### 1.1 数量级直觉

- **高并发下 GPU 显存 70%+ 被 KV Cache 占用**，不仅限制单卡并发与上下文长度，还会因缓存驱逐导致的重复计算推高算力成本（NVIDIA 测算）。
- **未做 KV 卸载优化的推理集群，单位 Token 生成成本提升 2–3 倍**，高并发峰值期最高 3.5 倍（NVIDIA 测算）。
- 案例：中等规模企业 AI 智能客服集群，KV Cache 显存瓶颈与驱逐导致的额外成本占单月总运营成本 **78%**；大型互联网企业因驱逐引发重复计算的算力成本超过硬件采购成本的 **5%**。
- 来源：腾讯云 FlexKV 新闻稿、DoNews 报道。

### 1.2 本仓库一手数据（可引用）

- **KV Cache 墙**（`09_inference_system/cost_analysis/llm_api_pricing_analysis.md` §3.1–3.4）：70B/8K 上下文单请求 KV 占 1.25GB（GQA 1/8 + FP8）~ 5.0GB（GQA 1/4 + BF16）；H100 80GB 装下 70GB 权重后仅剩 10GB，理论只支撑 **4–8 路并发**。
- **KV 密度速查**（`09_inference_system/memory_calc/memory_analysis.md` §4.5，BF16）：Qwen3-32B 约 256KB/token（128K 满上下文 ~32GB）；DeepSeek-R1（MLA）69KB/token；GLM-5 97.8KB/token；**DeepSeek-V4 仅约 9.4KB/token**（1M 上下文 9.62GiB，混合精度 4.3GiB）。
- **架构代际差**（`09_inference_system/post-kv-cache-era-challenges.md`）：标准 BF16 GQA8 在 1M token 时 KV 约 **250GB**，DeepSeek-V4-Flash 仅约 **5GB**（约 2%，与公开报道"V4 KV 仅需 5.48GB HBM"吻合，为部署/混合精度口径）；V4-Pro 的 KV 是 V3.2 的 10%。**口径注**：同文件 §1.2 另有 BF16 理论值 9.62GiB（`memory_analysis.md` §4.5）——5GB 与 9.62GiB 分别对应混合精度部署与 BF16 理论口径，引用时二选一，勿并列。
- **KV 是最大的显存消费者**（`09_inference_system/kv_cache/README.md`）：32K token prompt、70B、batch=8 时 KV 占 320GB 显存，是模型权重本身的 2 倍以上。
- **PagedAttention**（`09_inference_system/kv_cache/01_concepts/basic/kv_cache_basics.md`）：传统预分配内存利用率仅 20–40%，PagedAttention 将碎片率降到 4% 以下，同等显存服务 2–4 倍并发。
- **缓存 ROI**（`09_inference_system/kv_cache/01_concepts/capacity_planning/kv_cache_roi.md`）：KV Cache 分层后系统有效并发承载力提升近 2 倍（高命中场景 2.3–14 倍），单次 AI 调用算力成本摊薄 50% 以上；热请求 TTFT 缩短高达 75%（命中缓存可完全跳过 Prefill）。

### 1.3 三级降本技术（演讲用三句话讲完）

1. **分页**：PagedAttention——显存利用率 20–40% → 90%+，同等显存 2–4 倍并发。
2. **压缩**：架构改造（SWA/MLA/Hybrid）+ KV 量化——小米 MiMo-V2.5-Pro 70 层中仅 10 层 Full Attention、60 层 Sliding Window（128 token），**KV 存储降至全 Full Attention 的约 1/7**，Prefill 计算成本同步约 1/7；KV 量化（FP16→INT8/更低）再省 50–75%。
3. **复用**：前缀缓存 + 多级缓存——小米"窗口安全长度"规则将前缀缓存命中率推高至 **93%+**；腾讯云 FlexKV（GPU→CPU→SSD→远程云存储）将可用缓存容量扩展至 GPU 显存的 100 倍以上，实测 TTFT 降约 60%、TPOT 降 13%、QPM 提升 16%。

### 1.4 端到端降本效果

- **小米 MiMo-V2.5**：Hybrid SWA + KV 双池管理（容量效率约 7 倍）+ 前缀缓存重构 + GCache 分布式 SSD 缓存 + LLM-Router 亲和性调度 + 三层多 token 预测，API **最高降价 99% 且维持收支平衡**，缓存命中率 93–95%，单 GPU 并发用户提升至 5 倍。
- 通用手段：连续批处理 + PagedAttention 有效吞吐提升 2–5 倍；推测解码降低 20–40% 延迟；PD 分离分别优化计算密集与带宽密集阶段；Blackwell + FP4 可将 70B 模型推理成本降低 90–95%。
- 来源：新京报小米报道、DoNews 腾讯云 FlexKV。

## 二、利用率革命（调度与架构）

- **连续批处理（Continuous Batching）**：GPU 利用率 30% → 70–80%（行业常用口径；本仓库 `09_inference_system/prefill_decode/continuous_batching.md` 实测为：单请求 <1% → batch=64 约 31% → prefill batch=256 约 60–70%，引用时按实测值更稳妥）。
- **PD 分离**（Prefill-Decode 分离）：腾讯太极 16×H20 实测 15,800+ tokens/s，PD 分离端到端提升 30–40%（本仓库 `09_inference_system/deployment/deepseek_v3_h20_vllm_deep_dive.md`）。
- **硬件代际实测**（本仓库 `09_inference_system/vllm/hardware_optimization/deepseek_blackwell_wide_ep.md`）：GB200 vs H200 解码吞吐 2.2K → 4.6 倍提升。

## 三、硬件革命（每瓦 Token 的代际跃迁）

- **同一个 1GW 数据中心：Token 生成速率从 200 万跳到 7 亿，两年 350 倍**——同期摩尔定律约 1.5 倍（黄仁勋 GTC 2026 口径，方向可信、幅度存疑，引用时建议加诚实声明）。
- Grace Blackwell NVLink72 每瓦 Token 吞吐比 **Hopper 系（H100/H200）** 高 **35–50 倍**（黄仁勋先报 35 倍后承认故意保守，SemiAnalysis 实测约 50 倍）；Vera Rubin 官方口径为"每百万 token 成本降至 Blackwell 的 1/10"（"2–10 倍"为厂商口径，未检索到独立测试）。
- Rubin GPU 规格：3.6 ExaFLOPS、260TB/s 全对全带宽、288GB HBM；10 年算力增长 4000 万倍（黄仁勋口径；按两锚点自算约 2100 万倍，系 FP16/FP4 精度口径混用所致）。
- **分离式推理（Groq 协同）**：pre-fill/attention 交给 GPU（Rubin），decode 卸载给 Groq LP30（500MB 片上 SRAM、确定性数据流、静态编译）——延迟减半，最高价值层再提升 35 倍吞吐量。
- **液冷**：机架安装从两天缩短到两小时；PUE ≤1.2（GTC 口径）或 ≤1.1（白皮书口径），传统 IDC 1.5–2.0——引用时统一口径。
- 来源：投资界（GTC 2026 报道）、TokenFactory 白皮书。

## 四、价格历史：三年 214 倍的单价瀑布

### 4.1 里程碑时间线（输入 $/输出 $，每百万 token）

| 时间 | 模型 | 输入 $/M | 输出 $/M |
|---|---|---|---|
| 2023-03 | GPT-4 (8K) | $30 | $60 |
| 2023-11 | GPT-4 Turbo | $10 | $30 |
| 2024-05 | GPT-4o | $5 | $15 |
| 2024-05 | DeepSeek V2 | $0.14 | $0.28 |
| 2024-07 | GPT-4o mini | $0.15 | $0.60 |
| 2024-12 | DeepSeek V3 | $0.27 | $1.10 |
| 2025-01 | DeepSeek R1 | $0.55 | $2.19 |
| 2025-08 | GPT-5 | ~$1.25–2.50 | ~$10–15 |
| 2026 | GPT-5.5 / DeepSeek V4 | $3.00 / $0.14 | $15.00 / $0.28 |

### 4.2 关键降幅

- GPT-4o mini（2024-07）以 $0.15/M 输入提供 GPT-4 级性能，相比 GPT-4 发布价下降 **200 倍**，仅用时 16 个月。
- DeepSeek R1 相比 OpenAI o1-preview（$15/$60）便宜约 **97%**，引发全行业重新定价。
- 2026 年中前沿模型整体价格下降约 **50 倍**（Presenc AI）；价格指数从 2023-03 的 100 降至 2026-08 的 12（BenchLM，降 88%）。
- ~~参考任务成本（1000 入 + 500 出 token）从 GPT-4 32K 约 $0.09/任务降至 Gemini 2.0 Flash 约 $0.000085/任务——压缩约 1000 倍（Nesyona 数据集）~~ **【已弃用，勿引用】**：端点不可复现（$0.09 为混合口径；$0.000085 低于 Gemini 2.0 Flash 公开定价 $0.10/$0.40 约一个数量级）；且固定 token 量下任务降价倍数 = 输入/输出单价的加权平均，数学上不可能超过 214 倍。演讲统一口径：**单价 214 倍（可验证）+ 定性"模型变聪明、token 用得少，真实花费降得更多"**。
- 自托管开源模型（Llama 4）有效成本约 $0.05/M 输入；DeepInfra 在 Blackwell NVFP4 下达到 $0.05/M 的"近乎免费"水平。
- 下降并非单调：DeepSeek V3 相对 V2 涨价（输入 1.9 倍、输出 3.9 倍）；Claude Opus 定价 26 个月未变（$15/$75）——模型能力代际跃升时绝对价格可能回升，但"每美元智能"始终在改善。

### 4.3 结构性事实：输入/输出价差

> 口径注（开场体积锚）："写完一部 75 万字小说 400 元 → 2 元"是**纯输出价**口径（60 美元 → 0.28 美元 × 1M token）。真实的长文生成任务还会反复携带上下文输入（输入 token 同样计费），实际总成本高于纯输出估算——开场只用它做体积直觉，不承诺"写一本书的总花费"。

- GPT-3.5 时代 1:1 → GPT-5 已达 **1:8**——输出生成是自回归串行（快不起来），输入 Prefill 可并行。输出密集型任务（写作、代码）的成本下降幅度远小于输入密集型任务。
- 推理（reasoning）模型产生内部思维链 token，按输出价格计费，单个困难问题的有效成本可达数美元。

### 4.4 降价驱动因素（五条）

1. 硬件：H100 → H200 → B200，每美元吞吐提升 2.5–4 倍；
2. 软件：FlashAttention、PagedAttention（利用率 30% → 70–80%）、连续批处理、前缀缓存（3–10 倍）、投机解码；
3. 量化：FP8 成 2024 底默认精度，FP4 原生支持吞吐翻倍；
4. 架构：MoE（DeepSeek 671B 只激活 37B）有效计算成本降 2–4 倍；
5. 竞争：开源权重厂商（DeepSeek）设定价格下限，倒逼闭源跟进。

## 五、来源链接

- https://tech.ifeng.com/c/8tYPjWsdvAj（小米 MiMo 全链路优化技术细节）
- https://www.donews.com/news/detail/4/6512672.html（腾讯云 FlexKV 合入三大推理框架）
- https://m.bjnews.com.cn/detail/1780149584129681.html（小米：降价 99% 依然收支平衡）
- https://www.edgen.tech/zh/news/post/xiaomi-mimo-v25-cuts-inference-cost-99-with-kvcache-breakthrough（MiMo-V2.5 KV 突破）
- https://aws.amazon.com/cn/blogs/machine-learning/tiered-kv-cache-for-large-llms-on-amazon-sagemaker-hyperpod-with-curvine/（AWS 分级 KV Cache）
- https://nesyona.com/research/ai-token-price-decay-2026/（12 个前沿模型价格衰减追踪）
- https://tokencost.app/blog/ai-price-index（AI 价格指数：2023-2026 降 300 倍）
- https://benchlm.ai/llm-pricing-trends（2026-08 LLM API 定价趋势）
- https://presenc.ai/research/cheap-ai-token-pricing-cliff-2026（Token 定价悬崖）
- https://m.pedaily.cn/news/561782（GTC2026：每瓦 Token、Rubin、Groq 协同）
- https://deluair.com/consultancy/insights/ai-inference-economics-2026（2026 推理经济学）
