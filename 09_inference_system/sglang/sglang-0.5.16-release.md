# SGLang 0.5.16 解读：一棵 Radix Tree 统一四种 Attention，DSpark 让投机解码不再一刀切

> 2026 年 7 月 25 日，SGLang 发布 v0.5.16，574 个 PR 落地，169 位贡献者参与。
> 核心信号：当 SWA、Mamba、DSA、全注意力共享同一棵 Radix Tree，投机解码的逐请求动态调度和前沿模型的 Day-0 适配就有了统一的地基。
>
> 官方 Release：[v0.5.16](https://github.com/sgl-project/sglang/releases/tag/v0.5.16)

---

## 一、这不是"又多几个模型"的版本

SGLang 的版本迭代极快。过去两个月从 0.5.13 一路走到 0.5.16，节奏大致两周一个版本。0.5.13 铺开了 Spec V2 和一批新模型的支持，0.5.14 和 0.5.15 分别围绕 DeepSeek-V4 和 GLM-5.2 做生产级调优——GLM-5.2 在 0.5.15 上已经到了 500+ tok/s。

**0.5.16 不一样**：574 个 PR、169 位贡献者——从数字看，跟 0.5.13（约 700 PR）体量相当。但拉近看结构，差异是实质性的。169 个贡献者几乎是 0.5.13（~70 人）的 2.5 倍，而首次贡献者仅占 19%，相比之下 0.5.13 和 0.5.14 高达 99% 和 100%。推进 0.5.16 的不是大批新手小幅提交，而是老手与中生代的中等体量工作。12 项声明的破坏性变更——包括三个直接改名、不留兼容别名的参数——清理力度在近几个版本里前所未有。

但规模只是注脚。0.5.16 的 release notes 标了 8 个星——**UnifiedRadixTree 默认化**、**DSpark**、**Inkling**、**GLM-5.2 缓存层拆分（KV -74%）**、**ReplaySSM 投机显存缩减（-84%）**、**Blackwell 线性注意力**、**一批新模型**、**量化路径大清理**。

其中 **UnifiedRadixTree 是地基**。它让 SWA、Mamba、DSA 和全注意力四种 attention 类型共享同一棵 Radix Tree——前缀复用、状态同步、命中恢复从此不再各走一套路径。GLM-5.2 的 -74% KV 缩减和 ReplaySSM 的 -84% 投机显存缩减，都是在这层统一抽象之上才成立的；就连 DSpark 的逐请求动态验证窗口，也因为缓存命中时只需重置实际用到的状态（而不是整棵树），避免了动态调度引入额外的状态管理开销。

地基之上，0.5.16 最有杀伤力的两个可见特性是 **DSpark 置信度驱动投机解码**和 **Inkling Day-0 模型支持**。一个让投机解码从固定窗口走向逐请求动态预算，一个让 975B 的多模态 MoE 带着三种非标准架构组件发布当天就跑到了 171 tok/s。

574 个 PR 中大量属于工程维护：`sglang.kernels` 命名空间迁移、依赖升级（flashinfer 0.6.14、CuTe DSL 4.6.0、sgl-kernel 0.4.5）、量化路径清理。本文沿"地基 → 后果"的线索展开：第五章讲 UnifiedRadixTree 如何统一四种 attention；第二至四章讲地基之上 DSpark 和 Inkling 各自做了什么；第六章速览其余重要变更；最后两章给出升级指引。

---

## 二、投机解码的隐秘天花板

在展开 DSpark 之前，有必要先理解投机解码为什么在高并发下会碰到天花板。

投机解码的逻辑很直观：用一个小成本的动作（草稿模型预测几个 token），换取减少目标模型 decode 步数的可能。如果草稿模型的接受率足够高，每步解码就能推进不止一个 token，等效吞吐随之提升。这条路线已经有 EAGLE、MTP 等成熟方案集成进了 SGLang。

但这个逻辑有一个隐含的前提：验证的成本是可控的。

在高并发场景下，这种假设开始松动。批量大小为 B、推测 K 个 token 时，目标模型每步要验证 B×K 个 token。当 B 达到数十上百、K 取 4–8，每步验证的 token 数就已逼近一次小型 prefill 的规模——更糟的是，草稿尾部 token 的接受率远低于块内靠前的位置，这些大概率不被接受的 token 依然占用了同样多的验证算力。

固定验证窗口对所有请求一视同仁——无论这个请求的内容是数学推理（草稿模型大概率猜对），还是自由格式的诗歌生成（草稿模型大概率猜错），验证窗口都一样大。这意味着**坏请求的尾部在浪费所有请求的验证预算**。

这不是一个实现优化不够的问题，而是固定窗口策略本身的结构性问题。DSpark 的出发点正是在此：让每个请求在每个解码步，验证"刚好值得验证那么多"token。

---

## 三、DSpark：置信度驱动的可变长度验证

DSpark 的核心思想可以用一句话概括：**由草稿模型自身的置信度决定验证多长，停止为不太可能被接受的 token 浪费验证算力。**

它不是简单地给草稿加一个"好不好"的标签，而是给每个草稿 token 一个存活概率，然后让调度器在线决策每个请求的验证窗口。

### 3.1 三个草稿侧的组件

DSpark 在草稿侧有三个组件，分别解决三个独立的问题：

**半自回归块草稿器（Block Drafter）** ——一次前向生成整个 γ-token 块。区别于逐 token 自回归生成，块草稿器通过轻量级顺序头（稠密模型用 Markov 头，稀疏模型用 RNN 头）让每一步条件化于前一个 token，从而在一次前向中保持较高的块内接受率。

**置信度头（Confidence Head）** ——为每个草稿 token 评估其通过目标模型验证的概率。它不是判断 token 本身是否正确（那需要目标模型来验证），而是评估"草稿器自己对这个 token 有多大把握"。这是整个 DSpark 的信息源头——后续的调度决策全部建立在这个信号之上。

**顺序温度缩放（STS，Sequential Temperature Scaling）** ——置信度头输出的原始分数并不天然等于真实的接受概率，存在系统性的校准偏差。STS 负责校准这些分数，使它们反映调度器预算所依据的真实接受率。这是一个纯后处理步骤，不改变草稿器的推理逻辑。

### 3.2 Ragged Verify：不等长验证的工程落地

有了每 token 的置信度，调度器可以为每个请求独立决策验证多长。但这里有一个工程挑战：不同请求的验证长度不同，怎么放进固定形状的 CUDA graph？

答案是**按总 token 数索引 graph，前端打包（front-pack）进紧凑缓冲区**。

具体来说：不同请求的验证窗口大小不同，需要的 token 总数也不同。SGLang 根据总 token 数向上取整到最近的已捕获 graph 层级（tier），将不等长的请求前端打包进一个紧凑的 `cu_seqlens` 风格缓冲区，回放对应的 graph。验证预算被裁剪时，总 token 数降到更小的层级，回放真正更便宜的 graph——不是打 mask 的全宽前向，而是更少的 attention 和 MLP 行。

一个直观的对比：固定形状 graph 把每个请求填充到全块宽（18 格中 8 格为 padding）；ragged 紧凑 graph 只需把总数向上取整（12 格中 2 格为 padding）。**ragged 算的无效格远少于固定形状。**

更关键的是，ragged verify 不需要新的 attention 内核。后端已有的 varlen 路径（比如 DeepSeek-V4 上的 `flash_mla` 稀疏 MLA 路径）直接可用，每个后端在 graph 回放时重建 varlen 元数据即可。

### 3.3 零开销调度：将决策转化为墙钟时间

算法上的"动态调度"要变成真正的吞吐提升，需要把调度器的开销压缩到零。

SGLang 的 overlap 调度器（详见 [SGLang Overlap Scheduling 深度解析](sglang-overlap-scheduling.md)）已用独立流将下一步调度与当前前向重叠。DSpark 几乎无需特殊处理即可接入这一机制——置信度中继走同一通道，回读延后两步。decode 循环无每步气泡，比关闭调度器时紧凑约 1.5 倍。

辅助开销的压缩同样激进：compact scatter、SWA page-index、验证长度 top-k 调度、ragged 窗口打包等小操作集群被重写为融合 Triton 内核；块草稿器的采样路径折叠进融合内核，其矩阵乘法被分片。在一个示例 profile 中，目标验证之外的开销从约 1.7 ms 压缩到近乎不可见——作为对比，验证本身耗时约 7.3 ms。如果不做压缩，调度开销约占总步耗时的 20%。

### 3.4 性能数据：在什么地方赢

DeepSeek-V4-Pro、B300 TP8、batch size 1、平均接受长度约 5 的条件下，DSpark 达到 **383.7 tok/s**。这个数字不能线性外推——它是特定硬件、特定模型、单请求下的测量，团队成员在博客中反复强调"所有加速都是对自身对照的测量，复现的是机制和曲线而非论文逐位数字"。

更有信息量的是曲线对比。H200 DP4 上，DeepSeek-V4-Flash 在 batch 1–256 的全范围内，DSpark 的吞吐/延迟曲线始终优于 MTP 和非推测基线。**差距在并发增长、吞吐开始平台化后才明显拉开**——batch size = 1 时两者持平，因为小批量的目标验证并不因更多 token 而明显变慢；当吞吐逼近硬件天花板时，裁剪冗余 token 的价值才真正体现出来。

混合流量下更能看出动态调度的意义。gsm8k（高接受率）、arena-hard（中接受率）、poetry（低接受率）三种负载混合时，调度器按难度分配截然不同的窗口：5.24、3.78、2.91 token。约 55% 的数学推理步骤填满 6 token 全窗口，而约 80% 的诗歌步骤只用 3 个或更少。利用率保持在 0.88–0.97。**调度器在按请求差异化分配，而不是取一个批量平均。**

### 3.5 可观测性：裁剪的上限

动态调度引入了一个新问题——**裁剪会审查上限**。compact 模式只验证窗口内的 token，完整块验证本来会接受多少 token 从未被观测到。没有这个上限信息，就无法区分"好的裁剪"和"有损的裁剪"。

DSpark 提供两种途径恢复这个信息：

- **cap-accept 模式**：验证完整块，但只提交窗口内的结果。提交内容与 compact 完全相同，同时暴露上限——"这步如果没裁剪，本来能再接受多少 token"。
- **块接受估计器**：利用未来步骤中目标 token 的 logprobs，在 compact 运行内部直接恢复估计的审查后上限，为反事实尾部计算估计区间。不需要额外运行。

这两个工具的价值在于，它们让 DSpark 的调度决策从"黑盒加速"变成"可审计的调度"——这在生产部署中不是锦上添花，而是确定性的基本要求。

---

## 四、Inkling Day-0 支持：非标准架构的适配极限

DSpark 的价值在于它可能直接进入你的生产环境——如果你的负载特征是混合难度、高并发，动态置信度调度有实际的吞吐收益。Inkling 的价值不一样：你大概率不会跑 Inkling 本身，但它证明了 SGLang 引擎对非标准模型架构的适配能力已经到了什么程度。

Inkling 是 Thinking Machines 推出的 975B 多模态 MoE，上下文窗口 1M token。它对标准 decoder-only Transformer 做了三处非标准修改——ShortConv 短卷积、相对位置嵌入注意力、共享专家 Sink MoE。这三处每一处都不是"换个 config 就行"的那种差异，而是需要引擎侧写专项内核的架构级改动。**发布当天 SGLang 就让它跑到了 171 tok/s 的解码速度。** 这个信号的意义超过了模型本身：如果你在构建或微调一个带有自定义 attention、特殊卷积或非标准 MoE 的模型，SGLang 的引擎抽象层已经足够灵活，不会让你的模型等引擎跟上来。

### 4.1 三处架构修改

Inkling 对标准 decoder-only Transformer 做了三处修改，每一处都对应一组 SGLang 侧的工程决策：

**ShortConv（短卷积）**：W=4 的逐通道因果卷积，每层出现在四处——K 流、V 流、注意力输出、MLP/MoE 输出。这个组件看似轻量（窗口才 4），但在 BCG/PCG 模式下每层 4 处站点都会回退到 Python/Triton 调用栈，单层产生 4 个 CUDA bubble。SGLang 的对策是**全前向 CUDA 图捕获**——将包含 ShortConv 在内的整个前向捕获为单图，消除所有 eager 回退气泡。在 launch-bound 形状下，这比 BCG 快 14–17%。

卷积的计算位置也很关键。ShortConv 位于 tensor-parallel all-reduce 之后，SGLang 直接将它融合进自定义 all-reduce 内核——自定义 all-reduce 比 torch 的 `multimem_all_reduce_` 快 2.1×，融合 ShortConv 后进一步快 2.08–3.60×，端到端吞吐提升 5–8%。

**相对位置嵌入注意力**：学习式逐头相对位置偏置直接加到注意力 logits，替代标准位置嵌入。布局为 5 层滑动窗口 + 1 层全注意力交替（`layer_id mod 6 == 5`），混合了滑动窗口、全注意力和 Mamba2 线性注意力。SGLang 做了一个剪切偏置内核（sheared-bias kernel）——相对 logits 被预剪切为列对齐的偏置张量，内核通过普通 tile load 加偏置，而不是每分数单独索引。这是 shipping path，但需要 Inkling 专用的 FA4 fork。

**共享专家 Sink MoE**：路由和共享专家在同一权重预算内共同归一化，而非传统"始终开启"的独立共享路径。SGLang 用一个融合 top-k 内核将 sigmoid+bias、top-k、renorm 折叠为单次 Triton 遍历，比未融合链路快 1.6–5.6×；一个形状特化的 CUDA-JIT 版本在 T=4096 下达到 7.72 μs vs 26.15 μs（3.4×），T=16384 下达到 20.09 μs vs 52.88 μs（2.6×）。共享专家的融合走"线性化布局"——gate/up 权重沿输出维度堆叠，down 权重沿规约维度拼接，专家求和发生在 down GEMM 的规约内部，消除了复制输入、逐专家中间结果和独立求和步骤。

### 4.2 投机解码双轨

Inkling 自带 8 层链式 MTP（多 token 预测），SGLang 将整条链——8 次草稿前向 + token 旋转 + 注意力元数据 + 采样——捕获为单个 CUDA 图，全程无设备同步。支持温度 > 0 时的分布精确拒绝采样（以概率 min(1, p/q) 接受，拒绝时从残差重采样）。

同时，SGLang 与 Modal 合作训练了 DFlash 独立草稿模型：单次前向填充整块掩码未来位置，相比原生 MTP 吞吐提升 67%。

### 4.3 性能与覆盖

Blackwell B200 上：输入吞吐最高 **71.7k tok/s**，每用户解码 **171.0 tok/s**（TP8, bs=1）。功能覆盖完整的 PD 分离、NVIDIA bf16 + NVFP4、AMD bf16、多 LoRA 服务（从 1 个 LoRA 到 4 个 LoRA 多路复用仅损失 0.9% 吞吐），以及 HiCache。

---

## 五、一棵树，四种 Attention

前面三章分别讲了 DSpark 和 Inkling——地基之上最可见的两个成果。现在回到地基本身。

**UnifiedRadixTree 在 0.5.16 成为 SWA、Mamba、DSA 和全注意力四种 attention 类型的默认缓存路径。** 这不是性能优化，而是架构简化。在此之前，每种 attention 类型维护自己的缓存树——前缀复用、状态同步、命中恢复各有一套实现。UnifiedRadixTree 把四套路径统一为一套，减少了混合架构模型在缓存管理上的路径分叉。

这项工作的收益不是一次性的吞吐提升。它体现在两个层面：一是以它为前置条件的子特性立刻获得了可观的资源节约；二是此后每一个依赖缓存状态的新功能，都不需要为每种 attention 类型单独实现一遍。

两个以 UnifiedRadixTree 为前提的子特性值得注意：

**GLM-5.2 DSA 缓存层拆分**：在启用 prefill context parallelism 时，KV 和 indexer 缓存层按 CP rank 分片，每个 rank 只拥有不重叠的层范围。8192 token、78 层、cp_size=4 条件下，每 rank KV 从 0.77GB 降至 0.20GB（约 -74%）。启用方式：`--enable-dsa-cache-layer-split`，需配合 `--enable-prefill-cp --cp-strategy interleave`。

**ReplaySSM Ring Spec-Verify**：去掉 GDN 投机解码中逐草稿的 SSM 快照保存。Qwen3.5-35B-A3B TP1 上，投机 scratch 从 11.5GB 降至 1.8GB（缩小 6.4 倍），精度和吞吐持平。默认关闭，通过 `--enable-gdn-replayssm-spec` 启用，仅限线性草稿链（`--speculative-eagle-topk` 为 {None, 1}）。

---

## 六、其他值得关注的变化

**CPU 上首次支持投机解码**：DSpark 及相关投机解码能力扩展到了 CPU 环境（#27862），这是将推理优化从 GPU 专有走向异构的重要一步。

**Elastic EP 运行时扩容**：专家并行支持在推理运行时热扩展（#30164），不需要重启服务即可增加 EP 规模。

**gRPC 分离式部署原生支持**：gRPC 分离式生成请求（#30440）+ 原生 gRPC server launcher（#23508）+ gRPC sidecar 模块 launcher（#31076），标志着 SGLang 在 PD 分离架构上从内部实验走向标准化接口。

**多模态统一特征传输**：VLM 统一多模态特征传输（#30904）+ 批量跨请求 ViT 编码并复用 attention 元数据（#24013），减少图像预处理在高并发下的开销。配合原生 Rust 扩展实现的 GIL-free 预处理流水线（哈希计算移出调度器热路径），图像密集型负载的 TTFT 降低 14–44%，吞吐提升约 20%。

**量化路径清理**：QServe W4A8、FBGEMM FP8、CUTLASS FP8 blockwise（SM90/SM100）、树内 NVFP4 JIT kernel 一次性移除。NVFP4 GEMM 统一依赖 FlashInfer。这是一次大规模的实验性路径清理——这些路径的维护成本已经超过了它们的使用价值。

**新增模型**：Inkling 之外，LongCat 2.0 FP8、JetBrains Mellum v2、Pi0.5（VLA，视觉-语言-动作模型）、LongLive 2.0（扩散模型）也加入支持列表。MiniMax-M3 完成了代码接入，但官方标记为"开发镜像阶段"——"代码在仓库里"与"生产可用"之间仍有距离。

**Blackwell (SM100) 线性注意力**：首个正确的 KDA MTP 路径落地。`recurrent_kda` 解码 kernel 达到 29.6 μs vs Triton 的 36.8 μs（B=64），B=256 时完整解码路径达 1.35×。小 batch 下更慢，但这是正确性优先于性能的路线——先保证对的，再追求快的。

---

## 七、升级前必须确认的破坏性变更

0.5.16 的破坏性变更列表有 12 项，其中几项直接让旧启动命令报错：

| 变更                                                             | 影响程度             | 说明                                              |
| ---------------------------------------------------------------- | -------------------- | ------------------------------------------------- |
| `--enable-deepep-waterfill` → `--enable-waterfill`               | **高危**             | 无兼容别名，旧命令直接报 `unrecognized arguments` |
| `--optimistic-prefill-retries` → `--optimistic-prefill-attempts` | **高危**             | 同上，无废弃别名                                  |
| `num_tokens_per_bs` → `num_tokens_per_req`                       | **高危**             | 参数重命名，配置文件和脚本需更新                  |
| QServe/FBGEMM FP8 量化路径移除                                   | **中危**             | 使用这些量化路径的需迁移                          |
| `--fp4-gemm-backend cutlass` 移除                                | **中危**             | NVFP4 GEMM 强制依赖 FlashInfer                    |
| UnifiedRadixTree 成为 SWA/Mamba/DSA 默认                         | **中危**             | 行为变化，缓存命中逻辑改变                        |
| Chunked input-logprob 默认开启                                   | **中危**             | 内存行为变化                                      |
| FA3 sparse mask kernel 默认关闭                                  | **低危**             | 需要此功能的需手动开启                            |
| 旧 Sphinx `docs/` 移除                                           | **低危**             | 文档迁移到 Mintlify                               |
| `sglang.kernels` 命名空间                                        | **低危**             | 仅 import 路径变化，内核本身不变                  |
| 扩散 rollout 端点改为 MsgPack                                    | **高危（扩散用户）** | 服务端和训练端必须同步升级                        |

特别提醒：**参数重命名没有兼容别名**是 SGLang 此次最激进的变更。如果你在启动脚本或配置管理系统中硬编码了这些参数名，升级后会直接启动失败而非降级警告。建议在测试环境先用 `--help` 扫一遍所有参数名再做生产升级。

---

## 八、已知问题与 DSpark 的当前局限

0.5.16 标注了 5 个已知问题，以下是其中与生产环境相关的三项：

- **温度 0 的不确定性**：DeepSeek-V4-Flash FP4 + DP attention + breakable prefill CUDA graph 组合下，相同 temperature=0 请求可能产生不同结果。根因是 breakable prefill CUDA graph（#30898）引入的 idle-rank dummy extend 扰动真实请求 logits。不启用 breakable prefill CUDA graph 可规避，但这意味着放弃一部分吞吐。
- **Mamba overlap scheduler 的 seqlen 问题被回退**：修复（#31369）后又被回滚（#31622），底层问题仍未解决。
- **flashinfer 锁定 0.6.14**：0.6.15 升级被回滚。

对于 DSpark 本身，有几个当前阶段的局限值得关注：

**成本模型是第一版近似**。调度器用于决策的步时间估计采用加性模型 `T(bs, K) = bias + alpha(bs) + theta(M)`，其中 `theta(M)` 是唯一能被裁剪回收的项。这个模型假设验证成本与总 token 数线性相关——在大多数场景下是合理的近似，但上下文长度对单步成本的影响尚未被建模。更长的序列意味着更贵的 attention，而当前模型没有区分"验证 4 个 token 的长序列"和"验证 8 个 token 的短序列"之间的单步成本差异。

**收益高度依赖任务类型**。低接受率的负载上（如诗歌生成），裁剪收益大且更早出现；高接受率负载上（如数学推理），块的大部分都被接受，调度空间有限。DSpark 不是"对所有场景加速 X%"，而是"在高并发且接受率异质性强时提供比固定窗口更好的调度"。

**校准质量决定调度质量**。如果置信度头的校准不准——比如系统性地高估低接受率 token 的存活概率——调度器就会分配过大的窗口，白白浪费验证算力。STS 负责校准，但它是离线拟合的，在分布偏移（模型更新、数据漂移）下是否会退化，目前缺少长期生产环境的数据。

**作者明确将其定位为"可用的初版"**。博客中写道"the exact operating point the scheduler lands on is likely improvable"，issue #30344 列出的未来计划包括更强更自适应的在线成本模型、更多模型覆盖、更多并行模式。这些计划本身说明当前版本是机制验证，而不是最终调优。

---

## 九、总结

SGLang 0.5.16 不是一个"又多了几个模型"的版本。它的实质性变化可以归纳为三个方向：

1. **调度范式的位移**：DSpark 让投机解码从"一批一策"走向"逐请求动态预算"，ragged verify + CUDA graph + 零开销调度让这个决策链条不增额外开销。这是推理引擎在资源分配粒度上的一次实质性细化。
2. **模型适配的纵深**：Inkling 带着 ShortConv、混合注意力、共享专家 Sink 三种非标准组件，发布当天就在 SGLang 上跑到了 171 tok/s 的解码速度。这个能力的背后是全前向 CUDA 图捕获、自定义融合 all-reduce、相对位置嵌入专用内核、共享专家线性化布局等一系列专项优化。
3. **基础设施的收敛**：UnifiedRadixTree 默认化、量化路径的大规模清理、sglang.kernels 命名空间迁移——这些不产生新功能，但降低了后续迭代的复杂度。

对于准备升级的团队，建议的优先级是：（1）先在测试环境扫一遍参数名变更，确保启动不报错；（2）如果你的负载特征是高并发 + 请求难度差异大，DSpark 值得基准测试——从 compact 模式 + SPS 成本表开始；（3）如果使用量化路径，确认没有依赖已移除的 QServe / FBGEMM FP8 / CUTLASS FP8 blockwise。

素材来源：

- [SGLang v0.5.16 Release Notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.16)
- [DSpark in SGLang: Speculative Decoding with Confidence-Driven, Variable-Length Verification](https://www.lmsys.org/blog/2026-07-06-dspark-sglang)
- [SGLang and Miles Add Day-0 Support for Inkling, a Frontier Multimodal Model](https://www.lmsys.org/blog/2026-07-15-inkling-day0-support/)
- [574 个 PR 落地，SGLang 0.5.16 用置信度调度推测解码](https://www.atyun.com/81784.html)（ATYUN 中文报道）
