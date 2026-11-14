# 05 · Ave：用数据流不变量指导 Agentic GPU 优化

> SOSP 2026 论文精读笔记。来源：`../papers/05_ave-agentic-gpu-optimization.pdf`（arXiv:2604.18616 预印本，2026-11-14 下载）。文中数字均为论文口径。

## 一句话

在 tile-based GPU kernel DSL 里引入编译期可验证的数据流不变量，把侵入式优化的试错从「跑挂了才知道」变成一次带反例的断言失败。

## 问题与背景

论文的出发点很具体：GEMM kernel 把矩阵 A、B 分块搬进 shared memory，再交给 MFMA 矩阵指令；MFMA 要求操作数满足硬件特定的 swizzled 布局，布局变换做错，乘的就是配错对的元素（§1，p.2）。flash attention 更极端，MI300X 上要同时做操作数重排、shared memory 复用、软件流水、wavefront 分组与指令调度，论文称这些数据搬运与布局变换是 error-prone 的（§2，p.3）。

LLM agent 的现状是能写出功能正确的 kernel，性能追不上专家库：GPT-5.6 Sol 在 AMD MI300X 上，现有 agentic 系统产出的 GEMM 比 hipBLASLt 慢 1.8–14.3×（§1，p.1）。痛点有二：其一，验证这类底层变换要在汇编层面推理 memory aliasing、data race 等性质，消耗 token 并撑爆上下文；其二，单测和 profile 只给稀疏的端到端信号，能说 kernel 错了或慢了，定位不了是哪个变换弄坏的（§1，p.1-2）。ROCm Triton 能把差距收到约 2×，但不暴露指令调度与 wavefront specialization 的细粒度控制（§1，p.2）。Ave 把 information-flow security 里 taint tracking 的思路搬到 GPU 正确性上：标签当污点，断言当策略，编译器当监控器。

## 核心设计

整体三件套（§3，p.3-4）：带 tag function 与断言的 tile-based Pythonic DSL、零运行时开销的编译期不变量分析、ICRL 驱动的 agentic harness。

**Tag functions 与 tag assertions（§4，p.5-6）**。DSL 受 CuTe 与 TileLang 影响，layout 函数把多维逻辑坐标映射到一维物理偏移，支持 composition、logical division、logical projection，用来表达 tensor core 的 swizzled 布局。Tag function 给 tensor 元素挂符号标签（坐标元组或符号表达式），标签随赋值传播，如 flash attention 例子里 `T_Q = Q[sq,h_q,d] → (h_q/gqa, d)`、`T_K → (h_v, d)`（Figure 1，p.4）。断言分两类：conformity 要求使用点标签一致，`assert tag(tK[...]) == tag(tQ[...])` 保证 S = QK^T 的操作数属同一 head 且 reduction 维对齐；non-conformity 要求标签不同，`assert tag(tV[...]) != tag(rS[...])` 保证并发的 V 生产者不覆写同一段 shared memory。一套机制同时压住正确性 bug 与并发冲突。

**编译期验证（§5，p.6）**。编译器把 DSL 降到 MLIR，做 flow-sensitive、path-insensitive 的标签传播，内存按字节粒度追踪，16-byte 向量化 load 跨宽度传播标签。控制流汇合处在格 `⊥ < t < ⊤` 上 merge：常量携带 ⊥，`merge(⊥, t) = t`，所以 OOB guard 的零值分支不污染真实标签；shared memory 标签可在流水线级间重置为 ⊥，允许 buffer 复用。标签与断言编码为 SMT 约束交 Z3 求解，path-insensitive 合并避免枚举分支条件。失败时返回 concrete witness：违规抽象状态下持错元素的具体线程 ID、数据元素与程序点，如 `attn:26: tag(tK[0]) ≠ tag(tQ[0]), Witness: i=0 ⊢ tid=0, ...`（§6，p.7）。全程编译期完成，注意力 kernel 单次断言检查 10–40 ms（§9.1，p.12）。边界：不建模 full heap、不追踪写入 global memory 的标签，MoE expert 排序与多 GPU buffer 上的不变量表达不了（§5，p.6）。

**优化循环与奖励（§6，p.6-7）**。每轮：planner 观察当前最优 kernel、检索到的知识与反馈历史，输出带前置条件与不变量模板的排序 proposal；selector 排出兼容计划；lowering agent 逐条实施并把模板实例化为断言；validation 编译、查断言、跑单测，通过才 profile，只有实测变快的 kernel 替换当前最优。奖励四项相加 `R_t = C_t + I_t + U_t + Δ_t`：编译成功、不变量得分 `I_t = 1 − |V_t|/|A_t|`、单测通过、运行时提速，验证过的减速得负 Δ_t。ICRL 用 TextGrad 式文本梯度更新 benchmark-local 的 planner prompt，模型权重与知识库都不动（Algorithm 1，p.7）。论文 walkthrough 了一个完整回合：planner 要求把 QK^T 折进 32×32×16 累加，lowering agent 摆错 tK 的 layout，断言失败给出 witness，三个 LLM 阶段生成 critique 改 prompt，下一轮 planner 给出正确 binding 并通过全部检查（§6，p.7-8）。

**知识库与实现（§6，p.6；§7，p.8）**。知识库为人工策展：硬件事实、DSL 参考、优化 recipe，recipe 声明适用条件、变换与不变量模板，来源含 KernelWiki 式文档、vendor 示例、公开 bug 报告与 microbenchmark。实现约 22,000 行 C++ 加 7,000 行 Python，DSL 经 MLIR 方言降到 LLVM 20 AMDGPU 后端，扩展了 barrier 调度、寄存器类选择与 buffer load/MFMA 指令选择；harness runtime 用 Codex。

## 效果与代价

Workload（§9，p.10）：MI300X 上 GEMM、GQA flash attention（LLaMA-3 70B prefill 八路 TP 配置）、DeepSeek-V3.2 FP8 fused MoE 各五配置，从 naive 实现起步。GPT-5.6 Sol 口径（Table 1，p.9）：GEMM Ave 达 hipBLASLt 的 0.89×，agentic 侧最强的 KernelFalcon 只有 0.55；attention Ave 达 AITER 的 0.99×，即匹配手写汇编，KernelFalcon 0.33、KSearch 0.15、CUDAForge 0.01；fused MoE Ave 达 AITER 的 0.94×，相对 KernelFalcon 8×、相对 CUDAForge 51×。对 uncontaminated agentic baseline 的几何平均吞吐提升 1.62–1176×（§1，p.1）。成本：GEMM 单 workload 完整优化 $6.49、12 turns、0.70M tokens；Atrex 同 workload $146.26、327 turns、61M tokens（Table 1，p.9）。

其余证据：H200 上对 cuBLASLt、FlashAttention-3、DeepGEMM 分别 0.98×/1.00×/0.99×（§9.3，p.12）；Hopper 与 CDNA4 后端初始移植约 1 人周（§8，p.9）。KernelBench 带示例提示，GPT-5.6 Sol 三次尝试内通过 100/100（L1）与 88/100（L2）（§1，p.2；Table 3，p.12）。不变量消融（§9.4，Table 4，p.12）：在要求 MFMA、软件流水、HW OOB-guarded load 三变换全部生效的 60 道任务上，六个数据集-模型组合成功率全部提升，如 L1-GPT 8/60 升至 18/60。harness 消融（Figure 3b，p.11）：GQA attention，只靠运行时反馈 190 TFLOPS，知识库无 ICRL 255，知识库加 ICRL 去不变量 194，全系统 388；五级优化合计 2.7–3.6× Naive，wavefront 流水级贡献最大增量（几何平均 +100%）。

论文承认的局限（§8，p.8-9）：不变量只验证具体变换，不产生算法级改动（如 tiled online softmax 重构），算法选择与初始 kernel 须先行；不建模 heap 与 global memory 写，MoE 路由、多 GPU 通信的不变量表达不了；FP4 nibble 置换不可见；要求 workload 维度编译期已知；对 reward hacking，人工 inspection 与不变量都不是穷尽的。

## 总评

这套设计回答的问题很具体：agent 做跨 kernel 的侵入式变换时，单测和 profile 指不出哪一步改坏了语义，Ave 把定位成本压到读一条带线程号与程序点的 witness。最有说服力的证据是 harness 消融：同样的知识库、同样的 ICRL，只去掉不变量，388 TFLOPS 掉到 194，甚至低于知识库无 ICRL 的 255；增益主要来自变换正确性可定位，planner 排序只是辅助。Table 4 里所有组合成功率上升、turns 与 tokens 一般下降，也支持同一结论。

折扣清单：知识库与不变量模板的人工成本只在架构移植处给了 1 人周，覆盖一类新变换要写多少模板没有量化；R_t 四项等权相加，无权重消融；generation 阶段的人工审查仍标记出 L2-DSv4 13 例 PyTorch fallback 或输入特化作弊，不变量词汇表之外的作弊要靠人工兜底（§9.3，p.12）；benchmark-local prompt 意味着每个 workload 单独积累 planner 状态，跨任务迁移收益未测。

对做 AI 编码或自动调优系统的读者，「领域不变量当过程奖励」是可迁移的：任何约束可写成机器可检查断言的领域（并发、分布式协议、数值 kernel），都能让编译器给 agent 返回反例而非一个标量。与 TVM/Ansor 一系是互补：编译器自动调优在固定调度原语上搜参数，Ave 把调度原语本身交给 LLM，再用不变量兜正确性；论文自己也承认，若编译器自动化吸收更多底层决策，这套机制的适用面会收缩（§8，p.9）。

## 疑点与待核

1. Figure 3(b) 消融 (iii) 的正文因果链没读顺：该配置已移除不变量，论文却把 invariant check 诊断修复与 V^T layout「mispairs reduction coordinates」写进同一句（p.11）；194 TFLOPS 与「性能停在 Naive 附近」两个结果清晰，机制表述存疑。
2. 摘要「attempts 100% of Level 1 and 88% of Level 2」是 GPT-5.6 Sol 的 Pass@3 口径（Table 3）；DSv4 与 DSv4f 的 L2 Pass@3 只有 86 与 84，摘要以最强模型为准，易高估跨模型稳定性。
3. Table 1 中 Atrex 只跑 5 个 trial（其余框架 10 个），理由是 API 成本；失败后不恢复重试。预算不对称对 baseline 结论的影响论文未讨论。
4. 论文把 Atrex 检索库含 AITER 参考实现标注为 contaminated（footnote 1，p.11），但 Ave 自身知识库策展自 KernelWiki 式文档与 vendor 示例，与归一化基准 hipBLASLt/AITER 的重叠度未披露。
5. SMT 求解只有注意力 10–40 ms 一个时延数，更大 kernel 上的退化边界（超时率、失败率）无规模曲线。
