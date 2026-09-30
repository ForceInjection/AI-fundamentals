# Ascend 950 微架构速查：来自 FlashMLA 算子报告的硬件信息整理

> 2026-09-30 | 信息整理。全部数值来自单一来源：deepseek-ai/FlashMLA 仓库官方文档《Ascend Sparse Attention Forward 算子与优化技术简析》（深度求索著，MIT License，本站存档为 [《Ascend950 稀疏注意力 Forward 算子与优化技术简析》](Ascend950_稀疏注意力Forward算子与优化技术简析.md)，下称「源文档」），逐条标注出处章节。口径是**算子视角的微架构**，不是官方 data sheet；性能数字为厂商自报，无第三方复现。另有三处补充：DeepGEMM-Ascend 与 DeepEP-Ascend 仓库 README（缩放因子格式、卡级型号），以及 2026-09-30 对 FlashMLA 仓库的代码深读（B_TOPK 最终取值、MTE2 在飞数、VLOOP 检出情况，随条标注 file:line）。

## 一、AI Core：1 Cube + 2 Vector

| 项                        | 数值                                                                                            | 源文档出处                                                              |
| ------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| AI Core 构成              | **1 CUBE Core + 2 VECTOR Core**（本算子实际采用 1C2V，1C1V 被三个瓶颈否决，见第四节）           | 〈能否用 1C1V〉                                                         |
| CUBE Core 吞吐            | **4096 FMA/cycle**（打满口径）；关闭 HF32 mode 时降至 256 FMA/cycle                             | 〈每个 AI Core 负责多少个 q head〉〈FA 的 output accumulator 在哪里做〉 |
| MMAD 特性                 | M 维取 64 即可打满 CUBE 吞吐                                                                    | 〈每个 AI Core 负责多少个 q head〉                                      |
| L0C 容量                  | 约 **256 KB**（128×512 float32 即占满；只能被 FixPipe 搬出或被 CUBE 用作累加器，无法原地乘标量） | 〈FA 的 output accumulator 在哪里做〉                                   |
| 核数 / 频率（该分析场景） | 32 个 AI Core，主频 1.65 GHz                                                                    | 〈每个 AI Core 负责多少个 q head〉的存算比脚本                          |

## 二、存储层次：L1 双 bank、L2 5 TB/s

| 层级   | 数值 / 特性                                                                                           | 源文档出处                         |
| ------ | ----------------------------------------------------------------------------------------------------- | ---------------------------------- |
| L1     | **512 KB，分两个 bank**（0–256K / 256K–512K 两段地址空间）；Tensor 切两份配平摆放可打满双 bank 带宽   | 〈L1 bank 配平〉                   |
| L2     | 总带宽 **5 TB/s**（32 核分摊）——注意是片上 L2，不是 HBM                                               | 〈每个 AI Core 负责多少个 q head〉 |
| UB     | 有 bank conflict 问题（NZ buffer 额外 pad 一行规避）；读写带宽可被单个 SIMD VF 打满                   | 〈高速 ND2NZ〉                     |
| icache | CUBE **32 KB** / VECTOR **16 KB** / SIMD VF **8 KB**——代码体积需控制（报告建议用 `VLOOP` 硬件循环、慎用内联；代码深读未在源码检出该符号） | 〈icache miss 问题〉               |

## 三、搬运与流水约束

- **MTE2 在飞请求上限 16 个**（单 VECTOR Core 的 Issue Queue 深度）——稀疏 gather 逐 token 发请求会爆队列，催生 gather2 技巧（`burst_count=2`、`src_stride` 设两 token 地址差，一次拷两个 token）。（〈gather2〉〈能否用 1C1V〉）
- **FixPipe 向外搬运 128 Byte/cycle/AI Core**——1C1V 被否的三个瓶颈之一；1C1V 口径下由此推出 `B_TOPK > 73.14`。最终 1C2V 代码实际取 **`B_TOPK = 64`**（`config.h:55`），每 AIV 32 token、gather2 折半后恰好 **16 个在飞 MTE2 请求**——与 16 的队列深度持平。（〈能否用 1C1V〉；代码深读 `kernel_body.h:1041`）
- **BIU 访存合并条件**：源和目标**都连续**才合并。「源连续、目的不连续」无法合并，L2 带宽利用率低——KV 的 ND2NZ 被迫绕行 UB 用 SIMD VF 完成，而不是走 GM→L1 随路转换。（〈KV 的 ND2NZ Layout 转换何时进行〉）
- **硬件 PIPE 顺序发射、顺序执行**：`PIPE_V` 前一条不发射会阻塞整条管线；跨核信号用 SS buffer + cross core flag（能读写 SS buffer 的只有 scalar core）。（〈如何进行 skip-scale 的同步〉〈流水线〉）

## 四、数值格式与 KV Cache layout

- KV Cache 支持 **FP8 / FP4**，使用时反量化为 bfloat16 送 CUBE；decoding 工况下 1C1V 时反量化带宽追不上 CUBE（micro-benchmark），也是 2V 的理由之一。（〈能否用 1C1V〉）
- 本算子把**一个 token 的 data 与 scale 相邻放置**（旧版 FlashMLA 是块内 data 全在前、scale 全在后），MTE2 一次拷齐。（〈Decoding KV Cache Layout 优化〉）
- scale-O 在 NPU 上代价高（FixPipe 搬 L0C→UB、再 VF 累加），因此 skip-scale（rescale 阈值 `exp(6)`）在 NPU 上收益比 GPU 更大。（〈算法回顾〉〈如何进行 skip-scale 的同步〉）
- 补充（另一来源）：DeepGEMM-Ascend README 载明昇腾侧缩放因子格式为一对 **UE8M0 打包进 int16、MN-major**，与 NVIDIA 格式不同。

## 五、设计决策的硬件依据

源文档最有价值的部分，是把四个设计选择全部还原成硬件数字：

1. **为什么每 AI Core 64 个 q head**：存算比按 CUBE 4096 FMA/cycle、L2 5 TB/s 计算，head ≥ 64 才是 compute bound；head=128 会把 L0C 撑爆（128×512 float32 = 256 KB），且 MMAD M=64 已够快。（存算比对照表见源文档，head×topk 从 16×512 到 128×1024 共 8 组数据）
2. **为什么 1C2V 而非 1C1V**：三个 bound——MTE2 队列深 16、反量化带宽、FixPipe 128 B/cycle。
3. **为什么 KV 经 UB 绕行做 ND2NZ**：BIU 只在源、目的都连续时合并访存。
4. **B_TOPK 的两个口径**：1C1V 讨论里 FixPipe bound 要求 `> 73.14`（报告建议 96/128）；最终 1C2V 代码取 64，gather2 折半后每 AIV 恰好 16 个在飞 MTE2 请求，与 16 的队列深度持平（FlashMLA `config.h:55`、`kernel_body.h:1041`）。

## 六、性能锚点与边界

- DeepSeek V4.1 典型工况：稀疏注意力 prefill **410 TFLOPS**（硬件峰值 95%）、decode **360 TFLOPS**（83%）——厂商自报。
- **本文档没有的**：HBM 容量与带宽、TDP/功耗、卡间互联、Die 面积。L2 的 5 TB/s 是片上带宽，勿与 HBM 混淆。卡级信息全批开源材料里仅出现「950DT」型号名与 Atlas 850E 超节点字样（出自 DeepEP-Ascend README），无 data sheet。

## 来源

- 源文档：[deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) `docs/20260930-ascend-prefill-deep-dive-zh.md`（本站存档：[Ascend950 稀疏注意力 Forward 算子与优化技术简析](Ascend950_稀疏注意力Forward算子与优化技术简析.md)）
- 补充：[deepseek-ai/DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend)、[deepseek-ai/DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend) README（缩放因子格式、卡级型号）
- 代码深读：2026-09-30 浅克隆核对（B_TOPK 取值、MTE2 在飞数、VLOOP 检出情况）
- 背景报道：深度求索公众号《DeepSeek 开源昇腾基础组件》（2026-09-30）
