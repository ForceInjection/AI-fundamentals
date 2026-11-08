# 哈佛 CS2680 · Modern AI Systems

哈佛 SEAS 课程 **CS2680「Modern AI Systems: Agents and System Optimizations」**（Fall 2026）的课件存档。课程由 Juncheng Yang（MaDSys 组）主讲，沿「Agent 循环 → LLM serving（batching / scheduling / KV cache / 量化 / 投机解码）→ 底层硬件」一条线讲 AI 系统优化，作业（Mads-Lens 到 Mads-Stack）也沿这条栈走。

## 已收录讲次：Efficient LLM serving basics I–III

| 文件                           | 讲次                                                         | 页数 | 内容                                                                                                                                                                                                                   |
| ------------------------------ | ------------------------------------------------------------ | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `09_llm_serving_basics_1.pdf`  | Basics I: Introduction to LLM（2026-10-19）                  | 36   | Transformer 架构、Llama 3、MoE（Mixtral）、KV cache、prefill/decode 两阶段、算术强度（FLOP/字节）与 bound 分析                                                                                                         |
| `10_llm_serving_basics_2a.pdf` | Basics II: Intro to GPUs（2026-10-21，上半）                 | 41   | GPU 基础与架构：GPU 形态（消费卡 / 数据中心 8-GPU 板）、GPU 是吞吐机器、内存层级、roofline 模型、prefill compute-bound vs decode bandwidth-bound                                                                       |
| `10_llm_serving_basics_2b.pdf` | Basics II: Intro to GPUs（2026-10-21，下半）                 | 44   | GPU 编程与 kernel 进展：CUDA 编程模型（warp / Triton）、nsys 与 ncu 的分工（timeline 定位慢区域，ncu 解释单个 kernel）、GPU utilization 指标口径、FlashAttention 到 FlashAttention-4 的演进、kernel fusion、CUDA graph |
| `11_llm_serving_basics_3.pdf`  | Basics III: scheduling, paging and parallelism（2026-10-28） | 59   | LLM serving 与传统负载的差异（输出长度未知 / 瓶颈漂移 / Agent 与 reasoning 拉大差距）、性能评估口径（TTFT / TPOT / SLO / goodput）、调度与准入、batching、PagedAttention、TP / PP / DP / EP / SP 并行                  |

三讲主线连贯：Basics I 建立模型侧的算力与访存账，Basics II 落到 GPU 硬件与 kernel，Basics III 回到系统层（调度、分页、并行）。Basics I 的 readings 是 Transformer / Llama 3 / Mixtral 原始论文；Basics II 的 readings 覆盖 CUDA 编程指南、roofline、FlashAttention 谱系、FlashInfer、TVM / Mirage / MPK、CUDA graph 与 PyTorch 2。

## 来源

- 课程主页（schedule 含全部 slide 链接）：<https://cs2680.com>
- Readings：[Basics I](https://cs2680.seas.harvard.edu/readings/oct-19) · [Basics II](https://cs2680.seas.harvard.edu/readings/oct-21)
- 下载日期：2026-11-08。课件仍在随课程更新（schedule 标注 tentative），引用其中数字前建议核对最新版本。
