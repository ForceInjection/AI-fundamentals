# 哈佛 CS2680 · Modern AI Systems

哈佛 SEAS 课程 **CS2680「Modern AI Systems: Agents and System Optimizations」**（Fall 2026）的课件存档。课程由 Juncheng Yang（MaDSys 组）主讲，沿「Agent 循环 → LLM serving（batching / scheduling / KV cache / 量化 / 投机解码）→ 底层硬件」一条线讲 AI 系统优化，作业（Mads-Lens 到 Mads-Stack）也沿这条栈走。

## 已收录讲次：全课程已发布课件

### Part I · Agents（2026-09-02 至 2026-10-07）

| 文件                   | 讲次                                                             | 页数 | 内容                                                                                                                            |
| ---------------------- | ---------------------------------------------------------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------- |
| `1_intro.pdf`          | Course overview（2026-09-02）                                    | 50   | 课程概览与政策、AI 系统与 agentic systems 全景、讲师背景（CMU，曾任职 AWS / Snowflake / Cloudflare / Twitter）                  |
| `2_agent_use_1.pdf`    | Agent use I: How Claude Code Works（2026-09-09）                 | 53   | 以用户视角拆解 Claude Code 的工作机制；课程调整公告：LLM 内部背景后移、多讲 agent 少讲 GPU——把 LLM 当「推理黑盒」不影响学 agent |
| `3_agent_use_2.pdf`    | Agent use II: Multi-agent and Best Practice（2026-09-14）        | 72   | 多智能体使用与最佳实践；recap agent loop、状态管理、context 即工作记忆（每轮 prompt / 响应 / 工具结果追加）                     |
| `4_agent_design_1.pdf` | Agent design I: context management and tool design（2026-09-16） | 56   | Agent 循环（ReAct）、工具设计、context 管理、prompt 工程                                                                        |
| `5_agent_design_2.pdf` | Agent design II: Multi-Agent Systems（2026-09-23）               | 46   | 多智能体系统设计                                                                                                                |
| `6_agent_design_3.pdf` | Agent design III: memory, and reliability（2026-09-28）          | 31   | agent memory 与可靠性；课程配套 AI 生成参考资料（notes/agent-evaluation，讲师与 TF 审校）                                       |
| `7_agent_design_4.pdf` | Agent design IV: evaluation（2026-09-30）                        | 37   | agent evaluation 评测方法                                                                                                       |
| `8_agent_design_5.pdf` | Agent design V: cost-effective agents（2026-10-07）              | 54   | 成本与性能：更便宜更快的 agent（Assignment 3 配套，带 leaderboard 规则说明）                                                    |

### Part II · Systems for LLMs（2026-10-19 起）

| 文件                               | 讲次                                                         | 页数 | 内容                                                                                                                                                                                                                             |
| ---------------------------------- | ------------------------------------------------------------ | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `09_llm_serving_basics_1.pdf`      | Basics I: Introduction to LLM（2026-10-19）                  | 36   | Transformer 架构、Llama 3、MoE（Mixtral）、KV cache、prefill/decode 两阶段、算术强度（FLOP/字节）与 bound 分析                                                                                                                   |
| `10_llm_serving_basics_2a.pdf`     | Basics II: Intro to GPUs（2026-10-21，上半）                 | 41   | GPU 基础与架构：GPU 形态（消费卡 / 数据中心 8-GPU 板）、GPU 是吞吐机器、内存层级、roofline 模型、prefill compute-bound vs decode bandwidth-bound                                                                                 |
| `10_llm_serving_basics_2b.pdf`     | Basics II: Intro to GPUs（2026-10-21，下半）                 | 44   | GPU 编程与 kernel 进展：CUDA 编程模型（warp / Triton）、nsys 与 ncu 的分工（timeline 定位慢区域，ncu 解释单个 kernel）、GPU utilization 指标口径、FlashAttention 到 FlashAttention-4 的演进、kernel fusion、CUDA graph           |
| `11_llm_serving_basics_3.pdf`      | Basics III: scheduling, paging and parallelism（2026-10-28） | 59   | LLM serving 与传统负载的差异（输出长度未知 / 瓶颈漂移 / Agent 与 reasoning 拉大差距）、性能评估口径（TTFT / TPOT / SLO / goodput）、调度与准入、batching、PagedAttention、TP / PP / DP / EP / SP 并行                            |
| `12_llm_serving_spec_decoding.pdf` | Speculative decoding（2026-11-11）                           | 78   | 投机解码全栈：为什么 decode 慢（memory-bound 账）、draft–verify–accept 与拒绝采样的保真性、加速比算术；drafter 谱系（自投机 / Medusa / EAGLE / 检索式）；验证树与接受规则；serving 系统集成（SmartSpec / MagicDec 等吞吐侧权衡） |

### 作业分享与反馈

| 文件                                | 场次                                                       | 页数 | 内容                                                            |
| ----------------------------------- | ---------------------------------------------------------- | ---- | --------------------------------------------------------------- |
| `assignment3_sharing_protected.pdf` | Student sharing II: Harness optimization（2026-10-26）     | 49   | 学生 Assignment 3（agent harness 优化）分享，图片型页面无文本层 |
| `feedback_assignment.pdf`           | Assignments feedback（2026-11-02）                         | 58   | 作业复盘：更好地使用 agent、GPU 编程补充、LLM serving III 延伸  |
| `assignment3_feedback_protected.pdf` | Assignment 3 feedback（受密码保护，2026-11-21 新挂出）    | 17   | 与另外两份 protected 文件同为打开密码保护，无法提取文本与渲染；按命名归入 Assignment 3 反馈场次，具体日期以课程为准 |
| `assignment4_sharing_protected.pdf` | Student sharing III: serving engine showcase（2026-11-18） | 64   | 学生 Assignment 4（serving engine）展示，图片型页面无文本层     |

课程主线两段衔接：Part I 从「用 agent」到「设计 agent」（loop → context 与工具 → 多智能体 → 记忆与可靠性 → 评测 → 成本），Part II 从模型账本落到 GPU 与 kernel，再回到系统层（调度、分页、并行、投机解码）。Basics I 的 readings 是 Transformer / Llama 3 / Mixtral 原始论文；Basics II 覆盖 CUDA 编程指南、roofline、FlashAttention 谱系、FlashInfer、TVM / Mirage / MPK、CUDA graph 与 PyTorch 2；Speculative Decoding 的 readings 按猜想来源（EAGLE 谱系 / 检索式 / 块扩散）、验证树与接受规则、serving 系统集成、drafter 训练四组展开，收录论文 40+ 篇。

## 来源

- 课程主页（schedule 含全部 slide 链接）：<https://cs2680.com>
- Readings：[Basics I](https://cs2680.seas.harvard.edu/readings/oct-19) · [Basics II](https://cs2680.seas.harvard.edu/readings/oct-21) · [Speculative Decoding](https://cs2680.seas.harvard.edu/readings/nov-11)
- 下载日期：2026-11-08（Basics I–III）；2026-11-19（补收 Speculative Decoding 与全课程已发布课件）；2026-11-21（补收 Assignment 3 feedback，其余 16 个文件经 Content-Length 与线上逐一核对无版本变化）。课件仍在随课程更新（schedule 标注 tentative；截至 2026-11-21，scheduling/prefix caching、剪枝与量化、KV cache 优化三讲的 slide 尚未挂出，后续还有 routing 与 load balancing、agentic systems 两讲），引用其中数字前建议核对最新版本。
