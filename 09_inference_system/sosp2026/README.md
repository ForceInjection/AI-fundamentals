# SOSP 2026 AI 系统论文精读

SOSP（ACM 操作系统原理大会）第 32 届，2026 年 9 月 29 日–10 月 2 日，布拉格。本届约 65 篇录用论文中，与 LLM 推理、GPU 系统和 Agent 执行环境相关的超过 20 篇；本目录从中选出 9 篇做全文精读，覆盖三条线：推理服务（4 篇）、GPU 系统（3 篇）、Agent × OS（2 篇）。

**入口综述：[系统的边界在移动：SOSP 2026 九篇 AI 系统论文精读](sosp2026-ai-systems-review.md)** — 三条线组织九篇论文：负载假设 → 分线综述 → 三条主线（同步协调退场 / 编译世界观扩张 / 安全边界下沉）与折扣清单、对号表。

每个条目一份编号笔记（`references/`），论文原文 PDF 在 `papers/`（作者版/arXiv/官网公开版，2026-11-14 下载）。笔记中所有数字均为论文口径，生产系统论文（Janus、ACDC、Bedrock）的数字另属厂商口径，引用时注意区分。

## 论文清单

| # | 论文 | 机构 | 主题 | 笔记 | PDF |
|---|------|------|------|------|-----|
| 01 | Janus: Multi-LLM Serving at Production Scale | 北京大学 + 京东等 | 多 LLM 共存服务：双时间尺度调度 + 快照激活 | [笔记](references/01-janus-multi-llm-serving.md) | [PDF](papers/01_janus-multi-llm-serving.pdf) |
| 02 | Taming Inference Workloads at Global Scale: Foundation Model Serving in Amazon Bedrock | AWS | 全球多区域 FM 服务：准入/分类/跨区路由/调度 | [笔记](references/02-bedrock-fm-serving.md) | [PDF](papers/02_bedrock-fm-serving.pdf) |
| 03 | LLM-42: Enabling Determinism in LLM Inference with Verified Speculation | Microsoft Research | 确定性推理：验证-回滚协议按请求计费 | [笔记](references/03-llm-42-determinism.md) | [PDF](papers/03_llm-42-determinism.pdf) |
| 04 | A Few GPUs, A Whole Lotta Scale: Faithful LLM Training Emulation with CrystalLLM | 阿里 + 上交大 + 哈佛 | 少量 GPU 忠实仿真大规模训练（平均误差 0.58%） | [笔记](references/04-crystalllm-training-emulation.md) | [PDF](papers/04_crystalllm-training-emulation.pdf) |
| 05 | Ave: Guiding Agentic GPU Optimization Using Data-Flow Invariants | HKUST + Stanford 等 | 数据流不变量做过程奖励，指导 agent 优化 GPU kernel | [笔记](references/05-ave-agentic-gpu-optimization.md) | [PDF](papers/05_ave-agentic-gpu-optimization.pdf) |
| 06 | StreamEP: Straggler-Tolerant MoE Decoding without Communication Barriers | USC + Google DeepMind | 去掉 MoE 解码的集合通信屏障，commodity 集群 1.7× | [笔记](references/06-streamep-moe-decoding.md) | [PDF](papers/06_streamep-moe-decoding.pdf) |
| 07 | Skill VM: Write Once, Run Everywhere Efficiently | 上交大 IPADS | Agent Skill 的编译与运行时（AOT + JIT） | [笔记](references/07-skill-vm.md) | [PDF](papers/07_skill-vm.pdf) |
| 08 | Don't Let AI Agents YOLO Your Files: Information and Control in Agent-Native Filesystems | UW-Madison + MSR | 290 份事故研究 + Agent 原生文件系统 YoloFS | [笔记](references/08-agent-native-filesystems.md) | [PDF](papers/08_agent-native-filesystems.pdf) |
| 09 | Batched in Back: Characterizing and Optimizing Offline LLM Inference in Production with ACDC | 上交大 + 阿里 | 离线推理刻画（150 万任务 trace）+ 三轴优化 + 在离线混合调度 | [笔记](references/09-acdc-offline-inference.md) | [PDF](papers/09_acdc-offline-inference.pdf) |

## 阅读顺序建议

- 只关心推理服务：01 → 09 → 02 → 03（Janus 与 ACDC 是同代多模型/多任务系统的两种解法，Bedrock 补全球视角，LLM-42 补确定性专题）
- 只关心 GPU 系统：06 → 04 → 05（通信范式 → 仿真评估 → AI 优化 kernel）
- 只关心 Agent 基础设施：08 → 07（文件系统安全边界 → Skill 执行环境）

## 与本仓库其他内容的关联

- 06（StreamEP）与 [KV Cache 与部署系列](../README.md) 互补：PD 分离 + decode-only worker 在顶会语境下已是默认部署形态
- 07（Skill VM）与 [agent_skills](../../08_agentic_system/agent_skills/) 互为镜像：那边讲怎么写 Skill，这边讲 Skill 进入异构环境后系统怎么接
- 08（YoloFS）与 [agent_infra 沙箱系列](../../08_agentic_system/agent_infra/README.md) 构成谱系：进程/VM 沙箱管「代码能不能跑」，文件系统管「动没动不该动的东西」

## 来源说明

PDF 下载渠道：01/04/06/07/09 为作者或实验室公开版（yangtonghome.github.io、ennanzhai.github.io、seojinpark.net、ipads.sjtu.edu.cn），02 为 Amazon Science（cdn.amazon.science），03/05/08 为 arXiv 预印本（2601.17768、2604.18616、2604.13536）。会议官方录用名单见 [sigops.org SOSP 2026 accepted papers](https://sigops.org/s/conferences/sosp/2026/accepted.html)。预印本与会议正式版可能有差异。
