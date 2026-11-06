# Agent 基础设施（Agent Infra）

Agent 跑起来需要一整套地基：harness 把模型装成能干活的 Agent，沙箱给它安全边界，编排层把它接进 K8s，存储层为它的文件与记忆做支撑。本目录按六条线组织 15 篇文章（编号即推荐阅读顺序，01–02 为综述入口）。

## Harness 生态（01–06）

- **[01 · AI Agent 基础设施——三个决定性层次：工具、数据、编排](01_ai-agent-infra-stack.md)** — 全面梳理工具层、数据层与编排层的三层架构体系。入口综述。
- **[02 · AI Agent 基础设施的崛起](02_the-rise-of-ai-agent-infrastructure.md)** — 基础设施生态的演进趋势、核心玩家与投资方向。
- **[03 · OpenHarness 深入浅出](03_openharness-deep-dive.md)**（[配套 PPT](03_openharness-deep-dive.pptx)）— 开源智能体基础设施：工具、持久记忆与安全隔离运行边界。
- **[04 · Pi 的极简 harness](04_pi-agent-deep-dive.md)** — 4 个工具、42 家供应商、11 万 star 的极简路线，与 DeepSeek Harness、OpenHarness 的路线对比。
- **[05 · 一切皆插件：DeepSeek Harness](05_deepseek-harness-deep-dive.md)** — 基于 Cordus 的「一切皆插件」架构、profile/bundle 分层组装、事件溯源会话日志与四条边界。
- **[06 · DeepSeek Harness：一本书讲透 Agent 怎么组织](06_deepseek-harness-book-charts.md)** — 书介与概念图解：五张手绘图讲 Harness 位置、「一切皆插件」、Skill/Tool/Hook 与 Guard/Approval/Sandbox。

## 沙箱与执行环境（07–09、11）

- **[07 · 一天 300 万个沙盒：DSec](07_dsec-deep-dive.md)** — 梁文锋署名论文（arXiv:2609.22978）：单单元 160 节点、日 300 万沙盒的 Agent RL 执行环境，含 Agent 钻训练环境漏洞的生产实录。
- **[08 · Agent Sandbox 的演进与设计范式](08_agent-sandbox-design.md)**（[配套 PPT](08_agent-sandbox-design.pptx)）— 从「硬件级隔离」向「策略优先」演进的技术趋势。
- **[09 · Kagent：Kubernetes 运维智能体](09_deep-dive-kagent-k8s-ops-agent.md)**（[配套 PPT](09_deep-dive-kagent-k8s-ops-agent.pptx)）— Kagent 核心架构 + 阿里云 ACK 运维智能体实战。
- **[11 · Claude Code Sandbox 安全隔离机制](11_claude-code-sandbox.md)** — Bubblewrap 底层隔离架构与工程实现。

## K8s Operator（10）

- **[10 · OpenClaw Operator 架构深度解析](10_openclaw-operator-deep-dive.md)** — Server-Side Apply 冲突解决、StatefulSet 持久化绑定、容器级软隔离与进程级沙箱。

## 接口解耦（13）

- **[13 · 扩展托管智能体](13_scaling-managed-agents.md)** — AI 原生基础设施的「POSIX 时刻」：通用接口解耦智能体应用与底层模型。

## 虚拟文件系统（14–15）

- **[14 · 在 Elasticsearch 之上实现虚拟文件系统（中文）](14_virtual-filesystem-elasticsearch_zh.md)**（[英文原文](15_virtual-filesystem-elasticsearch.md)）— 以 Mintlify 设计为蓝本：DLS 访问控制、`ls`/`cd`/`find` 内存支撑、两阶段检索优化 `grep`，为 Agent 暴露形似 POSIX 的只读文件接口。

## TUI 实战（12）

- **[12 · DeepSeek-TUI 实战](12_deepseek-tui-in-practice.md)** — 1M 上下文与前缀缓存切入，Plan/Agent/YOLO 审批模式全流程，「发现 Bug → 修复 → 验证」典型工作流。

## 相关目录

- [agent_skills](../agent_skills/) — Skill 编写与多会话协作
- [agent_course_hub](../agent_course_hub/README.md) — 课程学习路径（多篇文章与本目录互链）
