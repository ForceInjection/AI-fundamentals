# Agent 基础设施（Agent Infra）

Agent 跑起来需要一整套地基：harness 把模型装成能干活的 Agent，沙箱给它安全边界，编排层把它接进 K8s，存储层为它的文件与记忆做支撑。本目录按六条线组织 16 篇文章（编号即推荐阅读顺序，01–02 为综述入口）。

## Harness 生态（01–06）

harness 是「把模型装成能干活的 Agent」的那一层。本组沿三条路线展开：OpenHarness 的大而全、Pi 的极简、DeepSeek Harness 的插件化，路线对比是贯穿全组的主线。

- **[01 · AI Agent 基础设施——三个决定性层次：工具、数据、编排](01_ai-agent-infra-stack.md)** — 全面梳理工具层、数据层与编排层的三层架构体系。入口综述。
- **[02 · AI Agent 基础设施的崛起](02_the-rise-of-ai-agent-infrastructure.md)** — 分析基础设施生态的演进趋势、核心玩家与投资方向。
- **[03 · OpenHarness 深入浅出](03_openharness-deep-dive.md)**（[配套 PPT](03_openharness-deep-dive.pptx)）— 开源智能体基础设施参考实现：43+ 工具、持久记忆与安全隔离运行边界，与 Pi 的极简路线互为对照。
- **[04 · Pi 的极简 harness](04_pi-agent-deep-dive.md)** — 极简 harness 源码深读（Pi @ `a276dabe5`）：默认 4 个工具、系统提示实测 677 token、41 个事件与「内置实现可被撞名整体接管」的可替换协议、42 家供应商的统一层；并与 DeepSeek Harness（其 LLM 层用的正是 Pi 的 pi-ai 包）、OpenHarness 做路线对比，附六条诚实边界。配两张手绘图。
- **[05 · 一切皆插件：DeepSeek Harness](05_deepseek-harness-deep-dive.md)** — 拆解 DeepSeek 开源的 Agent 框架 dsh：基于 Cordus 的「一切皆插件」架构（连 agent loop 本身都是插件）、profile/bundle 分层组装、事件溯源会话日志、四个内置预设与能力 seam，以及开发预览阶段必须正视的四条边界。
- **[06 · DeepSeek Harness：一本书讲透 Agent 怎么组织](06_deepseek-harness-book-charts.md)** — 书介与概念图解：以同模型换 Harness 差 8.6 分的评测数字切入，用五张手绘图讲 Harness 的位置、「一切皆插件」、Skill/Tool/Hook 与 Guard/Approval/Sandbox、四套 Preset；概念解释对照 dsh 源码核对结果，并评点全书「先跑起来、再拆开、最后动手」的认知阶梯。

## 沙箱与执行环境（07–09、11、16）

从单机进程沙箱（Bubblewrap）到论文级 RL 训练环境（DSec），再到集群级运行时（Substrate），覆盖沙箱议题从本机到数据中心的三个尺度。

- **[07 · 一天 300 万个沙盒：DSec](07_dsec-deep-dive.md)** — 梁文锋署名论文深读（arXiv:2609.22978）：单单元 160 节点、日 300 万沙盒的 Agent RL 执行环境，分层组合、按需镜像加载与高密度资源管理，含 Agent 钻训练环境漏洞的生产实录。
- **[08 · Agent Sandbox 的演进与设计范式](08_agent-sandbox-design.md)**（[配套 PPT](08_agent-sandbox-design.pptx)）— 对比 OpenShell、Sandlock 等沙箱方案，梳理从「硬件级隔离」向「策略优先」演进的技术趋势。
- **[09 · Kagent：Kubernetes 运维智能体](09_deep-dive-kagent-k8s-ops-agent.md)**（[配套 PPT](09_deep-dive-kagent-k8s-ops-agent.pptx)）— Kagent 核心架构与工作机制，并以「构建阿里云 ACK 运维智能体」为实战案例。
- **[11 · Claude Code Sandbox 安全隔离机制](11_claude-code-sandbox.md)** — 从 Claude Code 的实际运行环境切入，剖析基于 Bubblewrap 的底层隔离架构与工程实现。
- **[16 · Agent Substrate 深拆](16_substrate-deep-dive.md)** — Google 主导、捐入 CNCF 生态的 K8s Agent 沙箱运行时（@ `cf5768a0`）：Actor/Worker 快照复用实现 10-30x 超售，双层资源模型（CRD + PostgreSQL）绕开 apiserver 规模瓶颈，请求停车吸收池饱和抖动，沙箱内零凭证的零信任安全模型；配手绘机制图，性能数字均标官方口径并附安全 In Progress 诚实清单。

## K8s Operator（10）

K8s 侧的 Agent 工作负载编排：把 Agent 的生命周期管理做成 Operator 模式。

- **[10 · OpenClaw Operator 架构深度解析](10_openclaw-operator-deep-dive.md)** — Server-Side Apply 冲突解决、StatefulSet 持久化绑定、容器级软隔离与进程级沙箱的安全边界设计。

## 接口解耦（13）

Agent 应用与底层模型之间的接口标准化尝试。

- **[13 · 扩展托管智能体](13_scaling-managed-agents.md)** — AI 原生基础设施的「POSIX 时刻」：定义通用接口解耦智能体应用与底层模型，实现长周期任务智能体的灵活托管与扩展。

## 虚拟文件系统（14–15）

为 Agent 的文件操作造一层带访问控制的视图：形似 POSIX，能力可裁剪。

- **[14 · 在 Elasticsearch 之上实现虚拟文件系统（中文）](14_virtual-filesystem-elasticsearch_zh.md)**（[英文原文](15_virtual-filesystem-elasticsearch.md)）— 以 Mintlify 设计为蓝本：DLS 访问控制、`ls`/`cd`/`find` 内存支撑、两阶段检索优化 `grep`，为 Agent 暴露形似 POSIX 的只读文件接口。

## TUI 实战（12）

把 harness 的能力落到手边：终端里的编程 Agent 工作流。

- **[12 · DeepSeek-TUI 实战](12_deepseek-tui-in-practice.md)** — 以 DeepSeek V4 的 1M 超长上下文与前缀缓存为切入，走通从安装、配置到 Plan/Agent/YOLO 审批模式的完整实战闭环，并通过「发现 Bug → 修复 → 验证」演示 Coding Agent 的典型工作流。

## 相关目录

- [agent_skills](../agent_skills/) — Skill 编写与多会话协作
- [agent_course_hub](../agent_course_hub/README.md) — 课程学习路径（多篇文章与本目录互链）
