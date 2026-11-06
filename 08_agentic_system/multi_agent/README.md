# 多智能体协作（Multi-Agent）

从理论到落地的一条完整线：两篇系列文章讲清多智能体的基础理论与企业级实现，一篇操作手册把同一套思路落成 Claude Code 上的可复现工作流，一个可运行项目给出参考实现。编号即推荐阅读顺序。

## 文章

- **[01 · Claude Code 原生多会话协作的操作手册](01_claude-code-multi-session-playbook.md)** — 从上下文隔离与多视角评审出发，覆盖跨窗口消息、Agent Teams 编队与 Dynamic Workflow 三条路线，附消息模板、审批规则、故障排查与评判者（judge）设计。配三张手绘图（见 [assets/](assets/README.md)）；其 skill 化与端到端实测见 [agent_skills 02 篇](../agent_skills/02_multi-role-agent-skill.md)。
- **[02 · 多智能体 AI 系统基础：理论与框架](02_part1_multi_agent_ai_fundamentals.md)** — 系列第一篇：多智能体的基本概念、协作模式与框架选型。
- **[03 · 企业级多智能体 AI 系统构建实战](03_part2_enterprise_multi_agent_system_implementation.md)** — 系列第二篇：从架构设计到工程落地的企业级实现路径。

## 可运行项目

- **[multi_agent_system/](multi_agent_system/)** — 02/03 两篇的参考实现（[README](multi_agent_system/README.md)）：多智能体系统源码（research / analysis 等 Agent、消息总线通信），含 Dockerfile 与 pytest 测试。

## 相关目录

- [agent_skills](../agent_skills/README.md) — 01 篇的 skill 化封装（multi-session-collab）与 Skill 方法论
- [agent_infra](../agent_infra/README.md) — harness 生态、沙箱与 K8s 基础设施
