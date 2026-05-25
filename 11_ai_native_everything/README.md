# AI Native 全栈实践

"AI Native" 已成为行业热词，但它的工程边界常被模糊化——是说每个函数都要包一层 LLM？还是把 AI 塞进现有系统的某个角落？本模块给出可落地的答案：**AI Native ≠ 处处用 LLM**，最终决策、风险承担与上线责任仍由明确的人工 Owner 承担。

核心内容已拆分至独立仓库：[ai-native-devops](https://github.com/ForceInjection/ai-native-devops)，按三层框架组织——从个人到团队再到架构，形成人机协同工程的完整方法论。

---

## 1. 三层框架

```text
Vibe Coding（个人）──→ AI Native DevOps（团队）──→ AI Native Architecture（应用架构）
   建立术语基础             将 AI 嵌入开发流程              将 AI 嵌入产品系统
```

### 1.1 Vibe Coding

AI 作为"一等协作者"参与工程：开发者提供意图、约束、契约为输入，AI 产出草稿与自动化验证，所有输出经人工确认后方可合入主线。

**四个基础概念**：

| 概念      | 定义                         |
| --------- | ---------------------------- |
| **Agent** | 在不确定环境下自主推理与行动 |
| **MCP**   | Agent 调用 Tool 的标准协议   |
| **A2A**   | Agent 间通信协议             |
| **Skill** | 固定 SOP 编排                |

**成本金字塔**（贯穿三层）：

| 层级  | 成本 | 特征                                |
| ----- | ---- | ----------------------------------- |
| Tool  | 最低 | 确定性 I/O，不产生幻觉              |
| Skill | 中等 | 固定流程，最小化 LLM 调用           |
| Agent | 最高 | LLM 推理循环，Token 成本 + 幻觉风险 |

核心原则：**能用 Tool 就不要用 Agent**。

### 1.2 AI Native DevOps

8 阶段全流程框架，覆盖软件交付完整生命周期：P1 愿景 → P2 原型 → P3 领域建模 → P4 OpenSpec 规范 → P5 实现与测试 → P6 质量验收 → P7 部署交付 → P8 变更演进。每阶段明确定义 AI 的输入、输出、建议工件与人工确认节点。

> 📂 详见：[ai-native-devops](https://github.com/ForceInjection/ai-native-devops)（[在线版](https://forceinjection.github.io/ai-native-devops/)）

**核心设计原则**：

- **增强而非替换**：AI 主要用于生成轮子、提供选项、执行自动化分析与验证，不直接替代关键决策
- **阶段化参与**：每阶段标注 AI 参与程度——"生成轮子""提供选项""自动校对""人工审核后确认"等
- **人机交环明确**：PRD、用户旅程、领域模型、OpenSpec、上线审批等关键资产，只有经过人工确认后才能进入下一阶段
- **可验证优先**：任何 AI 生成内容都应转化为可验证工件——测试、检查、规范差距分析、审计记录与异常报告

### 1.3 AI Native 应用架构

以"认知性质决定技术分层"为唯一裁断准则，将业务能力收敛为 Agent / Skill / Tool 三层，并在共享治理平面上落地七项工程实践。从真实电力现货交易场景推导完整链路。

> 📂 详见：[ai-native-devops/ai-native-architecture](https://github.com/ForceInjection/ai-native-devops/tree/main/ai-native-architecture)（[在线版](https://forceinjection.github.io/ai-native-devops/ai-native-architecture)）

**三问决策启发法**：① 能否用非 LLM 系统可验证完成？→ Tool（MCP 暴露）；② 是否固定流程只需编排？→ Skill；③ 是否需在新颖情况下决定下一步？→ Agent

**五反模式**：Agent 化一切 / LLM 直接控物理设备 / 缺少治理平面 / MCP Server 沦为裸 RPC / Agent 缺终止条件

---

## 2. CloudPilot 端到端案例

CloudPilot 是一个云管理平台 MVP，验证三层框架的协同效应。以 Vibe Coding 为日常工作流，完成 P1-P4 阶段（访谈 → OpenSpec），仅需 6 层工件：访谈笔记 → PRD → Mock UI → DDD 模型 → OpenSpec → 代码桥接。所有 Prompt 可录制并由 `ddd-modeler` 和 `openspec-author` 两个 sub-agent 重放。

> 📂 详见：[ai-native-devops/cloudpilot-case](https://github.com/ForceInjection/ai-native-devops/tree/main/cloudpilot-case)

---

## 3. 两线交汇

应用架构侧与 DevOps 侧虽从不同起点出发，但在以下四个主题上形成交叉验证：

| 主题     | 应用架构                    | DevOps                                 | 共同原则               |
| -------- | --------------------------- | -------------------------------------- | ---------------------- |
| 分层思想 | Agent / Skill / Tool 三层   | 8 阶段 AI 参与度明确                   | 划清 AI 与人的责任边界 |
| 治理机制 | 七项工程实践 + 共享治理平面 | HITL 审批 + OpenSpec 门禁 + 可验证工件 | 可追溯、可审计、可回滚 |
| 工具协议 | MCP 作为 Agent→Tool 协议    | OpenSpec `/opsx:*` 指令体系            | 标准化互操作接口       |
| 反模式   | Agent 化一切等五条          | AI 输出直接上线无确认等                | 不逾越 AI 的能力边界   |

---

## 4. 按角色推荐阅读路径

| 角色             | 推荐入口                                                           |
| ---------------- | ------------------------------------------------------------------ |
| 所有读者         | 先读 `vibe-coding-intro-for-traditional-dev.md`，统一术语          |
| 产品经理         | §1–§4.1, §7.6, §9.2, §10.4, §11.1                                  |
| 架构师           | `ai-native-architecture.md` 全文 + §4.3–§4.4, §7.2–§7.3, §7.8–§7.9 |
| 开发 / Tech Lead | §4.5–§4.6, §6.1, §7.4, §7.7, §12 + `cloudpilot-case/`              |
| 平台 / SRE / QA  | §4.6–§4.7, §7.5, §7.8, §9, §11.3                                   |

---

## 5. 关联模块与参考

- **[08_agentic_system](../08_agentic_system/README.md)** — Agent 系统全栈工程，补充单 Agent 内部机制与基础设施
- **[04_cloud_native_ai_platform](../04_cloud_native_ai_platform/README.md)** — Tool 层（MCP 暴露）与 DevOps 实践所需的集群底座
- **[06_llm_theory_and_fundamentals](../06_llm_theory_and_fundamentals/README.md)** — LLM 理论基础，影响 Agent 层推理成本与性能边界
- **[domain-driven-design-skills](https://github.com/domain-driven-design-skills)** — DDD 建模 Skill 集，覆盖战略/战术建模与 OpenSpec 桥接
- **[OpenSpec-practise](https://github.com/ForceInjection/OpenSpec-practise)** — 规范驱动开发工作流，含 `proposal.md` / `design.md` / `tasks.md` / `specs/` 及 `/opsx:*` 指令体系
