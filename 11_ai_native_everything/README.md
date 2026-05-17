# AI Native 全栈实践

"AI Native" 已成为行业热词，但它的工程边界常被模糊化：是说每个函数都要包一层 LLM？还是把 AI 塞进现有系统的某个角落？本模块从两个正交维度给出可落地的答案——**应用架构**侧以"认知性质决定技术分层"为唯一裁断准则，将业务能力收敛为 Agent / Skill / Tool 三层；**DevOps** 侧则将 AI 定位为需求梳理、建模分析、规范生成、代码实现、自动验证与运行治理的增强器，而非替代者。两者共享同一条底线：**AI Native ≠ 处处用 LLM**，最终决策、风险承担与上线责任仍由明确的人工 Owner 承担。

---

## 1. AI Native 应用架构

从一笔真实的电力现货交易出发，沿"业务能力 → 认知性质 → 技术分层 → 工程清单"链条，推导出一套可复用的 AI Native 应用架构方法论。核心回答三个问题：哪些角色真正需要 Agent、哪些应当退到 Skill 与普通 Tool、以及七项工程实践如何同步就位。

- **[AI Native 应用架构：从一笔真实电力交易看起](ai_native_application/ai-native-architecture.md)** — 完整推导链路，覆盖 5 阶段 12 动作拆解、三层技术分层（Agent / Skill / Tool 经 MCP 暴露）、五个业务角色落层分析、五反模式与三问决策启发法、七项工程治理清单。
- **[分层架构交互图](ai_native_application/ai-native-architecture-diagram.html)** — 可交互 HTML 版三层金字塔结构图，包含 Agent 层（交易策略）、Skill 层（调度 SOP / 结算流程 / 合规检查 / 风险预案）、Tool 层（电价预测 / 负荷预测 / 气象 / 市场行情 / IoT 控制 / 数仓结算六大 MCP），以及共享治理平面的七项工程实践详情面板。

**核心方法论速览**：

| 概念               | 定义                                                                                                                        | 锚点 |
| :----------------- | :-------------------------------------------------------------------------------------------------------------------------- | :--- |
| AI Native 应用     | 业务闭环中至少一个能力需 LLM 推理循环承载（三问第三问为真），其余按认知性质退守 Skill / Tool                                | §7.1 |
| AI Native 应用架构 | 以认知性质为唯一裁断准则，映射到 Agent / Skill / Tool 三层，并在共享治理平面上同步落地七项工程                              | §7.2 |
| 三问决策启发法     | ① 能否用非 LLM 系统可验证完成？→ Tool（MCP 暴露）；② 是否固定流程只需编排？→ Skill；③ 是否需在新颖情况下决定下一步？→ Agent | §5.3 |
| 五反模式           | Agent 化一切 / LLM 直接控物理设备 / Skill 承载物理计算 / MCP Server 沦为裸 RPC / Agent 缺终止条件                           | §5.2 |

---

## 2. AI Native DevOps

AI Native DevOps 是一套以 AI 为增强器的 DevOps 协同框架，说明 AI 在需求、设计、建模、规范、实现、验证、交付与演进各阶段如何参与、如何与人工协同，以及如何衡量价值与控制风险。其核心立场不是"让 AI 全自动替代团队"，而是"让 AI 成为需求梳理、建模分析、规范生成、代码实现、自动验证与运行治理的增强器"。

> 📂 外部参考：[ForceInjection/ai-native-devops](https://github.com/ForceInjection/ai-native-devops)

**文档结构**：

| 文件                                                                                                                 | 说明                                                          |
| :------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------ |
| [`ai-native-devops.md`](https://forceinjection.github.io/ai-native-devops/ai-native-devops.html)                     | 主文章：框架定义、AI 参与度分析、目标架构、治理机制与实施路线 |
| [`ai-native-devops-panorama.html`](https://forceinjection.github.io/ai-native-devops/ai-native-devops-panorama.html) | 全景图（可交互 HTML 版本）                                    |

**按角色推荐阅读路径**：

| 角色             | 推荐章节                               |
| :--------------- | :------------------------------------- |
| 产品经理         | §1、§4.1、§7.6、§9.2、§10.4、§11.1     |
| 架构师           | §1、§4.3、§4.4、§7.2、§7.3、§7.8、§7.9 |
| 开发 / Tech Lead | §1、§4.5、§4.6、§6.1、§7.4、§7.7、§12  |
| 平台 / SRE / QA  | §1、§4.6、§4.7、§7.5、§7.8、§9、§11.3  |

**核心设计原则**：

- **增强而非替换**：AI 主要用于生成轮子、提供选项、执行自动化分析与验证，不直接替代关键业务与工程决策。
- **阶段化参与**：每个阶段都应标注 AI 的参与程度，如"生成轮子""提供选项""自动校对""人工审核后确认"等。
- **人机交环明确**：PRD、用户旅程、领域模型、OpenSpec、上线审批等关键资产，只有经过人工确认后才能进入下一阶段。
- **可验证优先**：任何 AI 生成内容都应转化为可验证工件，例如测试、检查、规范差距分析、审计记录与异常报告。

**参考基线**：文中"当前已有能力"判断参考两个公开项目——[domain-driven-design-skills](https://github.com/domain-driven-design-skills)（DDD 建模 Skill 集，覆盖战略/战术建模与 OpenSpec 桥接）与 [OpenSpec-practise](https://github.com/OpenSpec-practise)（规范驱动开发工作流，含 `proposal.md` / `design.md` / `tasks.md` / `specs/` 及 `/opsx:*` 指令体系）。

---

## 3. 两线交汇

应用架构侧与 DevOps 侧虽从不同起点出发，但在以下四个主题上形成交叉验证：

| 主题     | AI Native 应用架构（§1）                            | AI Native DevOps（§2）                                                      | 共同原则               |
| :------- | :-------------------------------------------------- | :-------------------------------------------------------------------------- | :--------------------- |
| 分层思想 | Agent / Skill / Tool 三层，以认知性质为唯一裁断准则 | 需求 → 设计 → 建模 → 规范 → 实现 → 验证 → 交付 → 演进，每阶段 AI 参与度明确 | 划清 AI 与人的责任边界 |
| 治理机制 | 七项工程实践 + ML Platform / Audit Log 共享治理平面 | HITL 审批节点 + OpenSpec 规范门禁 + 可验证工件                              | 可追溯、可审计、可回滚 |
| 工具协议 | MCP 作为 Agent → Tool 调用的强类型协议              | OpenSpec `/opsx:*` 指令体系作为规范驱动开发的工具协议                       | 标准化互操作接口       |
| 反模式   | Agent 化一切、LLM 直接控物理设备等五条              | 将 AI 输出直接上线无人工确认、用 LLM 替代确定性校验                         | 不逾越 AI 的能力边界   |

---

## 4. 关联模块

- **[08_agentic_system](../08_agentic_system/README.md)** — Agent 系统全栈工程：认知理论、上下文/记忆/MCP/Sandbox 核心组件、多智能体协作与企业级落地。当前模块侧重"架构分层与治理"，08 模块补充"单 Agent 内部机制与基础设施"。
- **[04_cloud_native_ai_platform](../04_cloud_native_ai_platform/README.md)** — 云原生 AI 基础设施：Kubernetes 生态下的 GPU 资源池化（HAMi）、弹性调度、分布式推理/存储系统。当前模块的 Tool 层（MCP 暴露）与 DevOps 实践需依托此模块的集群底座。
- **[06_llm_theory_and_fundamentals](../06_llm_theory_and_fundamentals/README.md)** — LLM 理论基础：量化、MoE、Embedding、Token 等直接影响 Agent 层推理循环的成本与性能边界。
