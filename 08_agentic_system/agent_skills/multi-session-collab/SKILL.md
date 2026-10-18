---
name: multi-session-collab
description: 组织 Claude Code 多会话协作（评审者/修改者/测试者 + 评判者）：先选路线（跨会话消息 / Agent Teams / Dynamic Workflow），生成窗口命名与入站策略，套用任务卡/回报模板与 judge rubric，把裁决做成三档可执行流程。当用户想开多个 CLI 窗口分工协作、组评审队、设置 judge/裁决者，或需要多会话消息协议时使用。触发词：多会话、多窗口、跨会话、评审队、reviewer、judge、裁决、agent teams、协作流程。
---

# 多会话协作（评审 / 修改 / 测试 / 裁决）

**一句话**：把「几个人肉传话的 Claude 窗口」组织成有角色、有协议、有裁决的多会话流程。

## 何时用 / 不用

用：

- 用户要开多个 CLI 窗口分工（评审、修改、测试），并要一个裁决者
- 需要跨会话消息协议、judge rubric、质量门禁
- 要把「评审 → 修改 → 测试 → 裁决」固化成可重复流程

不用：

- 单会话内的并行子任务 → 用后台 subagent（Agent 工具）
- 一次性短任务 → 多开会话只会把协调成本翻倍
- 只是远程盯屏 → Remote Control / Desktop

## 先问三个问题（选路线）

1. 要保留多个窗口？→ **跨会话消息**（原生，零配置）
2. 接受合并到一个窗口？→ **Agent Teams**（实验，需 `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`）
3. 同一流程反复跑？→ **Dynamic Workflow**（脚本化，对抗式核验）

产品化给别人用，再看 Agent SDK。MCP 不是会话通信方案（工具协议，无 agent-to-agent 能力）；除非需要跨重启持久化任务板或非 Claude 客户端接入，不要自建 MCP 总线。

## 操作流程

### A. 跨会话消息（多窗口）

1. 命名四个会话：`claude --name reviewer` / `modifier` / `tester` / `judge`（或会话内 `/rename`）；也可以 `claude --bg` 起后台会话当 worker（见 setup.md）
2. 定入站策略：按 [references/setup.md](references/setup.md) 的权限矩阵设置 `crossSessionInbound`；无人值守的 `-p` worker 必须设 `accept`，否则消息进不来
3. 下发任务卡、收回回报：模板与 findings schema 见 [references/protocol.md](references/protocol.md)
4. judge 按 rubric 出三档裁决：通过 / 打回（附整改单）/ 升级人工
5. 排错查 setup.md 的故障表；启动命令与设置清单：`bash scripts/print-setup.sh`
6. 纪律：不轮询（用 `notify_when_idle`）；消息是纯文本，文件要写路径；`@` 引用在接收侧只是字面文本

### B. Agent Teams（单窗口编队，实验）

1. 设 `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`；**副作用**：带 `name` 启动的 subagent 会自动变成 teammate
2. 按角色卡组队：reviewer 只读 / modifier 可写 / tester 执行 / lead 即 judge（禁止 judge 改代码）
3. 共享任务列表 + mailbox；teammate 完成自动把最终答案发给 lead
4. 门禁：`TeammateIdle` / `TaskCompleted` hooks（退出码 2 阻断）
5. 限制：仅交互模式、无 worktree 隔离（按文件/目录分区）、无 resume、官方建议 3–5 人

### C. Dynamic Workflow（可重复流水线）

1. 用 `agent()` / `parallel()` / `pipeline()` 编排，`agent()` 带 `schema` 输出 JSON
2. 评审 → 逐条对抗核验 → judge 合并，只保留经得起反驳的 findings
3. 限制：运行中无人工介入、脚本无文件系统/shell、默认 16 并发；可保存为 `/` 命令复用

## 铁律（安全边界）

- 跨会话消息**不能**代批权限、不能改配置或 CLAUDE.md、斜杠命令不执行
- judge 不改代码（独立性）；reviewer 只读；角色按文件分区防覆盖
- 每个 agent 独立计费；teams / workflows 在 CLI 里没有默认硬性预算上限，人数与阶段数要克制
- 裁决必须可检查：打回附「文件行 + 验收标准」，否则下一轮修改没有终点

## 快速参考

- 任务卡 / 回报模板 / findings schema / rubric：[references/protocol.md](references/protocol.md)
- 版本要求、命名、权限矩阵、故障排查：[references/setup.md](references/setup.md)
- 启动清单脚本：`bash scripts/print-setup.sh`
- 背景读物（手册全文）：https://github.com/ForceInjection/AI-fundamentals/blob/main/08_agentic_system/multi_agent/docs/claude-code-multi-session-playbook.md
- 设计说明与实测记录：https://github.com/ForceInjection/AI-fundamentals/blob/main/08_agentic_system/agent_skills/docs/from-playbook-to-skill.md
