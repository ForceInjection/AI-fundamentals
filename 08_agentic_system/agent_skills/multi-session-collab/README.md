# multi-session-collab：评审、修改、测试、裁决的多会话协作 Skill

把「评审者 / 修改者 / 测试者 / 评判者」四个角色的协作固化成一份可执行 skill：先选路线（跨会话消息 / Agent Teams / Dynamic Workflow），再按协议下发任务卡、收集结构化回报，最后由 judge 出三档裁决。

本目录是 skill 的仓库副本，可直接复制到用户级目录使用：

```bash
cp -R 08_agentic_system/agent_skills/multi-session-collab ~/.claude/skills/
```

## Structure

- `SKILL.md`：触发条件、三问选路、三条操作流程、安全铁律
- `references/protocol.md`：任务卡、回报模板、findings schema、rubric、裁决三档
- `references/setup.md`：版本要求、后台会话、命名与发现、入站矩阵、故障排查
- `scripts/print-setup.sh`：启动清单（只读；不修改配置、不启动会话）

## 配套阅读

- 操作手册：[评审、修改、测试、裁决：Claude Code 多会话协作实践手册](../../multi_agent/docs/claude-code-multi-session-playbook.md)
- 设计说明与实测：[从手册到 Skill：把多会话协作交给 Agent 执行](../docs/from-playbook-to-skill.md)
