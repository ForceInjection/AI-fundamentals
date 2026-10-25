# assets

本目录存放《[一个人调度四个角色：多会话协作 Skill 的实测与边界](../multi-role-agent-skill.md)》与《[12 小时、636 个测试：自驱型 Skill 能走多远](../self-driving-skill-for-complex-projects.md)》的配图。

- 多会话协作 skill：三张手绘图（`skill-01` … `skill-03`），rough.js + 霞鹜文楷程序化生成，1080 逻辑宽 2x 渲染 PNG
  - 图 1：skill 三层结构（决策 / 协议 / 动作，`skill-01-three-layers`）；图 2：实测暴露的四个缺口（`skill-02-four-gaps`）；图 3：一次端到端闭环（`skill-03-closed-loop`）
  - 数据来源：用户级 skill `~/.claude/skills/multi-session-collab`（253 行）与 2026-10-18 的端到端实测；图内为示意，行数以实测时为准
- 自驱型 skill：三张手绘图（`selfdriving-01` … `selfdriving-03`）
  - 图 1：Skill / Workflow / Harness 三层职责（`selfdriving-01-three-layers`）；图 2：自驱回路与状态文件（`selfdriving-02-loop`）；图 3：自驱边界与三条路径（`selfdriving-03-paths`）
