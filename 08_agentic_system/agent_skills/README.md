# Agent Skills（智能体技能）

Skill 是把成熟工作流装进模型能力包的最小单元：一个 SKILL.md 描述何时用、怎么用，配上脚本与参考资料就是可分发的能力包。本目录收三篇方法论文章（编号即阅读顺序）与两个可安装的 Skill 项目。

## 文章

- **[01 · 给 Claude 写本「标准操作手册」：Agent Skills 实战与深度解析](01_claude_skills_guide.md)** — 扩展智能体能力的工具定义规范、调试方法与最佳实践。配套阅读：[The Complete Guide to Building Skills for Claude](01_the_complete_guide_to_building_skill_for_claude.pdf)。
- **[02 · 一个人调度四个角色：多会话协作 Skill 的实测与边界](02_multi-role-agent-skill.md)** — 把《[评审、修改、测试、裁决](../multi_agent/01_claude-code-multi-session-playbook.md)》装进一个用户级 skill：四个角色各开一个会话、一个人调度与拍板的实践，含三层结构、端到端实测、四个缺口与设计边界。配三张手绘图（见 [assets/](assets/README.md)）。
- **[03 · 12 小时、636 个测试：自驱型 Skill 能走多远](03_self-driving-skill-for-complex-projects.md)** — 从 Simon Willison 的 SPARQLite 实验切入，区分 Skill 的策略层与运行时：Skill 单独适合单会话有界任务，跨 session 的复杂项目需要 Workflow 或 Agent SDK harness。含协作层、编排层、运行时三层路径与最小骨架。配三张手绘图。

## 可安装的 Skill 项目

- **[multi-session-collab/](multi-session-collab/)** — 02 篇的配套 Skill（[SKILL.md](multi-session-collab/SKILL.md)）：复制到 `~/.claude/skills/` 即可使用；附 [references/](multi-session-collab/references/) 与 [setup 脚本](multi-session-collab/scripts/)。
- **[pdf_translator/](pdf_translator/)** — PDF 翻译 Skill（[SKILL.md](pdf_translator/SKILL.md)）：附文本抽取、Markdown 生成等 [scripts/](pdf_translator/scripts/) 与 API 参考。

## 相关目录

- [multi_agent](../multi_agent/01_claude-code-multi-session-playbook.md) — 02 篇所封装的多会话协作手册原文
- [agent_infra](../agent_infra/README.md) — harness 生态、沙箱与 K8s 基础设施
