# UC Berkeley · LLM Agents（MOOC）

> 机构：Berkeley RDI · 主讲：Dawn Song；客座：Xinyun Chen（Google DeepMind）、Kaiyu Yang（Meta FAIR）
> 开课：2024 Fall 基础版 → 2025 Spring 进阶版（Advanced LLM Agents），此后未再更新
> 资源可访问性核实：2026-11-19

## 定位

讲 LLM Agents：agentic 工作流、tool use、推理、安全，把 agent 线展开成独立体系。在课程体系里承担「加强 Agent 系统」的角色，与仓库 `08_agentic_system` 方向对位；serving 与基础设施层不涉及（那部分由 [CS2680](harvard-cs2680/README.md) 与 [CMU 11-868](cmu-11-868.md) 补）。

## 课程主线

12 讲，覆盖：agent 基础与推理模式（ReAct 类）、agentic 工作流设计（规划/分解/反思）、tool use 与 code execution、multi-agent 协作、推理增强（self-consistency、验证）、benchmark 与评测、安全与对齐（prompt injection、越狱）。进阶版（SP25）加入了更多产业界视角的客座讲座。

## 资源地图

| 资源             | 位置                                                                      | 公开情况                            |
| ---------------- | ------------------------------------------------------------------------- | ----------------------------------- |
| 课程主页（SP25） | <https://llmagents-learning.org/sp25>                                     | 完全公开                            |
| 讲课录像         | 主页逐讲链接 YouTube                                                      | 12 讲全公开                         |
| Slides           | 课程主页逐讲直链（`rdi.berkeley.edu/llm-agents-mooc/slides/<讲次>.pdf`，目录索引页不开放，从主页进） | PDF 全公开                          |
| Quiz 与 lab      | 主页：12 个 quiz（Google Forms）+ 2 个 Lean 代码 lab（starter code 公开） | 公开，quiz 提交后可选领证书（五档） |

## 备注

检索这门课时注意：<https://llm-agents.io> 不是本课，那是一个无关的博客站；Berkeley 课程真身在 llmagents-learning.org。

内容截至 2025 Spring，agent 领域发展快，lecture 里工具生态相关的具体产品名可能已过时，方法论部分（推理模式、评测、安全）仍然稳定。仓库 `08_agentic_system` 的记忆、多 Agent、MCP、上下文工程文章可与对应讲次互为补充。
