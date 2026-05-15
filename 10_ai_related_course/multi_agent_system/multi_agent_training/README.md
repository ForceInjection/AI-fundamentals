# 多智能体 AI 系统培训材料

本培训围绕多智能体系统的理论基础与企业级落地展开，配套两份核心文档：[part1_multi_agent_ai_fundamentals.md](../../../08_agentic_system/multi_agent/docs/part1_multi_agent_ai_fundamentals.md) 提供理论框架与 LangGraph、LangSmith 解析，[part2_enterprise_multi_agent_system_implementation.md](../../../08_agentic_system/multi_agent/docs/part2_enterprise_multi_agent_system_implementation.md) 给出架构设计、技术实现与部署实践。培训材料将这两份文档拆解为五天可教学的结构化课程。

## 培训目标

- 理解多智能体系统核心概念、BDI 架构与 LLM 驱动架构
- 使用 LangGraph 构建工作流，使用 LangSmith 实现全链路监控
- 设计并实现企业级多智能体系统架构
- 完成智能客服等典型场景的开发与部署
- 掌握开发、部署、监控、运维的最佳实践

## 培训信息

- **对象**：AI 架构师、后端 / AI 工程师、产品与技术管理人员
- **时长**：5 天 / 40 学时
- **前置要求**：Python 3.8+、基础 AI/ML 概念、Docker、基本系统架构知识

## 课程结构

```bash
multi_agent_training/
├── README.md                          # 培训总览（本文档）
├── 01-理论基础/01-多智能体系统概论.md           # Part1 第一部分
├── 02-LangGraph框架/02-LangGraph深度应用.md   # Part1 第二部分
├── 03-LangSmith监控/03-LangSmith监控平台集成.md # Part1 第三部分
├── 04-企业级架构/04-企业级系统架构设计与实现.md   # Part2 第一、二部分
└── 05-应用实践/05-应用实践与部署运维.md         # Part2 第三、四部分
```

| 模块              | 对应文档           | 学习重点                     |
| ----------------- | ------------------ | ---------------------------- |
| 01 理论基础       | Part1 第一部分     | BDI 架构、协作机制、系统优势 |
| 02 LangGraph 框架 | Part1 第二部分     | 节点、边、状态、工作流构建   |
| 03 LangSmith 监控 | Part1 第三部分     | 全链路追踪、告警、性能优化   |
| 04 企业级架构     | Part2 第一、二部分 | 架构设计、技术实现、代码实践 |
| 05 应用实践       | Part2 第三、四部分 | 智能客服、部署、运维         |

## 实践项目

- **智能客服系统（核心项目）**：对话、知识检索、工单处理多智能体协作，覆盖多渠道接入、情感分析、VIP 优先级与全链路监控
- **内容创作平台**：创意策划、内容生成、质量审核、发布管理智能体的协同工作流
- **金融分析系统**：市场分析、风险评估、投资建议、合规监控智能体在高频数据与合规要求下的应用

## 快速开始

```bash
git clone <repository-url>
cd multi_agent_training
pip install -r requirements.txt
docker-compose up -d
```

按目录顺序学习对应文档章节，并完成每个模块的实践练习与项目开发。
