# LangGraph 框架开发指南

本文档聚焦于 LangGraph 框架的核心概念与实际应用，旨在帮助开发者快速理解并使用该框架构建复杂的有状态多智能体（Multi-Agent）系统。

## 1. LangGraph 介绍

LangGraph 是由 LangChain 团队推出的一个用于构建有状态、多智能体应用程序的强大库。

它扩展了 LangChain Expression Language (LCEL) 的能力，通过引入图（Graph）计算模型，完美解决了传统 LLM 应用在复杂逻辑处理中的瓶颈。LangGraph 的核心优势在于它能够轻松定义具有**循环（Cycles）**和**持久性状态（State Persistence）**的工作流，这使得它非常适合开发需要多轮推理、自我反思（Reflection）以及多智能体协同工作的复杂 Agent 系统。

相比于线性的工作流编排，LangGraph 允许开发者以节点（Nodes）和边（Edges）的形式直观地定义智能体的行为路径和决策条件，从而实现更高自由度与可控性的应用架构。

---

## 2. 实践总结

本节汇总了我们基于 LangGraph 框架进行探索与开发的实战案例及总结文档。

- [LangGraph 简介](langgraph_intro.md) - 深入解析 LangGraph 的核心架构与设计理念，提供入门级的实战指南，帮助开发者快速掌握图工作流的构建方式。
- [AI 客服系统实战](aics.ipynb) - 以一个完整的 AI 客服系统为例，演示了如何利用 LangGraph 的状态管理和循环路由机制，实现支持多轮对话、工单分发与状态回滚的复杂交互式 Notebook 实践。
