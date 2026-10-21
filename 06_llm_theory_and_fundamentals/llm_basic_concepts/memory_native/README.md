# 记忆原生模型

把记忆从“外部上下文或检索”变成“模型内部状态”的架构与训练论文。

- **[当记忆成为模型状态：MemT-1.0 的架构、训练与边界](memt_1_0_memory_native.md)** — local/hyper memory、GDN-2 更新、MoS、34B token 训练，以及 11 项记忆基准、SWE-bench 和效率证据；附源码核对、许可与部署边界。
- **[EdgeRock-4B 拆解：把 KV Cache 变成固定容量记忆](edgerock-4b-constant-memory.md)** — 用 1024 token 局部窗口加 1024 条 RMA 锚点替换 8 层全注意力，历史固定 97.76 MiB/序列；附机制、四阶段转换、AIME/吞吐数据与口径边界。

## 相关

- **[AI 智能体记忆系统](../../../08_agentic_system/memory/README.md)** — Agent 侧的外部记忆系统：MemoryOS、Mem0、LangChain 的存储、检索与更新。
