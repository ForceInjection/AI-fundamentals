# Agent 推理服务

Agent 流量正在取代 Chat 成为主要负载，约束解码则给每一步生成加上一层「输出必须合法」的税。本目录收这两条线的深读：一条讲负载形态变了之后推理系统哪里会失效，一条讲结构化输出的性能账怎么算。

## 内容导航

- **[当 Agent 流量成为推理系统的主要负载](agent-workload-serving.md)**：KV 生命周期错配、调度语义失真、会话粘性、容量公式失效四个连锁问题，以及「保留 vs 重算」的系数变化；vLLM（`43d691ec6b`）与 SGLang（`f7101b0ae6`）源码级现状。
- **[约束解码的性能账单：vLLM 与 SGLang 的结构化输出实现拆解](constrained-decoding-engines.md)**：编译 / 每步 / 交互三笔账单、双引擎逐项对照，与 jump-forward 重分词差异。

## 相关链接

- [KV Cache 技术体系](../kv_cache/README.md)：KV 生命周期与复用机制的原理层
- [vLLM 推理系统](../vllm/README.md) · [SGLang 推理引擎](../sglang/README.md)：两篇引擎源码分析的对象
