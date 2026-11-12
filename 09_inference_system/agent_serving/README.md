# Agent 推理服务

Agent 流量正在取代 Chat 成为主要负载，约束解码则给每一步生成加上一层「输出必须合法」的税。本目录收这两条线的深读：一条讲负载形态变了之后推理系统哪里会失效、基础设施如何回应，一条讲结构化输出的性能账怎么算。

## 内容导航

- **[当 Agent 流量成为推理系统的主要负载](01_agent-workload-serving.md)**：KV 生命周期错配、调度语义失真、会话粘性、容量公式失效四个连锁问题，以及「保留 vs 重算」的系数变化；vLLM（`43d691ec6b`）与 SGLang（`f7101b0ae6`）源码级现状。
- **[Phase 切换从 681 秒到 6.2 秒：Evolving Infra 论文精读](02_evolving-infra.md)**：第一个把 harness 信号（branch ID、fork/compaction 事件）接入 planner、router、驱逐的 serving 框架——对称引擎原地换 phase、共享 L2 池 RDMA 直读、KV 目录按活跃分支计数驱逐；对 Dynamo 吞吐 1.10–2.73×、E2E 延迟降最高 4.12×，附 Arrow 重实现、closed-loop 单次运行等七条口径声明。
- **[基础设施的回应：当 serving 开始读懂 harness](03_infra-response.md)**：总纲四连锁与三件套的逐项映射（诚实标注约四分之三覆盖），SGLang PR #28403 role switch 源码核对（opt-in、须排空、每次翻转清空前缀缓存），以及 Dynamo/Arrow/xLLM/Llumnix/semi-PD/MemServe/PPD/ThunderAgent/KVTether 的 PD 适配方案对照地图。
- **[约束解码的性能账单：vLLM 与 SGLang 的结构化输出实现拆解](04_constrained-decoding-engines.md)**：编译 / 每步 / 交互三笔账单、双引擎逐项对照，与 jump-forward 重分词差异。

## 相关链接

- [KV Cache 技术体系](../kv_cache/README.md)：KV 生命周期与复用机制的原理层
- [vLLM 推理系统](../vllm/README.md) · [SGLang 推理引擎](../sglang/README.md)：两篇引擎源码分析的对象
