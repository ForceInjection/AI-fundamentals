# 线性注意力与推理系统

delta-rule 线性注意力（Qwen3-Next/3.5/3.8 的 Gated DeltaNet、Kimi Linear/K3 的 KDA）正在成为长上下文模型的主流选择。本目录从机制到系统，拆解这一类「没有 KV Cache」的模型落地推理引擎后引发的连锁变化。

## 内容导航

- **[总览：没有 KV Cache 的模型](linear-attention-serving.md)** — 谱系、三个被打破的系统假设（prefill 串行 / 前缀缓存重写为状态检查点 / 双账本）、vLLM 与 SGLang 双引擎实现对照。**从这里读起。**
- **[01 · KDA 机制拆解](01-kda-mechanism.md)** — 从 DeltaNet 到通道级门控：delta rule 递推、DPLR 特化、衰减门替代 RoPE，机制层面的地基。
- **[02 · chunkwise 串行 prefill 的调度](02-chunkwise-scheduling.md)** — 分清模型级串行与调度级切块：引擎已解决的与真正缺的原语。
- **[03 · 状态语义深水区](03-state-semantics.md)** — 投机解码的状态回滚、beam 分叉的 COW 关闭、检查点重放一致性、TP 按头切分后的本地计算与通信边界。

## 阅读顺序

机制（01）→ 调度（02）→ 状态语义（03），每篇建立在前一篇之上；总览可作任何一篇的入口。

## 相关链接

- [post-KV-cache 篇](../kv_compression/01-post-kv-cache-era.md)——本文系列的出发点（§3 chunkwise 串行、§5 状态检查点两块「需要解决」）
- [Agent 负载篇](../agent_serving/agent-workload-serving.md)——会话型负载与容量公式（状态池硬顶的推论见其 §2.4）
