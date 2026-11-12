# 基础设施的回应：当 serving 开始读懂 harness

> **元信息**：本文是 [Agent 推理服务总纲](01_agent-workload-serving.md)的续篇，机制细节与性能数字以姊妹篇 [Evolving Infra 论文精读](02_evolving-infra.md)为准。上游核对基线：SGLang 本地 HEAD `f4de6abee6`（2026-09-28，含 PR #28403，2026-09-18 合入）；Evolving Infra 未开源，其机制以论文（arXiv 2611.12304 v1）为准。写作时点 2026-11-12。

[总纲](01_agent-workload-serving.md)从源码出发诊断了 Agent 负载下的四个连锁问题：KV 生命周期错配、调度语义失真、会话粘性、容量公式失效。诊断之后的问题是：基础设施侧回应了吗？回应到什么程度？Evolving Infra（arXiv 2611.12304）是迄今最完整的一份答卷，本篇把它与总纲的四连锁对上，核对上游引擎的差距，最后给出一份 PD 适配方案的对照地图。

## 一、三件套对四连锁

Evolving Infra 的三件套（对称引擎 + 共享 L2 池、迭代时间模型 + planner、branch count 目录）不是三个孤立优化，每个都对应总纲的一个诊断：

| 总纲诊断的连锁问题                                              | 基础设施的回应                                                                                                                            | 论文依据   |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| **KV 生命周期错配**（驱逐不知道 agent 还会不会用）              | branch count 目录：驱逐从「最近没用」改成「没人引用」：compaction 替换的历史、已结束 sub-agent 的上下文先走，等待工具的分支最后动         | §4.3、§5.3 |
| **调度语义失真**（请求该去哪，引擎说了不算）                    | KV-reuse-aware 路由：对每个 prefill 节点预测 TTFT（含取数暴露与副本代价），发给预测最小者，配合异步取数把 KV 搬过来，亲和与负载不再二选一 | §5.3、§3.3 |
| **会话粘性**（粘性路由的存在理由是 KV 在固定节点）              | 对称引擎 + 共享池抽掉了这个理由：任意节点可读任意 KV，节点换 phase 不动 KV，粘性从刚需变成可选项                                          | §4.1、§4.2 |
| **容量公式失效**（全输入长度当 prefill 工作量，复用越多估越歪） | 迭代时间模型在公式层分离 cached 与 uncached：prefill 时间跟着未缓存 token 与 query-key 对走，agent 复用越多估得越准                       | §5.1       |

映射里有一处要诚实标注：四连锁中「请求间语义」这块，论文只覆盖了分支的线性追加与 compaction，fork 的计数有了，但「一个分支等待另一个分支的结果」只体现为等待时长，分支间的语义依赖没有进模型。总纲诊断的深度，论文回应了大约四分之三。

## 二、上游差距：SGLang 离这还有多远

SGLang 在 2026-09-18 合入了运行时角色切换（PR #28403），是上游离 Evolving Infra 最近的一步。我们在本地 HEAD（`f4de6abee6`）核对了它的实现，结论：解决了「不重启」，没有解决「不丢 KV、不排空、跟 agent 信号走」。

|            | SGLang runtime role switch（PR #28403，已核对）                                                                          | Evolving Infra 对称引擎（论文口径）      |
| ---------- | ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| 切换前提   | 实例必须 fully idle（排空全部请求，role_switch.py:73-75）                                                                | 排空该节点在跑请求即可，其余节点不受影响 |
| 切换后 KV  | **整个前缀缓存清空**：radix 树 reset，HiCache 存储后端一并清（role_switch.py:253-282，docstring 注明防新角色误配旧前缀） | KV 留在共享 L2 池，切换全程可读          |
| 默认开关   | opt-in：需显式 `--enable-pd-role-switch`（role_switch.py:164-167）                                                       | 默认形态                                 |
| 失败处理   | 切换失败对实例致命，无原地回滚（role_switch.py:47）                                                                      | 论文未报告失败案例                       |
| 驱逐与路由 | 不变（LRU + 网关策略）                                                                                                   | 分支计数驱逐 + TTFT 预测路由             |
| agent 信号 | 无                                                                                                                       | branch ID / fork / compaction 版本       |

差距的根源与本文精读篇判断一一致：上游改的是「节点的角色怎么换」，Evolving Infra 改的是「角色与 KV 的耦合关系」。前者是运维改进，后者才是架构改动。

## 三、PD 适配方案的对照地图

把论文 §A.1/§A.4 盘过的方案与本篇核对结果放进同一张表（信号列指分配依据里有没有 agent 侧信号）：

| 方案                     | 换相方式                                   | 换相后 KV                          | 分配信号                                    | agent 信号                      |
| ------------------------ | ------------------------------------------ | ---------------------------------- | ------------------------------------------- | ------------------------------- |
| Dynamo planner           | 重启 worker（681–754 s 离线）              | 空                                 | 队列深度、KV 利用率                         | 无                              |
| Arrow                    | 原地换                                     | 不可达（34 次切换丢 55.5% 复用）   | 全输入长度估 TTFT（代码未公开，作者重实现） | 无                              |
| xLLM                     | 原地换，但与 KV 感知路由互斥（启动二选一） | 丢                                 | 单请求触发的逐节点翻转                      | 无                              |
| Llumnix                  | 原地换 + 请求迁移                          | 随请求迁移                         | 插值成本模型（H200 补测 −12.2%）            | 无                              |
| semi-PD                  | 引擎内 SM 分配，统一存储                   | 引擎内保留                         | TTFT/TPOT 目标                              | 无                              |
| MemServe / PPD           | 不换相（split 固定，PPD 重启换）           | 多轮 KV 保留                       | 前缀位置路由                                | 多轮会话                        |
| ThunderAgent / Continuum | 不换相                                     | 暂停 agent 保 KV（假设上下文只增） | 程序抽象 / TTL                              | 部分                            |
| KVTether                 | 不换相                                     | 生命周期状态机管理                 | harness 上下文操作                          | 有                              |
| **Evolving Infra**       | **原地 6.2 s（含排空）**                   | **共享池保留**                     | **在线迭代时间模型 + 显著性判据**           | **branch ID / compaction 版本** |

地图上有两个观察。其一，agent 信号列此前几乎空白，KVTether 与 Evolving Infra 是仅有的两个非空项，且都要求 harness 改造打点。其二，换相方式与 KV 去留两列高度相关：不重启的方案里只有 Evolving Infra 保住了 KV，这正是「对称引擎 + 共享池」这对组合的稀缺性所在。

## 四、总评与观察点

三个前提决定这份回应能走多远。**硬件代际**：共享池依赖 NVLink-C2C 的 host-GPU 带宽与大容量 host memory，论文自己用 ThunderAgent 的历史结论作了对照，PCIe 时代这套设计是否成立没有验证。**打点契约**：harness 必须上报 branch ID、fork/finish/compaction；基础设施与 agent 框架之间的这份契约，是它区别于纯引擎侧方案的分水岭，也是部署摩擦的来源。**数字口径**：全部为作者自报，单次运行，Arrow 为重实现。

三个值得跟踪的观察点：上游（SGLang/vLLM）会不会吸收 branch 信号，radix 树的 extra_key 机制离「按分支隔离/复用」只差一个语义升级；Mooncake/3FS 存储生态会不会实现「直读对端 L2」绕开 L3 三跳（vLLM 的 Mooncake connector 已经在 GPU 内存直注册的方向上）；以及 Evolving Infra 会不会开源，论文发布即公开了 trace 数据集，代码是下一个自然动作。

与本仓锚点：四连锁的诊断见 [总纲](01_agent-workload-serving.md)；三件套的机制与数字见 [Evolving Infra 论文精读](02_evolving-infra.md)；分支计数的复用语义与本仓 [KV 复用总纲](../kv_cache/06_kv_reuse/01_kv-reuse-overview.md) §六 的「可声明接口」展望互为印证；HiCache 的 L1/L2/L3 分层与 Dynamo KV 事件机制分别见 [HiCache 深入详解](../sglang/hicache_deep_dive.md) 与 [Native KV Offloading](../vllm/module_analysis/native_kv_offloading.md)。

## 参考资料

- [Evolving Infra: Agent-Adaptive LLM Serving at Scale](https://arxiv.org/abs/2611.12304)（arXiv 2611.12304 v1，机制与数字来源）
- [SGLang PR #28403: runtime role switching](https://github.com/sgl-project/sglang/pull/28403)（上游核对对象，本地 HEAD 已含）
- [Arrow: adaptive scheduling for disaggregated LLM inference](https://arxiv.org/abs/2505.11916)
- 本仓库：[Agent 推理服务总纲](01_agent-workload-serving.md) · [Evolving Infra 论文精读](02_evolving-infra.md) · [KV 复用总纲](../kv_cache/06_kv_reuse/01_kv-reuse-overview.md)
