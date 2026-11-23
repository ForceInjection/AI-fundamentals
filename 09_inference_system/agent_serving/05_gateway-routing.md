# 引擎之上的那一层：SGLang Gateway 与 vLLM Router 源码对照

> **元信息**：SGLang sgl-model-gateway 核对基线 `f4de6abee6`（与[基础设施的回应](03_infra-response.md)一致）；vLLM 生态的 Rust 路由器核对基线 `e8ce7ea`（仓库内 crate 名 `vllm_router_rs`，版本 0.1.17）。均为 2026-11-23 的静态源码核对，无实测数字。文中 `gateway:` 前缀指 sgl-model-gateway，`router:` 前缀指 vLLM 路由器；引用路径相对各自仓库根。**上游动态**：同日复查发现，sglang main 已于 2026-11-16 移除 sgl-model-gateway（PR #50732），继任者是瘦身版 sgl-router 加实验性的 sgl-kv-indexer（详见 §2.3 与 §六）；本文对 gateway 的深读基于移除前最后一个完整形态，机制分析仍然成立，选型前请先读 §六。

[总纲](01_agent-workload-serving.md)诊断 Agent 负载时留过一个缺口：会话粘性。同一会话第 N 轮请求进来，上一轮几十 K token 的上下文 KV 还躺在某个实例的显存里，路由器把请求派给谁，决定这轮是两百毫秒的 TTFT 还是十几秒的全额重算。总纲当时的结论是「前缀感知已有，会话粘性靠应用层」：引擎内的 radix 树只认前缀不认会话，生产上靠网关按 session id 做一致性哈希，或者干脆接受外置 KV 层来解耦。

几个月过去，这一层动得很快。SGLang 把 sgl-router 升级成了 sgl-model-gateway（本文简称 gateway）：控制面加数据面，会话、响应、聊天历史都可以收拢到路由器集中存储。vLLM 生态的 Rust 路由器（本文简称 router）加进了两个新策略：kv_aware 让引擎上报 KV 事件、路由器维护精确驻留索引，sticky_least_loaded 内建会话粘性与 finish_session API，另有一个 opt-in 的程序级调度层。本文把两份源码放到同一张桌上：策略怎么选、会话状态放哪、PD 分离怎么路由、多副本状态怎么同步，最后对照 SOSP 2026 上 Bedrock 的全球 WLM，看开源网关与它之间隔着什么。

## 一、路由层在栈里的位置

vLLM 生态这条线走了三步：Rust 路由器（单进程网关，本文的 router）→ production-stack（Helm 交付的 K8s 部署栈）→ llm-d（K8s 原生分布式推理，把路由下沉到 Gateway API inference extension 与 EPP）。SGLang 那条线走了一个来回：sgl-router 升级为 gateway 0.3.2（控制面加数据面的大网关），2026-11-16 又被从 main 移除（PR #50732，删除 403 个文件、约 12.6 万行），回到瘦身版 sgl-router，另带一个实验性的 sgl-kv-indexer。两边的定位句都写得直白：router 自述「为 vLLM 大规模部署提供高级负载均衡与 prefill/decode 分离支持」（router README.md:6）；gateway 自述「大规模 LLM 部署的路由控制面与数据面……针对 SGLang serving 深度优化」（gateway README.md:3）。

「深度优化」是如实的。gateway 的 CLI 有 `--backend vllm/trtllm/anthropic` 三个选项，运行时全都会打印警告回退到 regular routing（gateway main.rs:1252-1259）；worker 元数据里虽有 `RuntimeType::{Sglang, Vllm, External}` 标签（gateway core/worker.rs:474-483），真正适配的只有 SGLang。反方向的依赖也不存在：gateway 不依赖 sglang 仓库 `rust/` 目录下任何 crate，树实现是自带的（gateway Cargo.toml:86-91）。所以「双引擎网关」这个说法在两边都不成立：router 只懂 vLLM，gateway 只懂 SGLang，跨引擎是 llm-d 那一层的事。

形态上，gateway 比 router 多了一层控制面：worker 注册校验、能力发现、后台健康检查与负载监控、可选的 K8s 服务发现，加一个 `--enable-igw` 的多模型网关模式（多个 router 实例同时跑、按模型施加各自的策略）。router 的对应物是 service_discovery 加一组路由处理器，没有独立控制面。数据面两边都覆盖 HTTP 与 gRPC，gateway 额外有一个 OpenAI 路由，把请求代理到远端厂商（OpenAI、xAI、Gemini 及其他 OpenAI 兼容服务），并托管 responses/conversations 存储。

栈的位置给了这一层别处没有的视角：引擎只见自己的 KV，KV 存储层只见自己的数据，只有路由器同时看得到全部 worker。总纲缺口清单里「KV 感知路由不跨引擎、不跨存储层」说的就是这里——2026 年末，这个判断依然成立，但两边都在往路由器里塞更多的事实源。

## 二、路由策略全景：趋同的门控，分叉的树

### 2.1 策略清单

两边各注册了八个策略，交集大得像一个模子：

| gateway（policies/factory.rs:77-91） | router（policies/factory.rs:80-92） |
| ------------------------------------ | ----------------------------------- |
| random                               | random                              |
| round_robin                          | round_robin                         |
| power_of_two                         | power_of_two                        |
| cache_aware                          | cache_aware                         |
| bucket                               | —                                   |
| manual                               | —                                   |
| consistent_hashing                   | consistent_hash                     |
| prefix_hash                          | —                                   |
| —                                    | kv_aware                            |
| —                                    | sticky_least_loaded                 |
| —                                    | rendezvous_hash                     |

相同的六个是教科书策略；分叉的五个各有来历。gateway 的 bucket 按请求字符数分桶、把不同体量的请求派给不同 prefill worker，每 5 秒按累计字符再平衡一次桶边界（gateway policies/bucket.rs:335-414）——它本质是 PD 场景的 prefill 分配器。manual 是把路由键到 worker 的映射交给外部管理，带 4 小时空闲逐出（gateway policies/manual.rs:65-136）。prefix_hash 是 cache_aware 的轻量替代：取前 256 token 做哈希进 HashRing，超载回退最低负载（gateway policies/prefix_hash.rs:1-27）。router 侧的三个新面孔，2.2 和 2.3 节展开。

注意 CLI 暴露并不完整：gateway 命令行只接受六种策略，bucket 与 consistent_hashing 不在内（gateway main.rs:150）；router 的 CLI 全量暴露八种（router main.rs:125）。

### 2.2 cache_aware：两边独立趋同的阈值门控

两边的 cache_aware 都不是加权打分，而是同一个三段门控。以 router 的实现为例（router src/policies/cache_aware.rs:365,446-447）：

```rust
let use_prefix = !imbalanced && best_rate > self.config.cache_threshold as f64;
// imbalanced 要求同时满足：
// max-min > balance_abs_threshold && max > min * balance_rel_threshold
```

负载失衡时选最低负载的 worker；均衡且最高前缀匹配率过线时，选匹配最高的 worker（平局按负载裁决）；否则选最低负载。gateway 的实现逐条对应（gateway policies/cache_aware.rs:412-413），连默认值都几乎一样：

| 参数            | gateway 库默认 / CLI 默认 | router 库默认 / CLI 默认 |
| --------------- | ------------------------- | ------------------------ |
| cache_threshold | 0.5 / 0.3                 | 0.5 / 0.3                |
| 绝对失衡阈值    | 32 / 64                   | 32 / 64                  |
| 相对失衡阈值    | 1.1 / 1.5                 | 1.1 / 1.5                |
| 驱逐间隔        | 30s / 120s                | 30s / 120s               |

两个独立团队把「亲和与均衡怎么折中」收敛到同一个形状：亲和是概率性的，均衡是兜底的，一条线画在中间。[总纲 4.1 节](01_agent-workload-serving.md)引用的 `cache_threshold`（当时写 0.5）需要精确化：那是库默认，命令行默认是 0.3，而且门控先看负载失衡再看匹配率，顺序不能反。

### 2.3 树的两代：路由器猜，还是引擎报告

真正的分叉在树。gateway 的 cache_aware 背后是一棵 2310 行的字符级多租户压缩前缀树（gateway policies/tree.rs）：以 `{pool}::{model}` 为键建树、每个 worker 作为树内一个 tenant 插入自己的请求文本（gateway policies/cache_aware.rs:92-101,471），root 分片 32 供并发读写，tenant 按 epoch LRU 逐出。这棵树的输入是路由器自己见过的请求文本，它是对「哪些前缀大概在哪个实例上」的近似，不是引擎的记账。router 的 cache_aware 同构，源码注释把这点写得很明白：「maintains an approximate radix tree from request history … This history is not an engine residency index」（router src/policies/cache_aware.rs:21-24）。

router 的 kv_aware 换了事实源：vLLM 引擎通过 ZMQ 上报 KV 事件（BlockStored / BlockRemoved，SHA-256 块哈希），路由器据此维护一棵**精确的驻留块索引**（router src/kv_index/indexer.rs:1;wire.rs:61-65）。策略名下的注释是「Exact KV-placement and live-load aware worker selection」（router src/policies/kv_aware.rs:1）。从「路由器猜」到「引擎报告」，这一步跨过了总纲缺口清单里「路由器只见自己的近似视图」的问题：驱逐、重启、淘汰，引擎记账了什么路由器就知道什么。配套的工程细节也做足了：KV 事件源掉线后进入显式的恢复状态机，可被 `--kv-event-strict-recovery` 收紧，并暴露 `vllm_router_kv_source_recovery_state` 等专属指标（router main.rs:223-226）。

gateway 侧的树是自建近似，但上游的答案在本文核对后浮出水面：继 gateway 移除后，新 sgl-router 附带实验性的 sgl-kv-indexer，自述「SGLang KV-cache 块的元数据服务」，记录「每个内容寻址块当前在哪个 worker、哪个存储层」，让路由器查询可能的缓存命中而不搬 KV 数据本身；架构是「SGLang worker → ZMQ PUB → bridge → gRPC → 进程内 indexer」，事件为 BlockStored / BlockRemoved / AllBlocksCleared，进程内内存态、重启即失（sgl-router/sgl-kv-indexer/README.md，main 分支）。这与 router 的 kv_aware 机制同构：引擎 ZMQ 上报块事件，路由器维护内存精确索引。两代路线至此收敛——第一代（路由器自建近似树）在两边都有完整实现且仍在策略清单里，第二代（引擎上报事实源）也已两边到齐，本文深读的 gateway 是第一代在 SGLang 侧的最后完整样本。

## 三、会话与状态：三种放法

总纲说会话粘性「靠应用层」。现在路由层自己提供了三种放法，两边各占其二：

**放法一：无状态 header 粘性。** 同一个 session id 哈希到固定 worker，状态就是一张小映射表。router 的 sticky_least_loaded 从六个 header 里取会话键，优先级 `x-session-id` → `x-user-id` → `x-tenant-id` → `x-correlation-id` → `x-request-id` → `x-trace-id`（router src/policies/hash_key.rs:13-20），TTL 默认 2 小时、每 60 秒清扫一轮（router src/policies/sticky_least_loaded.rs:41,47），并配了一个 `POST /finish_session?session_id=` 端点：调用后向 default/per-model/prefill/decode 四类策略广播清账，把该会话从映射里移除、对应副本的活跃会话数减一（router src/policies/registry.rs:283-299;sticky_least_loaded.rs:301）。gateway 的对应物是 manual 策略加 `X-SMG-Routing-Key` header（gateway routers/header_utils.rs:8-9），空闲 4 小时逐出（gateway policies/manual.rs:65-74）。

**放法二：会话状态集中到路由器。** gateway 把 OpenAI Conversations 风格的 `/v1/conversations` REST API 做进了网关（gateway server.rs:562-573），会话、响应、聊天历史三类状态按 conversation_id 集中存取，存储可插拔：内存、Oracle ATP、PostgreSQL、Redis（gateway config/types.rs:68-74）。这样跨模型、跨 MCP 工具循环的上下文可以在路由器层续接，不再依赖某个具体实例活着。代价是路由器从无状态组件变成了有状态服务。这个代价最终没有换来留存：gateway 移除后，新 sgl-router 自述支持 session affinity，而 Conversations 与 Responses API 在新形态中仍返回 501 未实现（PR #50732 口径）。

**放法三：程序级原语。** router 里有一个 opt-in 的 program_scheduling 模块（`--enable-program-scheduling`，router src/program_scheduling/mod.rs:2-5）：给 Agent 任务分配 Program 身份、按 RequestPool 做准入、以 Progress-TTL 追踪进度、配全局队列放置。README 对它的定位是「与请求级负载均衡策略正交的 Program（智能体任务）级调度层」。这回应的是总纲缺口清单第一条「引擎无会话原语：session id → 亲和/配额/分组抢占只能应用层拼」，虽然还只是 opt-in 的雏形。

多副本是状态问题的另一半。gateway 的 cache_aware 树默认是单实例内存态，多副本部署时各副本视图不同；`--enable-mesh` 走 smg-mesh 的 P2P gRPC 同步，同步粒度是树操作（Insert/Remove op），mesh 端口 39527（gateway policies/cache_aware.rs:109-110,473-484;main.rs:617-631）。router 侧，kv_aware 的事实源在引擎，副本各自收同一批 KV 事件即天然一致；cache_aware（自建近似树）则和 gateway 一样没有同步。自建近似树需要同步协议，引擎上报的索引不需要：事实源放在哪，多副本的复杂度就落在哪。

## 四、PD 分离：双发、独立策略槽与失败分账

两边对 PD 分离的支持都是一等公民，形态高度相似。worker 类型上，两边的枚举一字不差：`WorkerType::{Regular, Prefill{bootstrap_port}, Decode}`（gateway core/worker.rs:515-525;router src/core/worker.rs:234-243）。策略上，两边都允许给 prefill 与 decode 配独立的路由策略（gateway main.rs:197-212 的 `--prefill-policy/--decode-policy`;router PolicyRegistry 的 prefill_policy/decode_policy 槽，registry.rs:30-34）。

分工方式略有不同。gateway 是双发：prefill 与 decode 各派一次，执行 `execute_dual_dispatch`，向 prefill 侧注入 bootstrap_host/bootstrap_port/bootstrap_room 与 disagg_prefill_dp_rank，让 prefill 把 KV 传输的目的地告诉 decode（gateway routers/http/pd_router.rs:237-321）。router 是 bootstrap 传递：prefill worker 创建时就携带 bootstrap_port（K8s 场景从 pod 注解 `vllm.ai/bootstrap-port` 读得，service_discovery.rs:53,163-168），同样是为了让两段对齐 KV 传输。

错误处理上 gateway 多想了一步：双发的失败按 prefill 与 decode 各自的结果记账，decode 挂了不把错误算到健康的 prefill 头上（gateway routers/http/pd_router.rs:80-84）。2.1 节的 bucket 策略也该放回这个语境看：按请求字符数分桶派 prefill，就是给 PD 拓扑的 prefill 池做准入分配。

router 还多覆盖了两个拓扑：EPD 三段编排（`--epd-config`，编码器-预填充-解码，多模态模型用，router main.rs:109-110）和 DP 感知 worker（单机多卡数据并行的 worker_url 带 DP rank 后缀、选路后剥离，router core/worker.rs:538;routers/http/router.rs:2086-2092）。

## 五、可靠性：熔断、重试与观测

把两边的可靠性层摆在一起，趋同程度最高的部分出现了：

| 机制       | gateway                                                                                 | router                                 |
| ---------- | --------------------------------------------------------------------------------------- | -------------------------------------- |
| 熔断状态机 | Closed/Open/HalfOpen 三态，CAS 迁移（core/circuit_breaker.rs:148-259）                  | 同三态（core/circuit_breaker.rs:9-23） |
| 熔断默认值 | core 层 5 失败/2 成功/30s；config 层 10/3/60s+120s 窗口（两层不一致，以 config 层生效） | 5 失败/2 成功/30s，60s 窗口            |
| 重试状态码 | 408/429/500/502/503/504（core/retry.rs:10-20）                                          | 同（core/retry.rs:9-18）               |
| 退避       | 50ms × 1.5，cap 30s，jitter ±20%（config/types.rs:412-421）                             | 同（config/types.rs:1281-1291）        |
| 重试次数   | 默认 5                                                                                  | 默认 5，`--disable-retries` 压为 1     |

差异在细节的成熟度。router 的熔断不计 4xx：「Client errors are not worker failures - only server errors should count against the circuit breaker」（router src/routers/http/router.rs:2049-2052），kv_aware 路径重试时把失败 worker 记入排除名单（router.rs:2054-2056），选路前统一过滤健康与熔断状态（policies/mod.rs:264-266）。gateway 的服务发现断线退避 1s→300s 重连（service_discovery.rs），router 断线固定间隔重启 watcher（service_discovery.rs:396）；两边都以 kube-rs watch Pod、label 全等匹配、静态模式走 `--worker-urls`。

两个单点特性值得记录。gateway 的 metrics_aggregator 不生产指标，只做搬运：抓取各 worker 的 /metrics 文本、打上 worker_addr 标签合并成 `/engine_metrics`（gateway core/metrics_aggregator.rs:15-40），gateway 自身的指标约 46 个函数覆盖 HTTP、熔断、重试、发现各层。router 的独门是 WASM 中间件：`--wasm-middleware` 挂载一个 WASM 组件到 `POST /v1/chat/completions`，WIT 接口约定 continue/reject/modify 三种动作，输入输出各限 10MB、内存 64MB、执行 100ms，失败即拒绝（fail-closed）（router src/wasm_middleware.rs:1-45;wit/router-middleware.wit:1-35）——给租户自定义路由逻辑开了一个带资源围栏的口子。

## 六、总评

趋同的部分说明这一层的工程共识已经定型：cache_aware 的阈值门控连默认值都一样；PD 的 worker 三分类、独立策略槽、bootstrap 传递两边全同；熔断与重试的参数表几乎可以互相替换。做推理网关的人现在有一份事实上的参考实现清单。

分叉的部分才是选型依据，三个问题各有一个源码级答案。会话状态放哪：gateway 集中存储（Conversations API + 可插拔后端）换来了跨模型续接，代价是有状态网关；router 的无状态粘性加 finish_session 换来了部署简单，代价是会话语义止步于「路由到同一实例」。前缀事实源放哪：kv_aware 的引擎上报路线在精确性上胜出，但要求引擎配合发事件；gateway 的自建树零侵入，多副本时要么接受视图分叉、要么开 mesh。Agent 原语放哪：router 的 program_scheduling 有了 Program/RequestPool/Progress-TTL 的骨架，gateway 把宝押在 Conversations 状态上，两条路线各持一块对方没有的拼图。

对照 Bedrock 的 WLM（[精读](../sosp2026/references/02-bedrock-fm-serving.md)），层差依然清晰：开源网关处理的是单集群内的「请求到实例」，WLM 处理的是跨 30+ 区域的「请求到区域」，前者没有的准入分片、有界不一致对账、跨区 cache 局部性权重，是量级不同的问题。但 WLM 用粘性路由把前缀复用成功率做到 99.94%（厂商口径）、把跨 1000 前缀请求的 TTFT 组件降 45% 的数字，给这一层的价值提供了量化锚点：路由层把亲和做对，收益就是这个量级。

三个观察点：CS2680 的 routing 与 load balancing 讲次（2026-11-23）slide 截至核对时尚未挂出，挂出后可与本文对照；llm-d 的 EPP 把 session-affinity 路由列为社区议题（总纲 3.3 节口径），进展值得跟踪；sgl-kv-indexer 目前自述 experimental、进程内内存态、重启即失，从软状态走向生产可用的持久化与分片，是 SGLang 侧第二代路线的下一个里程碑。

折扣清单：本文全部结论来自静态源码核对，无负载实测；gateway 的 `--backend vllm/trtllm/anthropic` 为占位未实现，「跨引擎」讨论以 llm-d 为准；两边的 cache_aware 各有 CLI 与库内两层默认值，本文表格分别标注，引用时注意口径；program_scheduling 为 opt-in 模块，默认关闭；Bedrock 数字为 AWS 厂商口径。

## 源文件索引

| 仓库              | 文件                                   | 关键内容                                                            |
| ----------------- | -------------------------------------- | ------------------------------------------------------------------- |
| sgl-model-gateway | policies/cache_aware.rs                | 门控判定 :412-413；树键隔离 :77-101；mesh 同步挂点 :109-110,473-484 |
| sgl-model-gateway | policies/tree.rs                       | 字符级多租户压缩前缀树，root 分片 32、tenant LRU                    |
| sgl-model-gateway | policies/bucket.rs                     | 按字符数分桶，5s 边界再平衡                                         |
| sgl-model-gateway | policies/manual.rs                     | X-SMG-Routing-Key 映射与 4h 逐出                                    |
| sgl-model-gateway | routers/http/pd_router.rs              | 双发与 bootstrap 注入 :237-321；失败分账 :80-84                     |
| sgl-model-gateway | server.rs                              | /v1/conversations 端点 :562-573                                     |
| sgl-model-gateway | core/circuit_breaker.rs, core/retry.rs | 三态熔断；重试状态码与退避                                          |
| sgl-model-gateway | core/metrics_aggregator.rs             | worker 指标聚合转发                                                 |
| sgl-model-gateway | main.rs                                | 策略 CLI :150；PD 配置 :197-212；backend 占位警告 :1252-1259        |
| vllm_router_rs    | policies/cache_aware.rs                | 门控判定 :365,446-447；近似树自述 :21-24                            |
| vllm_router_rs    | policies/kv_aware.rs, kv_index/        | 引擎 KV 事件精确驻留索引；来源恢复状态机                            |
| vllm_router_rs    | policies/sticky_least_loaded.rs        | 六 header 粘性、TTL 2h、rendezvous 平局裁决                         |
| vllm_router_rs    | policies/registry.rs                   | finish_session 广播 :283-299；prefill/decode 策略槽 :30-34          |
| vllm_router_rs    | program_scheduling/                    | Program 身份/RequestPool/Progress-TTL（opt-in）                     |
| vllm_router_rs    | core/worker.rs                         | WorkerType 三分类 :234-243；DP 感知 :538                            |
| vllm_router_rs    | routers/http/router.rs                 | 4xx 不计熔断 :2049-2052；kv_aware 排除名单 :2054-2056               |
| vllm_router_rs    | wasm_middleware.rs, wit/               | WASM 中间件与资源围栏                                               |

## 参考资料

- 本系列：[Agent 推理服务总纲](01_agent-workload-serving.md) · [Evolving Infra 论文精读](02_evolving-infra.md) · [基础设施的回应](03_infra-response.md) · [约束解码的性能账单](04_constrained-decoding-engines.md)
- sgl-model-gateway：基线 `f4de6abee6` 的 sglang 仓库内目录（上游 main 已于 2026-11-16 移除，见 [PR #50732](https://github.com/sgl-project/sglang/pull/50732)；官方文档 [docs.sglang.io](https://docs.sglang.io/advanced_features/sgl_model_gateway.html) 仍保留旧 URL 重定向）
- [sgl-router 与 sgl-kv-indexer](https://github.com/sgl-project/sglang/tree/main/sgl-router)（sglang main 现行形态，slim 路由器 + 实验性 KV 元数据服务）
- vLLM Rust 路由器（基线 `e8ce7ea`，crate `vllm_router_rs` 0.1.17）· [production-stack](https://github.com/vllm-project/production-stack) · [llm-d](https://github.com/llm-d/llm-d)
- Bedrock: FM Serving at Global Scale（SOSP 2026，DOI 10.1145/3830418.3843855；精读见 [sosp2026/references](../sosp2026/references/02-bedrock-fm-serving.md)）
