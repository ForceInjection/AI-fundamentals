# 云原生高性能分布式 LLM 推理框架 llm-d 介绍

`llm-d` 是一个开源的、`Kubernetes` 原生的分布式 `LLM` 推理服务栈（serving stack），目标是把单机推理引擎扩展成生产级的集群化服务。项目由 `Red Hat`、`Google Cloud`、`IBM Research`、`CoreWeave` 和 `NVIDIA` 共同发起，现为 `CNCF` sandbox 项目，当前已发布 v1.0（本文核对日 2026-11-05，版本信息以官方发布页为准）。

它不重新造推理引擎：真正执行模型计算的仍是 `vLLM`、`SGLang` 这类模型服务器。`llm-d` 补的是它们之外的部分——请求路由、前缀缓存感知调度、预填充/解码分离、`KV Cache` 分层卸载、多机模型服务编排，以及一套经过测试的部署路径。官方描述它为「production deployments on Kubernetes 优化的高性能分布式推理服务栈」，在 `NVIDIA`、`AMD`、`Intel GPU` 和 `Google TPU` 上都验证过。

## 目录

- [云原生高性能分布式 LLM 推理框架 llm-d 介绍](#云原生高性能分布式-llm-推理框架-llm-d-介绍)
  - [目录](#目录)
  - [1. 它要解决什么问题](#1-它要解决什么问题)
  - [2. 核心机制](#2-核心机制)
    - [2.1 入口：Gateway API + 推理扩展](#21-入口gateway-api--推理扩展)
    - [2.2 EPP：端点选择器](#22-epp端点选择器)
    - [2.3 前缀缓存感知路由](#23-前缀缓存感知路由)
    - [2.4 P/D 分离](#24-pd-分离)
  - [3. KV Cache 分层与 offloader](#3-kv-cache-分层与-offloader)
    - [3.1 分层结构](#31-分层结构)
    - [3.2 offloader 的三种实现](#32-offloader-的三种实现)
    - [3.3 LMCache 一节](#33-lmcache-一节)
  - [4. 模型服务编排：基于 LWS](#4-模型服务编排基于-lws)
  - [5. Well-Lit Paths：官方部署路径](#5-well-lit-paths官方部署路径)
  - [6. 性能数据与口径](#6-性能数据与口径)
  - [7. 总评](#7-总评)
  - [参考资料](#参考资料)

---

## 1. 它要解决什么问题

单机 `vLLM` 或 `SGLang` 跑到生产规模，会遇到几层问题，`llm-d` 的每个组件基本都对应其中一层：

**路由层：** `Kubernetes` 原生的 `Service` 负载均衡按连接分发请求，不看模型服务器手里有什么。两条 prompt 前缀相同的请求被分到不同实例，各自重新算一遍相同的 `prefill`，`GPU` 在重复劳动上空烧。多副本场景下，`KV Cache` 的命中率完全靠运气。

**资源形态层：** `prefill` 是计算密集型，`decode` 是访存密集型，两者对硬件的要求相反，混在同一实例里互相拖累。长上下文和多实例并存时，这种错配会被放大。

**`KV Cache` 层：** `HBM` 装不下的 `KV` 块被驱逐后直接丢弃，下一轮请求命中不了；跨实例之间没有共享机制，同一个会话换个副本就要全部重算。

**编排层：** 多机张量并行的推理服务是一个「要么全起、要么全废」的组，`Deployment` 和 `StatefulSet` 的单 Pod 语义管不住它，故障恢复需要整组原子操作。

这些问题没有一个是新问题，但 `llm-d` 之前，它们散落在各家自研网关和调度器里，没有形成开源的标准件。

## 2. 核心机制

### 2.1 入口：Gateway API + 推理扩展

`llm-d` 的流量入口是一个符合 `Gateway API Inference Extension`（社区里常缩写为 `GAIE`）的高性能 `L7` 代理。`Gateway API Inference Extension` 是 `Kubernetes` 社区为 AI 推理流量定义的标准扩展——它给 `Gateway` 增加了「按模型服务池路由」和「把推理负载特征暴露给路由器」的能力，`llm-d` 是这个扩展的主要实现方之一。

### 2.2 EPP：端点选择器

路由决策的核心组件是 `EPP`（`Endpoint Picker`，端点选择器）。官方架构文档对它的定义是：

- 一个可扩展的 **Filter-Score-Pick 流水线**，先过滤候选端点，再逐个打分，最后选出目标；
- 依据实时指标、`KV Cache` 亲和性（`KV-cache affinity`）和已配置的策略对模型服务器 Pod 打分；
- 支持用声明式 `YAML` 配置打分插件，可以按需替换打分逻辑。

EPP 是 `llm-d` 与传统网关的本质区别：传统网关知道「哪个副本活着」，EPP 知道「哪个副本手里已经有这条 prompt 的前缀缓存、当前负载多少」。

### 2.3 前缀缓存感知路由

EPP 的打分链路里有一类前缀缓存感知的 scorer：通过跟踪各模型服务器的 `KV Cache` 事件（官方描述为「event-driven tracking of cache state across all model servers」），路由器能知道哪台服务器缓存了请求的前缀，把请求发过去就能省掉重复 `prefill`。官方把这条路径称为「Precise Prefix-Cache Aware Routing」，提供了启发式和精确两类命中判断技术。

这里要拆清一个容易混淆的术语：**前缀缓存感知路由**解决的是「请求发到哪个副本」，**P/D 分离**解决的是「prefill 和 decode 在哪里执行」，两个是独立机制，可以单独启用也可以组合使用。下文的 P/D 分离特指 prefill/decode 解耦，与前缀缓存无关。

### 2.4 P/D 分离

`prefill`（计算密集）和 `decode`（访存密集）被拆到各自的专用 worker 上，`prefill` 完成后通过高速 `RDMA` 通道把 `KV` 状态传给 `decode` 侧（官方架构描述为「high-speed RDMA KV transfer」）。底层依赖 `vLLM` 的 `NixlConnector` 做 `RDMA` 传输。官方引用的 `AWS` 基准显示，在 `B200` 上对比标准 `vLLM` 部署，分离式部署的吞吐最高提升 70%（厂商口径，见第 6 节）。

## 3. KV Cache 分层与 offloader

### 3.1 分层结构

`HBM` 里驱逐的 `KV` 块不丢弃，而是下沉到 `CPU` 内存，可选再下沉到共享文件系统，后续请求命中前缀时再加载回来。官方 `tiered-prefix-cache` 指南描述的层级是：

```text
GPU HBM → CPU RAM → (可选) ReadWriteMany 共享文件系统
```

共享文件系统层是 2026 年 2 月新增的能力（官方博客「Native KV Cache Offloading to Any Filesystem」），多个副本可以复用同一份 `KV` 存储，官方给出的对比数据是 TTFT 最多快 16.8 倍（厂商口径）。

路由器侧配合做第二层缓存感知：scorer 除了跟踪各服务器 `GPU` 里的前缀，还会跟踪 `CPU` 层各持有哪些前缀（`gpu-prefix-cache-producer` / `cpu-prefix-cache-producer` 两个索引），因为 `vLLM` 不上报 `CPU` 层的大小，这一层的容量需要手工设定 `lruCapacityPerServer`。

### 3.2 offloader 的三种实现

分层缓存通过 `CONNECTOR` 参数选择实现，官方指南列出三类：

| CONNECTOR           | 实现                                                                       | 支持层级    | 适用引擎                            |
| ------------------- | -------------------------------------------------------------------------- | ----------- | ----------------------------------- |
| `native`（推荐）    | `vLLM OffloadingConnector`；`SGLang HiCache`；`vLLM TPU offload connector` | `cpu`、`fs` | vLLM / SGLang / TPU                 |
| `lmcache-connector` | [LMCache](https://lmcache.ai)                                              | `cpu`、`fs` | vLLM（NVIDIA / AMD GPU、Intel XPU） |
| `mooncake-store`    | `MooncakeStore`（`CPU DRAM` 池 + `SSD`，走 `RDMA`）                        | `cpu`、`fs` | —                                   |

### 3.3 LMCache 一节

`LMCache` 是三种 offloader 中社区认知度最高的一个，也是 [本仓库 KV Cache 系列文章](../../09_inference_system/kv_cache/02_systems/01_lmcache/01_lmcache_overview.md) 单独深析过的组件，这里只讲它与 `llm-d` 的接口关系：

- `llm-d` 通过 `CONNECTOR=lmcache-connector` 启用 `LMCache` 作为 offloader，覆盖 `cpu` 与 `fs` 两层；
- `CPU` 层容量按 `LMCACHE_MAX_LOCAL_CPU_SIZE` 配置（单位 GiB，按张量并行 rank 计）；
- 指标走 `LMCache` 自己的导出项，如 `lmcache:retrieve_hit_rate`（检索命中率）和 `lmcache:local_cache_usage`（本地缓存占用）。

`LMCache` 本身的多层级存储架构、`CacheGen` 压缩、`NIXL` 传输后端等细节，见 [LMCache 架构概览](../../09_inference_system/kv_cache/02_systems/01_lmcache/01_lmcache_overview.md)。

## 4. 模型服务编排：基于 LWS

多机推理服务（张量并行或专家并行跨节点展开）以 Leader-Worker 组为单位运行：Leader 起主进程，Worker 随后加入，故障时整组重建。`llm-d` 的模型服务编排直接采用 [`LeaderWorkerSet`](./04_lws_intro.md)（`LWS`）实现这种组语义，`LWS` 也因此与 `llm-d` 项目协同设计——这是 `K8s` 社区里「控制器原语为推理负载而改」的一个实际案例。

## 5. Well-Lit Paths：官方部署路径

`llm-d` 把经过测试、带基准数据的部署模式称为 `Well-Lit Paths`（照明路径），每条路径带 `Helm chart`、调优参数、基准结果和可观测性配置。截至核对日，官方站点列出的路径包括：

- **Precise Prefix-Cache Aware Routing**：精确前缀缓存感知路由
- **Tiered Prefix Cache**：分层前缀缓存（`CPU` / 文件系统卸载）
- **P2P Prefix Cache Sharing**：副本间前缀缓存共享
- **Prefill/Decode Disaggregation**：预填充/解码分离
- **Route Requests by Predicted Latency**：按预测延迟路由
- **Flow Control and Fairness**：流量控制与公平性
- **Autoscale your Inference Pool**：推理池自动扩缩容

生产落地建议从对应路径的 `Helm chart` 入手改参数，而不是从零拼组件——这些路径的基准数字也是这么测出来的。

## 6. 性能数据与口径

以下是官方渠道公布的性能声明，**均为厂商口径**，未在本仓库独立复测：

| 声明                                             | 出处                    |
| ------------------------------------------------ | ----------------------- |
| 输出吞吐最高 3 倍                                | llm-d.ai 官网           |
| TTFT 最快 2 倍                                   | llm-d.ai 官网           |
| P/D 分离对比标准 vLLM，`B200` 上吞吐最高提升 70% | 官方 README（AWS 基准） |
| TTFT 与 ITL 降低 40%（`Google Vertex` 场景）     | llm-d.ai 官网           |
| 文件系统 `KV` 卸载使 TTFT 最多快 16.8 倍         | 官方博客（2026-02）     |

这类数字的生成条件（模型、负载、硬件、对比基线）各不相同，直接迁移到你自己的环境大概率打折。项目迭代很快，接口和路径都在动，部署前以官方最新文档为准，本数据核对日 2026-11-05。

## 7. 总评

`llm-d` 把「推理网关 + 缓存调度 + 分层 KV + 多机编排」这条链路做成了 `Kubernetes` 上的标准开源件，架构落在社区标准（`Gateway API Inference Extension`、`LWS`、`CDI`）之上，而不是私有协议，这是它区别于各家自研推理网关的根本点。v1.0 发布后，核心 `API` 已稳定，适合认真评估生产引入。

**折扣清单**（采用前需要接受的条件）：

1. **性能数字全部是厂商口径**，不同声明的测试条件不透明，落地前必须用自己负载复测；
2. **组件链路长**：`Gateway` + `EPP` + offloader + `LWS` + 模型服务器，每一层都有自己的配置空间和故障面，运维成本明显高于单机 `vLLM`；
3. **`RDMA` 依赖**：`P/D` 分离和 `MooncakeStore` 路径需要集群具备 `RDMA` 网络，纯以太网集群收益打折；
4. **`CPU` 层缓存容量需手工设定**（`lruCapacityPerServer`），引擎不上报 `CPU` 层大小，容量配错会直接影响命中率；
5. **版本迭代快**：`Well-Lit Paths` 的参数形态随版本变动，锁定版本再部署。

## 参考资料

- [llm-d 官网](https://llm-d.ai)
- [llm-d GitHub 仓库](https://github.com/llm-d/llm-d)
- [llm-d 架构文档](https://llm-d.ai/docs/architecture)
- [Tiered Prefix Cache 官方指南](https://github.com/llm-d/llm-d/blob/main/guides/tiered-prefix-cache/README.md)
- [LeaderWorkerSet 官方文档](https://lws.sigs.k8s.io)
- [LMCache 官网](https://lmcache.ai)
- 本仓库相关文章：[vLLM + LWS 多机多卡推理](./04_lws_intro.md) · [LMCache 架构概览](../../09_inference_system/kv_cache/02_systems/01_lmcache/01_lmcache_overview.md)
