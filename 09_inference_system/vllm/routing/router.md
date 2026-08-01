# vLLM Router：高性能 LLM 推理网关

## 1. 项目概述

vLLM Router 是一个专为 vLLM 大规模部署场景设计的**高性能、轻量级**请求转发系统，采用**混合 Rust/Python 架构**实现——核心路由器完全使用 Rust 语言编写以获得最佳性能，同时通过 PyO3 绑定将其封装为易于安装和使用的 Python 包（`vllm-router`），使得用户既可以通过 Rust 二进制直接运行，也可以在 Python 环境中无缝集成。

### 1.1 核心特性

vLLM Router 提供了丰富的功能模块，涵盖从基础的请求路由到企业级的可靠性保障，以下表格对各功能模块进行了概述：

| 特性分类                | 说明                                                                        |
| ----------------------- | --------------------------------------------------------------------------- |
| **高性能架构**          | 基于 Rust + Axum 的异步请求处理框架                                         |
| **多种负载均衡**        | 支持 Random、Round Robin、Consistent Hash、Power of Two、Cache Aware 等策略 |
| **Prefill-Decode 分离** | 支持预填充/解码分离路由，兼容 NIXL 和 NCCL 连接器                           |
| **Kubernetes 原生**     | 内置服务发现、健康监控和自动工作节点管理                                    |
| **企业级功能**          | 熔断器、重试机制、速率限制、Prometheus 指标导出                             |
| **双协议支持**          | 同时支持 HTTP 和 gRPC 通信模式                                              |

### 1.2 技术栈

项目在 Rust 侧精心选择了一系列成熟稳定的核心依赖，以确保系统的高性能和可靠性：Web 服务层采用 **axum** 作为高性能异步 Web 框架，底层异步运行时则基于 **tokio** 构建；对于需要 gRPC 通信的场景，项目集成了 **tonic** 框架；Kubernetes 原生支持通过 **kube** 客户端实现服务发现和集群管理；为了实现 Rust 与 Python 的无缝互操作，项目使用 **pyo3** 创建 Python 绑定；此外，**tokenizers** 库提供了对 Hugging Face 分词器的支持，使路由器能够进行 token 级别的负载感知。

---

## 2. 系统架构

vLLM Router 的分层设计围绕一个目标：**请求处理路径上的每一层只做一件事**——中间件层处理横切关注点，路由层按协议分发，负载均衡层选节点，核心层维护节点状态。数据流自顶向下单向流动。

### 2.1 整体架构图

下图展示了从客户端发起请求到最终被工作节点处理的完整数据流，清晰地呈现了路由器内部的分层设计以及各层之间的职责划分：

```text
┌─────────────────────────────────────────────────────────────────────┐
│                          Client Requests                            │
│                    (HTTP/gRPC API Calls)                            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         vLLM Router                                 │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     Middleware Layer                          │  │
│  │  • Request ID Tracking    • Rate Limiting    • API Auth       │  │
│  │  • CORS Handling          • Concurrency Control               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                  │                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     Router Layer                              │  │
│  │  • HTTP Router         • OpenAI Router      • PD Router       │  │
│  │  • gRPC Router         • vLLM PD Router     • RouterManager   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                  │                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  Load Balancing Policies                      │  │
│  │  • Random              • Round Robin        • Consistent Hash │  │
│  │  • Power of Two        • Cache Aware                          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                  │                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                      Core Layer                               │  │
│  │  • Worker Registry     • Circuit Breaker    • Retry Executor  │  │
│  │  • Health Checker      • Token Bucket                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      vLLM Worker Nodes                              │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│    │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  │ Worker N │           │
│    └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 代码结构

Rust 核心位于 `src/`，按职责分为六组；Python 包装位于 `py_src/vllm_router/`：

| 目录         | 职责             | 关键文件                                                                                                 |
| ------------ | ---------------- | -------------------------------------------------------------------------------------------------------- |
| `core/`      | 节点状态与可靠性 | `worker.rs`（Worker trait）、`worker_registry.rs`、`circuit_breaker.rs`、`retry.rs`、`token_bucket.rs`   |
| `policies/`  | 五种负载均衡策略 | `random.rs`、`round_robin.rs`、`consistent_hash.rs`、`power_of_two.rs`、`cache_aware.rs` + `registry.rs` |
| `routers/`   | 协议与拓扑路由   | `http/`（标准/OpenAI/PD/vLLM PD 四种路由器）、`grpc/`、`router_manager.rs`                               |
| `tokenizer/` | 前缀缓存感知     | `huggingface.rs`、`tiktoken.rs` + `tree.rs`（Radix 树）                                                  |
| 顶层         | 基础设施         | `server.rs`（Axum 服务器）、`middleware.rs`、`metrics.rs`、`service_discovery.rs`、`protocols/`          |
| `py_src/`    | Python 包装      | `launch_router.py`（CLI）、`router.py`（PyO3 桥接）、`mini_lb.py`（纯 Python 备用 LB）                   |

---

## 3. 核心功能详解

vLLM Router 的核心功能分四组：路由模式（部署拓扑的选择）、负载均衡策略（请求分配的依据）、可靠性特性（故障下的行为）与 Kubernetes 服务发现（节点来源的自动化）。

### 3.1 路由模式

vLLM Router 针对不同的部署场景和性能需求设计了三种路由模式，用户可以根据自身的基础设施架构和业务特点选择最适合的模式。

#### 3.1.1 标准路由模式

标准路由模式是最基础也是最常用的路由模式，在该模式下，所有工作节点被组织成一个扁平的节点池，路由器根据用户选定的负载均衡策略将请求均匀地分发到各个健康的工作节点上。

```bash
./target/release/vllm-router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy round_robin
```

#### 3.1.2 Prefill-Decode 分离模式

Prefill-Decode（PD）分离模式是针对大语言模型推理特性优化的高级路由模式，它将计算密集型的预填充（Prefill）阶段和内存密集型的解码（Decode）阶段分离到独立的工作节点池中运行，允许用户为两个阶段配置不同的负载均衡策略，从而实现更精细的资源调度和更高的整体吞吐量。

**NIXL 连接器模式：**

```bash
./target/release/vllm-router \
    --vllm-pd-disaggregation \
    --prefill http://127.0.0.1:8081 http://127.0.0.1:8082 \
    --decode http://127.0.0.1:8083 http://127.0.0.1:8084 \
    --policy consistent_hash
```

**NCCL 连接器模式（ZMQ 发现）：**

```bash
./target/release/vllm-router \
    --vllm-pd-disaggregation \
    --vllm-discovery-address 0.0.0.0:30001 \
    --prefill-policy consistent_hash \
    --decode-policy round_robin
```

#### 3.1.3 数据并行模式

数据并行（Data Parallel，DP）模式专为单节点多 GPU 部署场景设计，当用户通过 `--intra-node-data-parallel-size` 参数指定大于 1 的数据并行度时，路由器会自动为每个工作节点 URL 创建相应数量的 DP-aware 子工作节点，并通过特定的路由键确保相关请求被路由到同一个 DP rank 上，以实现 KV 缓存的高效复用。

```bash
./target/release/vllm-router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy consistent_hash \
    --intra-node-data-parallel-size 8
```

### 3.2 负载均衡策略

vLLM Router 内置了五种经过精心设计的负载均衡策略，每种策略都针对特定的使用场景进行了优化，用户可以通过 `--policy` 命令行参数灵活选择最适合自身业务特点的策略：

| 策略              | 描述                   | 会话亲和性 | 负载感知 | 适用场景              |
| ----------------- | ---------------------- | ---------- | -------- | --------------------- |
| `round_robin`     | 按顺序循环分发请求     | 否         | 否       | 通用场景，均匀分发    |
| `random`          | 随机选择健康工作节点   | 否         | 否       | 简单部署              |
| `consistent_hash` | 相同会话路由到相同节点 | 是         | 否       | 多轮对话，KV 缓存复用 |
| `power_of_two`    | 两次随机选择最优       | 否         | 是       | 负载敏感型工作负载    |
| `cache_aware`     | 基于前缀缓存优化路由   | 是（缓存） | 是       | 重复提示词，few-shot  |

前两种是**无状态策略**：`round_robin` 维护一个游标按顺序轮流分发，`random` 从健康节点中均匀随机选取。两者都不需要负载信息、不维护会话状态，实现最简单、开销最低——代价是无法利用 KV cache 亲和性。它们是后三种有状态策略的基线：一致性哈希把"亲和"加进来，二次选择把"负载感知"加进来，缓存感知把"前缀复用"加进来。选型时先问"我需要亲和还是负载感知"——都不需要就用 `round_robin`。

#### 3.2.1 Consistent Hash 策略

Consistent Hash（一致性哈希）策略是实现会话亲和性的关键机制，它通过从请求中提取唯一的会话标识作为哈希键，确保来自同一会话或同一用户的请求始终被路由到同一个工作节点，从而最大化 KV 缓存的复用率并保持对话上下文的连贯性。

**算法原理**：

传统哈希算法在节点数量变化时会导致大量键的重新映射，而一致性哈希通过将哈希空间组织成一个虚拟环（通常范围为 0 到 2^32-1），并将节点和键都映射到这个环上来解决此问题：

1. **节点映射**：每个工作节点通过哈希函数计算其在环上的位置
2. **虚拟节点**：为每个物理节点创建 160 个虚拟节点（virtual nodes），均匀分布在哈希环上，解决数据倾斜问题
3. **键定位**：计算请求键的哈希值，顺时针找到第一个大于等于该值的虚拟节点，即为目标节点
4. **节点增减**：当节点加入或离开时，仅影响其相邻的一小部分键，大部分映射关系保持不变

这种设计使得在扩缩容时，只有约 1/N 的请求需要重新路由（N 为节点数），而非传统哈希的几乎全部请求。

**哈希键提取优先级**：

路由器在提取哈希键时会按照以下优先级顺序依次尝试，直到找到有效的标识符为止：

| 优先级 | 来源         | 头部/字段                                        |
| ------ | ------------ | ------------------------------------------------ |
| 1      | HTTP Header  | `X-Session-ID`                                   |
| 2      | HTTP Header  | `X-User-ID`                                      |
| 3      | HTTP Header  | `X-Tenant-ID`                                    |
| 4-6    | HTTP Header  | `X-Request-ID`, `X-Correlation-ID`, `X-Trace-ID` |
| 7      | Request Body | `session_params.session_id`                      |
| 8      | Request Body | `user` (OpenAI 格式)                             |
| 9-10   | Request Body | `session_id`, `user_id` (旧格式)                 |
| 11     | Fallback     | 请求体哈希                                       |

**推荐使用 HTTP Header 方式传递会话标识**，因为这种方式无需解析请求体 JSON，路由延迟更低，且与各种标准基础设施组件（如 Nginx、Envoy、Kubernetes Ingress）兼容性更好：

```bash
curl -X POST http://router:8000/v1/chat/completions \
  -H "X-Session-ID: conversation-12345" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3", "messages": [{"role": "user", "content": "Hello!"}]}'
```

#### 3.2.2 Cache Aware 策略

Cache Aware（缓存感知）策略是一种智能的负载均衡算法，它通过在路由器内部为每个工作节点维护一棵近似的 radix 树来追踪已缓存的请求前缀，当新请求到达时，策略会计算该请求与各节点缓存前缀的匹配程度，并优先将请求路由到能够复用最多缓存内容的节点上，从而显著减少重复计算并提升整体推理吞吐量。

**Radix 树（基数树）原理**：

Radix 树是一种压缩的前缀树（Trie），专门用于高效存储和检索字符串前缀。与标准 Trie 每个节点存储单个字符不同，Radix 树将连续的、没有分支的字符序列压缩存储在一个节点中，大幅减少内存占用和遍历深度。

**树结构特点**：

- **节点压缩**：路径上的连续字符被合并存储，例如 "hello" 和 "help" 共享 "hel" 节点后分叉
- **字符级索引**：使用 `DashMap<char, Node>` 实现 O(1) 的子节点查找
- **多租户支持**：每个节点维护 `tenant_last_access_time` 映射，记录不同工作节点（租户）的最后访问时间
- **线程安全**：采用节点级读写锁，支持高并发访问

**前缀匹配过程**：

```text
插入 "hello world" (Worker A):
  root -> "hello" -> " world" [Worker A]

插入 "hello vllm" (Worker B):
  root -> "hello" -> " world" [Worker A]
                  -> " vllm"  [Worker B]

查询 "hello world from user":
  1. 从根节点开始，匹配 "hello"（5 字符）
  2. 进入子节点，匹配 " world"（6 字符）
  3. 剩余 " from user" 无法匹配，返回
  4. 匹配率 = 11 / 21 ≈ 52.4%，建议路由到 Worker A
```

**双模式动态切换机制**：

Cache Aware 策略的核心创新在于动态平衡缓存命中与负载均衡：

1. **缓存优先模式**（系统负载均衡时）：
   - 计算请求与各节点前缀树的匹配率
   - 若最高匹配率 > `cache_threshold`，路由到该节点
   - 若匹配率较低，路由到树大小最小的节点（缓存容量最大）

2. **负载均衡模式**（系统负载不均衡时）：
   - 当同时满足：`(max_load - min_load) > balance_abs_threshold` **且** `max_load > min_load * balance_rel_threshold`
   - 切换到最短队列策略，将请求路由到负载最低的节点
   - 仍更新前缀树以维持缓存状态追踪

这种设计避免了在高负载下仍坚持缓存亲和性导致的负载倾斜问题。

**LRU 淘汰策略**：

为防止内存无限增长，系统启动后台线程定期执行 LRU（Least Recently Used）淘汰：

- **触发条件**：树节点数超过 `max_tree_size` 或达到 `eviction_interval_secs` 间隔
- **淘汰对象**：最久未访问的叶子节点（通过 `tenant_last_access_time` 追踪）
- **批量处理**：每次淘汰约 10% 的过期节点，减少性能抖动

Cache Aware 策略提供了丰富的可调参数，允许用户根据实际工作负载特征进行精细调优：

| 参数                     | 默认值 | 说明                                                                     |
| ------------------------ | ------ | ------------------------------------------------------------------------ |
| `cache_threshold`        | 0.5    | 当请求与某节点的前缀匹配率超过此阈值时，才会基于缓存进行路由决策         |
| `balance_abs_threshold`  | 32     | 当节点间负载绝对差异超过此值时，策略将切换到负载均衡模式而非缓存优先模式 |
| `balance_rel_threshold`  | 1.1    | 当节点间负载比率超过此值时，同样触发负载均衡模式                         |
| `eviction_interval_secs` | 30     | radix 树缓存条目的定期清理周期，用于移除过期的前缀记录                   |
| `max_tree_size`          | 10000  | 每个工作节点对应的 radix 树允许的最大节点数量                            |

```bash
./target/release/vllm-router \
  --policy cache_aware \
  --cache-threshold 0.5 \
  --balance-abs-threshold 32 \
  --balance-rel-threshold 1.1 \
  --worker-urls http://worker1:8000 http://worker2:8000
```

#### 3.2.3 Power of Two 策略

Power of Two（二次选择）策略源于著名的 "Power of Two Choices" 负载均衡理论，该理论表明：**仅从两个随机选择的候选者中挑选较优者，即可达到与全局最优选择相近的负载均衡效果**。

**数学原理**：

假设有 N 个节点，每个请求的负载为 1：

- **完全随机选择**：最大负载约为 O(log N / log log N)
- **二次选择**：最大负载仅为 O(log log N)

这意味着二次选择可以将最大负载从对数级降低到双重对数级，效果接近需要维护全局状态的完美调度。

**算法流程**：

1. 从所有健康节点中**随机选取 2 个候选节点**
2. 比较两个节点的当前负载（待处理请求数）
3. 选择负载较低的节点作为目标
4. 若负载相同，则随机选择其一

**为什么只选 2 个？**

研究表明，从 2 个候选者中选择已经捕获了大部分收益：

- 从 1 个（纯随机）到 2 个：负载均衡效果**指数级提升**
- 从 2 个到 3 个或更多：收益**边际递减**，但协调成本增加

这种设计在无需集中式协调的情况下，实现了接近全局最优的负载分布，特别适合请求处理时间差异较大的场景。

**策略选择建议**：

| 场景特征           | 推荐策略          | 原因                         |
| ------------------ | ----------------- | ---------------------------- |
| 请求处理时间差异大 | `power_of_two`    | 动态负载感知，避免慢节点堆积 |
| 需要会话状态保持   | `consistent_hash` | 相同用户路由到相同节点       |
| 存在大量重复提示词 | `cache_aware`     | 最大化前缀缓存命中率         |
| 简单均匀分发       | `round_robin`     | 实现简单，无状态             |

### 3.3 可靠性特性

在生产环境中，系统的稳定性和容错能力至关重要，vLLM Router 为此提供了多层可靠性保障机制，包括具有指数退避的自动重试、基于状态机的熔断器以及周期性的健康检查，这些机制协同工作，确保即使在部分工作节点出现故障时，整个系统仍能保持可用。

#### 3.3.1 重试机制

重试机制默认启用，当请求因临时性故障（如网络抖动或后端过载）而失败时，路由器会自动进行重试，重试间隔采用指数退避算法计算，并引入随机抖动以避免多个请求同时重试造成的"惊群效应"：

```bash
./target/release/vllm-router \
  --retry-max-retries 5 \
  --retry-initial-backoff-ms 50 \
  --retry-max-backoff-ms 30000 \
  --retry-backoff-multiplier 1.5 \
  --retry-jitter-factor 0.2
```

路由器会在收到以下 HTTP 状态码时触发重试逻辑：408（Request Timeout）、429（Too Many Requests）、500（Internal Server Error）、502（Bad Gateway）、503（Service Unavailable）和 504（Gateway Timeout），这些状态码通常表示临时性故障，通过重试往往能够成功恢复。

#### 3.3.2 熔断器

熔断器是防止故障级联扩散的重要机制，vLLM Router 为每个工作节点维护独立的熔断器实例，当某个节点连续出现多次失败时，熔断器会自动"跳闸"，暂时将该节点从可用节点池中移除，避免持续向故障节点发送请求。熔断器基于以下状态机模型运行：

```text
Closed ──(N次连续失败)──► Open ──(超时)──► HalfOpen ──(M次连续成功)──► Closed
                            ▲                │
                            └──(失败)─────────┘
```

```bash
./target/release/vllm-router \
  --cb-failure-threshold 10 \
  --cb-success-threshold 3 \
  --cb-timeout-duration-secs 60 \
  --cb-window-duration-secs 120
```

#### 3.3.3 健康检查

除了被动的熔断机制外，路由器还实现了主动的健康检查功能，会按照配置的时间间隔定期向所有工作节点发送健康探测请求，并根据探测结果自动更新节点的健康状态，将连续多次探测失败的节点标记为不健康并从路由目标中排除，同时在节点恢复后自动将其重新加入可用节点池：

```bash
./target/release/vllm-router \
  --health-failure-threshold 3 \
  --health-success-threshold 2 \
  --health-check-timeout-secs 5 \
  --health-check-interval-secs 60 \
  --health-check-endpoint /health
```

### 3.4 Kubernetes 服务发现

vLLM Router 提供了与 Kubernetes 集群的深度集成能力，能够通过 Kubernetes API 自动发现符合指定标签选择器的工作节点 Pod，并实时监听 Pod 的创建、更新和删除事件，确保路由器的工作节点列表始终与集群状态保持同步，无需手动维护静态的工作节点 URL 列表：

```bash
./target/release/vllm-router \
    --service-discovery \
    --selector app=vllm-worker role=inference \
    --service-discovery-namespace default \
    --service-discovery-port 8000
```

在 PD 分离模式下，服务发现支持为预填充节点和解码节点配置不同的标签选择器，路由器会分别监听两类节点并将它们加入对应的节点池中，同时支持通过 Pod 注解指定 bootstrap 端口等高级配置：

```bash
./target/release/vllm-router \
    --service-discovery \
    --prefill-selector app=vllm-prefill \
    --decode-selector app=vllm-decode \
    --bootstrap-port-annotation vllm.ai/bootstrap-port
```

---

## 4. API 端点

vLLM Router 对外暴露了完整的 RESTful API 接口，主要分为四大类：与 OpenAI API 完全兼容的推理端点、用于 Kubernetes 探针和监控集成的健康检查端点、支持运行时动态管理的管理端点，以及 Prometheus 格式的指标端点。

### 4.1 OpenAI 兼容端点

路由器完整实现了 OpenAI API 规范中的核心端点，使得现有基于 OpenAI SDK 或兼容 API 的应用程序无需修改代码即可无缝迁移到 vLLM 后端：

| 端点                        | 方法 | 描述     |
| --------------------------- | ---- | -------- |
| `/v1/chat/completions`      | POST | 聊天补全 |
| `/v1/completions`           | POST | 文本补全 |
| `/v1/embeddings`            | POST | 文本嵌入 |
| `/v1/models`                | GET  | 模型列表 |
| `/v1/responses`             | POST | 响应 API |
| `/v1/responses/{id}`        | GET  | 获取响应 |
| `/v1/responses/{id}/cancel` | POST | 取消响应 |
| `/v1/rerank`                | POST | 重排序   |

### 4.2 健康检查端点

健康检查端点专为 Kubernetes 存活探针（Liveness Probe）和就绪探针（Readiness Probe）设计，同时也可用于外部监控系统或负载均衡器进行健康状态检测：

| 端点               | 方法 | 描述         |
| ------------------ | ---- | ------------ |
| `/health`          | GET  | 健康状态     |
| `/liveness`        | GET  | 存活探针     |
| `/readiness`       | GET  | 就绪探针     |
| `/health_generate` | GET  | 生成健康检查 |

### 4.3 管理端点

管理端点提供了在运行时动态调整路由器配置的能力，包括添加或移除工作节点、查询各节点的负载状态、以及清空缓存等操作，这些端点对于自动化运维和故障处理非常有用：

| 端点             | 方法   | 描述                 |
| ---------------- | ------ | -------------------- |
| `/workers`       | GET    | 列出所有工作节点     |
| `/workers`       | POST   | 添加工作节点         |
| `/workers/{url}` | GET    | 获取工作节点详情     |
| `/workers/{url}` | DELETE | 移除工作节点         |
| `/add_worker`    | POST   | 添加工作节点（旧版） |
| `/remove_worker` | POST   | 移除工作节点（旧版） |
| `/list_workers`  | GET    | 列出工作节点（旧版） |
| `/get_loads`     | GET    | 获取负载信息         |
| `/flush_cache`   | POST   | 清空缓存             |

### 4.4 指标端点

路由器内置了 Prometheus 指标导出功能，默认在 `127.0.0.1:29000` 地址上暴露标准格式的指标数据，可直接被 Prometheus 服务器抓取，用于监控路由器的请求量、延迟分布、错误率等关键性能指标。

```bash
./target/release/vllm-router \
    --prometheus-host 0.0.0.0 \
    --prometheus-port 9000
```

---

## 5. 配置参考

配置入口有三个：命令行参数、环境变量与 JSON 配置文件——前两者适合运维脚本，JSON 配置适合声明式管理（Kubernetes ConfigMap、版本控制）。

### 5.1 主要配置项

以下表格汇总了路由器最常用的配置参数，这些参数可以通过命令行标志或环境变量进行设置：

| 配置项                      | 默认值      | 描述           |
| --------------------------- | ----------- | -------------- |
| `--host`                    | 127.0.0.1   | 服务器监听地址 |
| `--port`                    | 30000       | 服务器监听端口 |
| `--policy`                  | cache_aware | 负载均衡策略   |
| `--max-payload-size`        | 512MB       | 最大请求体大小 |
| `--request-timeout-secs`    | 1800        | 请求超时（秒） |
| `--max-concurrent-requests` | 32768       | 最大并发请求数 |
| `--queue-size`              | 100         | 请求队列大小   |
| `--queue-timeout-secs`      | 60          | 队列超时（秒） |

### 5.2 认证配置

vLLM Router 支持对传入请求进行 Bearer Token 认证，用户可以配置一个或多个外部验证服务的 URL，路由器会将请求中的 Token 转发到这些服务进行验证，只有验证通过的请求才会被转发到后端工作节点：

```bash
# 环境变量
API_KEY_VALIDATION_URLS=https://auth.example.com/validate

# CLI
./target/release/vllm-router \
    --api-key-validation-urls https://auth.example.com/validate
```

### 5.3 JSON 配置文件示例

除了命令行参数外，路由器还支持通过 JSON 配置文件以声明式的方式定义路由规则，这种方式更适合配置管理工具（如 Kubernetes ConfigMap）集成，以及需要版本控制的场景：

```json
{
  "route": {
    "type": "RoundRobinRoute",
    "servers": [
      { "host": "localhost", "port": 8000 },
      { "host": "localhost", "port": 8001 }
    ]
  }
}
```

---

## 6. 性能特点

vLLM Router 的性能优势来自架构选择与官方 benchmark 两组证据。架构层面，全异步 Tokio 运行时、持久连接池、优化序列化与 Thin LTO 编译优化共同保证了低开销转发。数据层面，vLLM 官方发布博客（2025-12-13）给出了与同类方案的对比：

| 场景                                          | 对比对象    | 吞吐            | TTFT           |
| --------------------------------------------- | ----------- | --------------- | -------------- |
| Llama 3.1 8B，8 Prefill + 8 Decode pods       | llm-d       | **+25%**        | **快 ~1200ms** |
| Llama 3.1 8B，8 Prefill + 8 Decode pods       | K8s 原生 LB | **+100%（2×）** | 接近           |
| DeepSeek V3，1 Prefill (TP8) + 1 Decode (TP8) | llm-d       | 接近            | **快 ~2000ms** |
| DeepSeek V3，1 Prefill (TP8) + 1 Decode (TP8) | K8s 原生 LB | **+100%**       | **快 ~2000ms** |

两个观察：**吞吐优势在 PD 分离场景下对 K8s 原生 LB 稳定在 2×**——因为 K8s 原生 LB 不感知 Prefill/Decode 状态，而 vLLM Router 的 PD 路由 + 一致性哈希让 KV cache 亲和性最大化；**TTFT 优势随模型变大而扩大**——DeepSeek V3 场景比 Llama 3.1 场景快得更多（~2000ms vs ~1200ms），说明 PD 感知路由在长序列、重 prefill 负载下价值更高。benchmark 排除了 vLLM 内置 DP/EP coordinator——其已知性能问题使吞吐仅为其他方案的 1/8。

---

## 7. 部署要点

vLLM Router 提供两条部署路径：Docker 容器适合快速验证与小规模部署（`docker build -f Dockerfile.router` 构建，默认端口 8080，指标端口 29000）；Kubernetes 编排适合生产环境，项目在 `scripts/k8s/` 下提供了 `llama3/`、`llama3.1/`、`deepseek-v31/`、`pd_disagg/` 四套经过验证的部署配置——其中 `pd_disagg/` 展示了 Prefill 与 Decode 节点分离部署的完整形态。完整的构建、测试与 pre-commit 流程见项目仓库 README。

---

## 延伸阅读

- [vLLM Semantic Router 深度剖析](semantic_router_deep_dive.md)：vLLM Router 与 Semantic Router 是 vllm-project 组织下**两个独立的项目**（独立仓库、独立技术栈）——vLLM Router 是 Rust 请求网关，解决"把请求分发到哪个**节点实例**"；Semantic Router 是 Go + Envoy ExtProc 的模型选择器，解决"把请求路由给哪个**模型**"。两者不是层级关系，而是两个不同的路由问题维度。生产中可以组合：Semantic Router 先基于信号选定模型，vLLM Router 再把请求分发到该模型的多个实例。

---
