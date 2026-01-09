# 容易被忽略的 containerd 运行时日志

在 Kubernetes/容器平台里，大家最常看的日志是容器进程的标准输出/错误，路径通常在 `/var/log/containers/<pod_name>_<namespace>_<container_name>-<container_id>.log`。

当容器创建失败、Pod 启动不起来时，有时候关键信息往往不在这里，而是在 `runc` 写出的运行时日志 `log.json`（通常位于 `/run/containerd/io.containerd.runtime.v2.task/k8s.io/<container_id>/log.json`）。这个文件记录了运行时的具体操作与错误。

但是若该文件无限增长占满 `/run` 的 tmpfs（或出现权限 / 路径异常），可能影响新容器创建与节点稳定性。

## 1. containerd 运行时日志增长问题及其影响

### 1.1 现象与影响

在生产环境中，`log.json` 可能无限增长。它由 `runc` 持续写入，但没有自动轮转机制，最终会耗尽磁盘（若挂载在 tmpfs，则是内存）空间，影响节点与集群稳定性。

**关键点**： `runc` 的 `--log` 只指定路径，不负责大小控制与轮转；日志会持续追加直至空间耗尽。

### 1.2 实际案例分析

#### 1.2.1 案例一：NVIDIA Container Runtime 配置重复记录

**问题描述**：

- 日志文件大小：`32085414 bytes (约 30.6 MB)`
- 重复记录 NVIDIA Container Runtime 的完整配置信息
- 每次记录包含约 1KB 的 JSON 配置数据
- 时间间隔较短（几秒到几十秒）

> 说明：以上数值来源于博主遇到的一个现场问题。

**日志模式**：

```json
{"level":"info","msg":"Running with config:\n{\n  \"AcceptEnvvarUnprivileged\": true,\n  \"NVIDIAContainerCLIConfig\": {\n    \"Root\": \"\"\n  },\n  \"NVIDIACTKConfig\": {\n    \"Path\": \"nvidia-ctk\"\n  },\n  \"NVIDIAContainerRuntimeConfig\": {\n    \"DebugFilePath\": \"/dev/null\",\n    \"LogLevel\": \"info\",\n    \"Runtimes\": [\n      \"docker-runc\",\n      \"runc\"\n    ],\n    \"Mode\": \"auto\"\n  }\n}","time":"2024-01-20T16:05:43+08:00"}
{"level":"info","msg":"Using low-level runtime /usr/bin/runc","time":"2024-01-20T16:05:43+08:00"}
```

#### 1.2.2 案例二：GitHub Issue #8972 - 生产环境节点故障

containerd 官方 GitHub 仓库中报告了一个严重的生产环境问题（[Issue #8972](https://github.com/containerd/containerd/issues/8972)）。

> "log.json of a container may grow to burst the tmpfs of /run, if a k8s user configure an exec liveness probe of a non exist executable file name."

**问题影响**：

- 当 Kubernetes 用户配置了不存在的可执行文件的 exec liveness probe 时
- log.json 文件会快速增长，最终撑爆 `/run` 目录的 tmpfs
- 导致节点无法创建新的容器，影响整个集群的可用性

#### 1.2.3 案例三：NVIDIA/nvidia-container-toolkit#511 - 大规模部署失败

NVIDIA Container Toolkit 项目中报告的问题（[Issue #511](https://github.com/NVIDIA/nvidia-container-toolkit/issues/511)）。

> "Excessive runtime logging could cause Kubernetes workload deployment failure"

**影响范围**：

- `/run/containerd/io.containerd.runtime.v2.task/k8s.io/<container-id>/log.json` 文件过大
- `/run` tmpfs 挂载点达到 100% 利用率
- 受影响节点上无法进一步创建容器

### 1.3 问题根本原因

#### 1.3.1 runc 日志轮转现状

通过对 runc 源码和官方文档的分析，发现：

1. **无内置轮转**：runc 的 `--log` 参数只是指定日志文件路径，不提供大小限制或轮转功能
2. **持续追加**：runc 会持续向指定的 log.json 文件追加日志，直到磁盘空间耗尽
3. **无配置选项**：runc 没有提供 `--log-max-size` 或类似的参数来控制日志文件大小

#### 1.3.2 与容器日志轮转的区别

containerd 确实支持容器日志（stdout/stderr）的轮转，但这与 runc 的 log.json 是两个不同的系统：

```go
// 容器日志轮转 - pkg/cri/sbserver/container_log_reopen.go
func (c *criService) ReopenContainerLog(ctx context.Context, r *runtime.ReopenContainerLogRequest) {
    // 重新打开容器的 stdout/stderr 日志文件
    // 这通常在日志文件被轮转后调用
}
```

**区别对比**：

| 特性     | 容器日志 (stdout/stderr)     | runc log.json   |
| -------- | ---------------------------- | --------------- |
| 轮转支持 | ✅ 支持                      | ❌ 不支持       |
| 配置方式 | CRI 配置、Docker daemon.json | runc --log 参数 |
| 管理机制 | containerd/CRI 管理          | runc 直接写入   |
| 用途     | 应用程序输出                 | 运行时操作日志  |

补充说明：runc 的 log.json 的保留与清理通常由任务生命周期或外部策略决定，不在 CRI 容器日志轮转范围内。

### 1.4 解决方案

NVIDIA 官方针对这个问题提供了解决方案([PR #560](https://github.com/NVIDIA/nvidia-container-toolkit/pull/560))

> "These changes reduce the verbosity of the logging of the NVIDIA Container Runtime -- especially for the case where no modifications are required."

**核心改进**：

1. **降低默认日志级别**：将不必要的 info 级别日志调整为 debug 级别
2. **减少重复配置输出**：避免在每次操作时都输出完整的运行时配置
3. **条件性日志记录**：仅在需要修改时才记录详细信息

**配置调整方法**：

**1. 配置文件方式**：

```toml
# /etc/nvidia-container-runtime/config.toml
[nvidia-container-runtime]
log-level = "error"  # 从 "info" 改为 "error"
```

**2. 环境变量方式**：

```bash
# 通过 XDG_CONFIG_HOME 环境变量指定自定义配置路径
export XDG_CONFIG_HOME=/path/to/custom/config
# 在 ${XDG_CONFIG_HOME}/nvidia-container-runtime/config.toml 中设置日志级别
```

**3. 定期清理脚本**：

```bash
#!/bin/bash
# 专门针对 NVIDIA Container Runtime 日志的清理脚本
find /run/containerd/io.containerd.runtime.v2.task -name "log.json" -size +50M \
  -exec grep -l "nvidia-container-runtime" {} \; \
  -exec truncate -s 0 {} \;
```

**注意事项**：

- 对 log.json 进行 truncate 操作需在维护窗口或确认无并发写入的情况下执行，避免造成日志损坏或竞争问题。
- NVIDIA 相关 PR 与配置的适用性依赖具体版本与部署环境，建议在生产环境前进行充分测试验证。

---

## 2. containerd 运行时日志深入解析

### 2.1 概述与作用边界

`log.json` 由 runc 写出，用于记录运行时错误与调试信息。containerd 在容器启动失败时，会读取其中的最后一条错误消息辅助定位。它采用 JSON 格式存储日志条目，便于程序化解析与排查。

**重要说明**： `log.json` 记录的是 **runc 运行时本身的操作信息**，而不是容器内进程的标准输出（stdout/stderr）。容器内进程的输出通过其他机制（如 containerd 的 CIO 系统）进行处理。

### 2.2 什么时候该看 log.json

当容器启动失败或运行异常时，containerd 会调用 `getLastRuntimeError()` 函数读取 `log.json` 文件，获取最新的错误信息用于诊断。

在生产环境中，`log.json` 的主要用途包括：针对容器创建或启动失败的快速定位、运行时配置与兼容性问题的系统化排查、以及资源和权限约束导致的异常诊断；调试侧用于还原 runc 的执行路径与系统调用状态；运维与监控侧可按需读取最新错误记录用于健康度评估与异常告警；事后分析与合规审计侧则作为运行时关键操作的留痕。

使用时建议优先关注 error 级别的最后一条 msg 字段，并结合时间戳与调用关系上下文交叉验证，以缩短根因确认时间；观察文件尺寸与增长速度，判断是否存在异常写入并避免占满 /run 的 tmpfs；排查过程中以只读方式打开文件，避免与 runc 并发写入产生竞争。

```bash
# 结合时间和级别快速定位错误（示例命令）
# 注意：以下命令均为只读操作，安全用于生产排查
grep '"level":"error"' /run/containerd/io.containerd.runtime.v2.task/k8s.io/<container-id>/log.json | tail -n 50
```

---

### 2.3 调用关系与架构位置

#### 2.3.1 调用关系图

```text
容器启动请求
    ↓
Runtime V2 Manager
    ↓
NewBundle() 创建 Bundle
    ↓
NewRunc() 创建 Runc 实例
    ↓
设置 Log 路径: bundle_path/log.json
设置 LogFormat: runc.JSON
    ↓
Runc 命令执行
    ↓
args() 构建命令参数
    ↓
runc --log log.json --log-format json
    ↓
runc 运行时写入日志到 log.json
    ↓
错误发生时调用 getLastRuntimeError()
    ↓
读取并解析 log.json
    ↓
返回最后一条错误消息给上层调用者
```

#### 2.3.2 架构集成概览

`log.json` 文件在 containerd 架构中的位置：

```text
┌─────────────────────────────────────────┐
│            containerd API               │
├─────────────────────────────────────────┤
│         Runtime V2 Manager              │
├─────────────────────────────────────────┤
│    Bundle Management    │  Shim Manager │
├─────────────────────────┼───────────────┤
│      Runc Instance      │   Log System  │
├─────────────────────────┼───────────────┤
│       log.json          │   CIO System  │
└─────────────────────────────────────────┘
```

**各组件功能说明**：

- **containerd API**：对外提供容器管理的 gRPC 接口，处理客户端请求
- **Runtime V2 Manager**：管理容器运行时，负责协调各个子组件的工作
- **Bundle Management**：管理容器 Bundle（包含配置文件和根文件系统的目录）
- **Shim Manager**：管理容器 Shim 进程，提供容器生命周期管理
- **Runc Instance**：OCI 运行时实例，负责实际的容器创建和管理
- **Log System**：处理容器 stdout/stderr 日志与轮转（容器日志），不管理 runc 的 log.json
- **log.json**：Runc 运行时的 JSON 格式日志文件，记录详细的运行时信息
- **CIO System**：容器 I/O 系统，管理容器的标准输入输出流

#### 2.3.3 数据流向

1. **写入流程**：containerd → Runtime V2 → Runc → log.json
2. **读取流程**：containerd ← getLastRuntimeError() ← log.json
3. **监控流程**：运维/监控系统 → 按需读取 → log.json

---

### 2.4 源码入口与关键片段

注：以下源码路径与函数名称基于 containerd 上游仓库（v1.6/v1.7 分支）与 go-runc；不同版本的实现细节可能有所差异，建议结合当前环境源码核对。参考：containerd 仓库 <https://github.com/containerd/containerd> ，go-runc 仓库 <https://github.com/containerd/go-runc>。

#### 2.4.1 日志文件路径管理

**主要文件**： `pkg/process/init.go`

**关键函数**： `NewRunc`

```go
// 日志文件路径构建逻辑
func NewRunc(root, path, namespace, runtime string, config map[string]string) *runc.Runc {
    // ...
    return &runc.Runc{
        Command:      runtime,
        Log:          filepath.Join(path, "log.json"), // 关键：log.json 路径设置
        LogFormat:    runc.JSON,                       // 设置为 JSON 格式
        PdeathSignal: unix.SIGKILL,
        Setpgid:      true,
        // ...
    }
}
```

**功能说明**：

- 在容器的 bundle 目录中创建 `log.json` 文件
- 设置日志格式为 JSON 格式
- 配置 runc 运行时的日志输出参数

#### 2.4.2 日志格式定义

**主要文件**： `vendor/github.com/containerd/go-runc/runc.go`

**日志格式常量**：

```go
// Format 类型定义
type Format string

const (
    none Format = ""
    JSON Format = "json"  // JSON 格式标识
    Text Format = "text"
)
```

**Runc 结构体定义**：

```go
// vendor/github.com/containerd/go-runc/runc_unix.go
type Runc struct {
    Command      string
    Root         string
    Debug        bool
    Log          string    // 日志文件路径
    LogFormat    Format    // 日志格式
    PdeathSignal syscall.Signal
    Setpgid      bool
    // ...
}
```

#### 2.4.3 日志结构体定义

**主要文件**： `pkg/process/utils.go`

**日志条目结构**：

```go
// log.json 中每条日志的数据结构
var log struct {
    Level string     // 日志级别（如 "error", "info", "debug"）
    Msg   string     // 日志消息内容
    Time  time.Time  // 时间戳
}
```

**字段说明**：

- `Level`: 日志级别，包括 "error"、"info"、"debug" 等
- `Msg`: 具体的日志消息内容
- `Time`: 日志记录的时间戳

---

### 2.5 读写机制与数据流

#### 2.5.1 日志文件创建和写入

**调用链路**：

1. **容器启动** → `NewRunc()` 函数
2. **Runc 配置** → 设置 Log 和 LogFormat 字段
3. **命令参数构建** → `args()` 函数
4. **Runc 执行** → 写入日志到 log.json

**关键代码片段**：

```go
// runc.go 中的参数构建逻辑
func (r *Runc) args() []string {
    var args []string
    if r.Log != "" {
        args = append(args, "--log", r.Log)
    }
    if r.LogFormat != none {
        args = append(args, "--log-format", string(r.LogFormat))
    }
    return args
}
```

**写入流程**：

1. containerd 启动容器时调用 `NewRunc()` 创建 Runc 实例
2. 设置 `Log` 字段为 `bundle_path/log.json`
3. 设置 `LogFormat` 字段为 `runc.JSON`
4. runc 运行时根据配置将日志写入指定文件

#### 2.5.2 日志文件读取和解析

**主要函数**： `getLastRuntimeError`

**功能描述**：

- 打开 `log.json` 文件进行只读访问
- 使用 JSON 解码器逐行解析日志条目
- 筛选错误级别的日志消息
- 返回最后一条错误消息用于故障诊断

**示意实现**：

```go
func getLastRuntimeError(r *runc.Runc) (string, error) {
    // 检查日志文件路径是否配置
    if r.Log == "" {
        return "", nil
    }

    // 以只读模式打开日志文件
    f, err := os.OpenFile(r.Log, os.O_RDONLY, 0400)
    if err != nil {
        return "", err
    }
    defer f.Close()

    var (
        errMsg string
        log    struct {
            Level string     // 日志级别
            Msg   string     // 日志消息
            Time  time.Time  // 时间戳
        }
    )

    // 创建 JSON 解码器
    dec := json.NewDecoder(f)

    // 逐行解析日志条目
    for err = nil; err == nil; {
        if err = dec.Decode(&log); err != nil && err != io.EOF {
            return "", err
        }
        // 筛选错误级别的日志
        if log.Level == "error" {
            errMsg = strings.TrimSpace(log.Msg)
        }
    }

    return errMsg, nil
}
```

**读取特点**：

- 只读取错误级别的日志消息
- 返回最后一条错误消息（最新的错误）
- 使用流式解析，内存效率高
- 自动处理文件结束标志

---

### 2.6 Runtime V2 集成与 Bundle/Shim 生命周期

#### 2.6.1 Runtime V2 架构概述

containerd Runtime V2 是 containerd 的新一代运行时架构。与 Runtime V1 相比，其隔离与可扩展性更好。`log.json` 的路径随 Bundle 确定，生命周期更可预测，便于定位与诊断。

##### 2.6.1.1 Runtime V2 核心特点

**Runtime V2 的核心特点**：

1. **Shim 进程模型**：每个容器都有独立的 shim 进程，提供更好的隔离性
2. **标准化接口**：通过 gRPC 接口实现运行时的标准化管理
3. **插件化设计**：支持不同的运行时实现（如 runc、kata-containers 等）
4. **改进的生命周期管理**：更精确的容器状态管理和资源清理

##### 2.6.1.2 Runtime V2 架构组件分层与职责划分

```text
┌────────────────────────────────────────────────────────────────────┐
│                    containerd                                      │
├────────────────────────────────────────────────────────────────────┤
│                Runtime V2 Manager                                  │
├─────────────────┬───────────────────┬──────────────────────────────┤
│   Bundle        │    Shim           │   Logging                    │
│   Management    │    Management     │   System                     │
├─────────────────┼───────────────────┼──────────────────────────────┤
│ • Bundle 创建    │ • Shim 启动        │ • 容器 stdout/stderr 日志处理 │
│ • 路径管理       │ • 进程监控          │ • 日志轮转（容器日志）          │
│ • 资源清理       │ • 状态同步          │ • 错误收集（容器日志）          │
└─────────────────┴───────────────────┴──────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     runc      │
                    │  (OCI Runtime)│
                    └───────────────┘
```

**Runtime V2 组件功能说明**：

- **containerd**：容器管理守护进程，提供高级容器管理功能
- **Runtime V2 Manager**：新一代运行时管理器，协调各个子系统的工作
- **Bundle Management**：Bundle 创建、路径管理、资源清理
- **Shim Management**：Shim 启动、进程监控、状态同步
- **Logging System**：容器 stdout/stderr 日志处理、日志轮转（容器日志）、错误收集（容器日志）
- **runc (OCI Runtime)**：符合 OCI 标准的底层容器运行时，负责实际的容器操作

#### 2.6.2 Bundle 生命周期管理机制

**主要文件**： `runtime/v2/bundle.go`

Bundle 是 Runtime V2 中容器工作目录的抽象，每个容器都有独立的 Bundle，其中包含了 `log.json` 文件。

**Bundle 结构定义**：

```go
type Bundle struct {
    ID        string  // 容器 ID
    Path      string  // Bundle 路径（包含 log.json）
    Namespace string  // 命名空间
}
```

**Bundle 创建流程**：

```go
// NewBundle 创建新的 Bundle
func NewBundle(ctx context.Context, root, state, id string, spec typeurl.Any) (b *Bundle, err error) {
    // 验证容器 ID
    if err := identifiers.Validate(id); err != nil {
        return nil, fmt.Errorf("invalid task id %s: %w", id, err)
    }

    // 获取命名空间
    ns, err := namespaces.NamespaceRequired(ctx)
    if err != nil {
        return nil, err
    }

    // 构建 Bundle 路径
    work := filepath.Join(root, ns, id)
    b = &Bundle{
        ID:        id,
        Path:      filepath.Join(state, ns, id), // log.json 将位于此路径下
        Namespace: ns,
    }

    // 创建目录结构
    // ...
}
```

#### 2.6.3 Shim 进程管理

**主要文件**： `runtime/v2/shim.go`

Shim 进程是 Runtime V2 架构的核心组件，负责管理单个容器的生命周期，包括 `log.json` 文件的创建和维护。

**Shim 接口定义**：

```go
type ShimInstance interface {
    ID() string
    Namespace() string
    Bundle() string
    Close() error
    Delete(context.Context) (*types.Exit, error)
}
```

**关键组件**：

- `ShimInstance`: Shim 实例接口
- `loadShim`: Shim 加载逻辑
- `shim`: Shim 实现结构体

#### 2.6.4 Runtime V2 日志系统集成

**主要文件**： `runtime/v2/logging/logging.go`

Runtime V2 的日志系统统一管理容器进程的 stdout/stderr 日志；`log.json` 为 runc 运行时诊断日志，独立于容器日志系统，由 runc 输出。

**日志配置结构**：

```go
// Config 日志配置结构
type Config struct {
    ID        string     // 容器 ID
    Namespace string     // 命名空间
    Stdout    io.Reader  // 标准输出
    Stderr    io.Reader  // 标准错误
}

// LoggerFunc 自定义日志函数类型
type LoggerFunc func(ctx context.Context, cfg *Config, ready func() error) error
```

**日志驱动实现**：

- Unix 系统：`runtime/v2/logging/logging_unix.go`
- Windows 系统：`runtime/v2/logging/logging_windows.go`

##### 2.6.4.1 log.json 在 Runtime V2 中的角色

在 Runtime V2 架构中，`log.json` 文件扮演着关键的诊断和监控角色：

- **Bundle 级别管理**：每个 Bundle 都有独立的 log.json 文件
- **Shim 进程集成**：通过 shim 进程管理日志的生命周期
- **标准化路径**：遵循 Runtime V2 的标准目录结构
- **错误传播**：为上层提供标准化的错误信息接口

注意：`log.json` 独立于容器日志驱动体系，由 runc 输出；容器日志系统不对其进行轮转管理。

### 2.7 核心源码文件分布与功能映射表

| **组件**      | **文件路径**                                        | **主要功能**             |
| ------------- | --------------------------------------------------- | ------------------------ |
| 日志路径配置  | `pkg/process/init.go`                               | 设置 log.json 路径和格式 |
| 日志读取解析  | `pkg/process/utils.go`                              | 读取和解析 log.json 内容 |
| Runc 集成     | `vendor/github.com/containerd/go-runc/runc.go`      | go-runc 库的核心实现     |
| Runc 结构定义 | `vendor/github.com/containerd/go-runc/runc_unix.go` | Runc 结构体定义          |
| Bundle 管理   | `runtime/v2/bundle.go`                              | Bundle 结构和路径管理    |
| Shim 管理     | `runtime/v2/shim.go`                                | Shim 实例管理            |
| 日志系统      | `runtime/v2/logging/logging.go`                     | Runtime V2 日志配置      |
| Manager 管理  | `runtime/v2/manager.go`                             | Runtime V2 管理器        |

---

## 3. 总结

`log.json` 是 containerd 运行时诊断的重要文件之一，采用 JSON 结构记录 `runc` 的运行时信息，便于程序化处理与检索。它支持流式解析，读取开销较低，适合在线上环境进行快速排查；同时关注 `level=error` 的最后一条消息可以直观指向问题根因；其路径随 Bundle 目录确定，定位与维护具有可预测性。

需要注意其关键限制与风险：在默认配置下，`runc` 的 `log.json` 不具备自动轮转能力，长期运行可能导致日志持续增长。生产环境应进行存储规划与定期清理，并对文件大小进行监控，避免占满 `/run` 的 tmpfs 或持久化分区，从而影响新容器创建与节点稳定性。

实践上，建议在排查时以只读方式打开该文件，结合时间戳与调用链上下文交叉验证错误；在监控策略中为异常增长设置阈值与告警，并在必要时通过外部轮转或清理机制进行治理，以保持系统的可用性与可维护性。

总结：`log.json` 不是容器日志系统的一部分，而是 `runc` 的运行时诊断输出。它能帮助快速定位创建失败与运行异常的原因，但也需要配合监控与清理策略，避免无限增长影响节点稳定性。

---
