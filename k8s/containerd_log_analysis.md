# 容易被忽略的 containerd 运行时日志

在容器化环境中，我们通常关注的是容器进程本身产生的日志（标准输出、标准错误），这些日志默认存储在 `/var/log/containers/<container_id>-<namespace>-<name>.log` 文件中。然而，容器运行时本身也会产生重要的运行时日志，这些日志往往被忽略，但它们对于诊断容器创建失败、Pod 启动异常等问题至关重要。

containerd 运行时日志（以 `log.json` 文件形式存储）就是这样一个容易被忽略但非常重要的组件。它由 runc 运行时生成，记录了容器的运行时操作信息，通常位于 `/run/containerd/` 目录下。当这个日志文件出现异常增长或损坏时，可能导致无法创建新的容器或 Pod，严重影响集群的正常运行。

## 1. containerd 运行时日志增长问题及其影响

### 1.1 问题概述

在 containerd 生产环境中，`log.json` 文件的无限制增长已成为一个严重的运维问题。该文件记录 `runc` 运行时的操作信息，但由于缺乏自动轮转机制，可能导致磁盘空间耗尽，进而影响整个容器平台的稳定性。

**核心问题：** runc 的 `log.json` 文件本身不支持自动日志轮转功能，会持续向指定文件追加日志，直到磁盘（如果是 tmpfs，则是内存）空间耗尽。

### 1.2 实际案例分析

#### 1.2.1 案例一：NVIDIA Container Runtime 配置重复记录

**问题描述：**

- 日志文件大小：`32085414 bytes (约 30.6 MB)`
- 重复记录 NVIDIA Container Runtime 的完整配置信息
- 每次记录包含约 1KB 的 JSON 配置数据
- 时间间隔较短（几秒到几十秒）

**典型日志模式：**

```json
{"level":"info","msg":"Running with config:\n{\n  \"AcceptEnvvarUnprivileged\": true,\n  \"NVIDIAContainerCLIConfig\": {\n    \"Root\": \"\"\n  },\n  \"NVIDIACTKConfig\": {\n    \"Path\": \"nvidia-ctk\"\n  },\n  \"NVIDIAContainerRuntimeConfig\": {\n    \"DebugFilePath\": \"/dev/null\",\n    \"LogLevel\": \"info\",\n    \"Runtimes\": [\n      \"docker-runc\",\n      \"runc\"\n    ],\n    \"Mode\": \"auto\"\n  }\n}","time":"2024-01-20T16:05:43+08:00"}
{"level":"info","msg":"Using low-level runtime /usr/bin/runc","time":"2024-01-20T16:05:43+08:00"}
```

#### 1.2.2 案例二：GitHub Issue #8972 - 生产环境节点故障

containerd 官方 GitHub 仓库中报告了一个严重的生产环境问题([Issue #8972](https://github.com/containerd/containerd/issues/8972))

> "log.json of a container may grow to burst the tmpfs of /run, if a k8s user configure an exec liveness probe of a non exist executable file name."

**问题影响：**

- 当 Kubernetes 用户配置了不存在的可执行文件的 exec liveness probe 时
- log.json 文件会快速增长，最终撑爆 `/run` 目录的 tmpfs
- 导致节点无法创建新的容器，影响整个集群的可用性

#### 1.2.3 案例三：NVIDIA/nvidia-container-toolkit#511 - 大规模部署失败

NVIDIA Container Toolkit 项目中报告的问题([Issue #511](https://github.com/NVIDIA/nvidia-container-toolkit/issues/511))

> "Excessive runtime logging could cause Kubernetes workload deployment failure"

**影响范围：**

- `/run/containerd/io.containerd.runtime.v2.task/k8s.io//log.json` 文件过大
- `/run` tmpfs 挂载点达到 100% 利用率
- 阻止在受影响节点上进一步创建容器

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

**区别对比：**

| 特性 | 容器日志 (stdout/stderr) | runc log.json |
|------|-------------------------|---------------|
| 轮转支持 | ✅ 支持 | ❌ 不支持 |
| 配置方式 | CRI 配置、Docker daemon.json | runc --log 参数 |
| 管理机制 | containerd/CRI 管理 | runc 直接写入 |
| 用途 | 应用程序输出 | 运行时操作日志 |

### 1.4 解决方案

NVIDIA 官方针对这个问题提供了解决方案([PR #560](https://github.com/NVIDIA/nvidia-container-toolkit/pull/560))

> "These changes reduce the verbosity of the logging of the NVIDIA Container Runtime -- especially for the case where no modifications are required."

**核心改进：**

1. **降低默认日志级别**：将不必要的 info 级别日志调整为 debug 级别
2. **减少重复配置输出**：避免在每次操作时都输出完整的运行时配置
3. **条件性日志记录**：仅在需要修改时才记录详细信息

**配置调整方法：**

**1. 配置文件方式：**

```toml
# /etc/nvidia-container-runtime/config.toml
[nvidia-container-runtime]
log-level = "error"  # 从 "info" 改为 "error"
```

**2. 环境变量方式：**

```bash
# 通过 XDG_CONFIG_HOME 环境变量指定自定义配置路径
export XDG_CONFIG_HOME=/path/to/custom/config
# 在 ${XDG_CONFIG_HOME}/nvidia-container-runtime/config.toml 中设置日志级别
```

**定期清理脚本：**

```bash
#!/bin/bash
# 专门针对 NVIDIA Container Runtime 日志的清理脚本
find /run/containerd/io.containerd.runtime.v2.task -name "log.json" -size +50M \
  -exec grep -l "nvidia-container-runtime" {} \; \
  -exec truncate -s 0 {} \;
```

---

## 2. containerd 运行时日志概述与作用机制

### 2.1 概述

`log.json` 文件是 containerd 中用于记录 runc 运行时错误和调试信息的重要组件。它采用 JSON 格式存储日志条目，主要用于容器运行时的错误诊断和故障排查。

**重要说明：** `log.json` 记录的是 **runc 运行时本身的操作信息**，而不是容器内进程的标准输出（stdout/stderr）。容器内进程的输出通过其他机制（如 containerd 的 CIO 系统）进行处理。

### 2.2 主要作用和使用场景

#### 2.2.1 错误诊断

当容器启动失败或运行异常时，containerd 会调用 `getLastRuntimeError()` 函数读取 `log.json` 文件，获取最新的错误信息用于诊断。

**典型场景：**

- 容器创建失败时的错误定位
- 运行时配置错误的排查
- 资源限制导致的启动失败分析

#### 2.2.2 调试支持

开发人员可以通过查看 `log.json` 文件了解 runc 的详细执行过程，包括：

- 容器创建过程的详细步骤
- 资源配置信息的验证结果
- 运行时错误的详细堆栈信息
- 系统调用的执行状态

#### 2.2.3 监控集成

监控系统可以定期读取 `log.json` 文件，提取关键信息用于：

- 容器健康状态监控
- 错误率统计和趋势分析
- 性能瓶颈识别
- 运行时异常告警

#### 2.2.4 故障排查

在生产环境中，`log.json` 提供了重要的故障排查能力：

- **事后分析**：容器异常退出后的原因分析
- **实时监控**：运行时错误的即时发现
- **性能调优**：识别运行时性能问题
- **合规审计**：记录容器运行时的关键操作

---

## 3. containerd 运行时日志系统调用关系与架构分析

### 3.1 调用关系图

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

### 3.2 架构集成概览

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

**各组件功能说明：**

- **containerd API**：对外提供容器管理的 gRPC 接口，处理客户端请求
- **Runtime V2 Manager**：管理容器运行时，负责协调各个子组件的工作
- **Bundle Management**：管理容器 Bundle（包含配置文件和根文件系统的目录）
- **Shim Manager**：管理容器 Shim 进程，提供容器生命周期管理
- **Runc Instance**：OCI 运行时实例，负责实际的容器创建和管理
- **Log System**：日志管理系统，处理容器运行时的日志收集和存储
- **log.json**：Runc 运行时的 JSON 格式日志文件，记录详细的运行时信息
- **CIO System**：容器 I/O 系统，管理容器的标准输入输出流

### 3.3 数据流向

1. **写入流程**：containerd → Runtime V2 → Runc → log.json
2. **读取流程**：containerd ← getLastRuntimeError() ← log.json
3. **监控流程**：监控系统 → 定期读取 → log.json

---

## 4. containerd 运行时日志相关核心代码分析

### 4.1 日志文件路径管理

**主要文件：** `pkg/process/init.go`

**关键函数：** `NewRunc`

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

**功能说明：**

- 在容器的 bundle 目录中创建 `log.json` 文件
- 设置日志格式为 JSON 格式
- 配置 runc 运行时的日志输出参数

### 4.2 日志格式定义

**主要文件：** `vendor/github.com/containerd/go-runc/runc.go`

**日志格式常量：**

```go
// Format 类型定义
type Format string

const (
    none Format = ""
    JSON Format = "json"  // JSON 格式标识
    Text Format = "text"
)
```

**Runc 结构体定义：**

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

### 4.3 日志结构体定义

**主要文件：** `pkg/process/utils.go`

**日志条目结构：**

```go
// log.json 中每条日志的数据结构
var log struct {
    Level string     // 日志级别（如 "error", "info", "debug"）
    Msg   string     // 日志消息内容
    Time  time.Time  // 时间戳
}
```

**字段说明：**

- `Level`: 日志级别，包括 "error"、"info"、"debug" 等
- `Msg`: 具体的日志消息内容
- `Time`: 日志记录的时间戳

---

## 5. containerd 运行时日志读写机制

### 5.1 日志文件创建和写入

**调用链路：**

1. **容器启动** → `NewRunc()` 函数
2. **Runc 配置** → 设置 Log 和 LogFormat 字段
3. **命令参数构建** → `args()` 函数
4. **Runc 执行** → 写入日志到 log.json

**关键代码片段：**

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

**写入流程：**

1. containerd 启动容器时调用 `NewRunc()` 创建 Runc 实例
2. 设置 `Log` 字段为 `bundle_path/log.json`
3. 设置 `LogFormat` 字段为 `runc.JSON`
4. runc 运行时根据配置将日志写入指定文件

### 5.2 日志文件读取和解析

**主要函数：** `getLastRuntimeError`

**功能描述：**

- 打开 `log.json` 文件进行只读访问
- 使用 JSON 解码器逐行解析日志条目
- 筛选错误级别的日志消息
- 返回最后一条错误消息用于故障诊断

**完整实现：**

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

**读取特点：**

- 只读取错误级别的日志消息
- 返回最后一条错误消息（最新的错误）
- 使用流式解析，内存效率高
- 自动处理文件结束标志

---

## 6. containerd Runtime V2 日志系统简介

### 6.1 Runtime V2 架构概述

containerd Runtime V2 是 containerd 的新一代运行时架构，相比 Runtime V1 提供了更好的性能、稳定性和扩展性。在 Runtime V2 架构中，`log.json` 文件的管理和集成更加系统化和标准化。

#### 6.1.1 Runtime V2 核心特点

**Runtime V2 的核心特点：**

1. **Shim 进程模型**：每个容器都有独立的 shim 进程，提供更好的隔离性
2. **标准化接口**：通过 gRPC 接口实现运行时的标准化管理
3. **插件化设计**：支持不同的运行时实现（如 runc、kata-containers 等）
4. **改进的生命周期管理**：更精确的容器状态管理和资源清理

#### 6.1.2 Runtime V2 架构组件分层与职责划分

```text
┌─────────────────────────────────────────────────────────┐
│                    containerd                           │
├─────────────────────────────────────────────────────────┤
│                Runtime V2 Manager                       │
├─────────────────┬───────────────────┬───────────────────┤
│   Bundle        │    Shim           │   Logging         │
│   Management    │    Management     │   System          │
├─────────────────┼───────────────────┼───────────────────┤
│ • Bundle 创建    │ • Shim 启动        │ • log.json 管理   │
│ • 路径管理       │ • 进程监控          │ • 日志轮转         │
│ • 资源清理       │ • 状态同步          │ • 错误收集         │
└─────────────────┴───────────────────┴───────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     runc      │
                    │  (OCI Runtime)│
                    └───────────────┘
```

**Runtime V2 组件功能说明：**

- **containerd**：容器管理守护进程，提供高级容器管理功能
- **Runtime V2 Manager**：新一代运行时管理器，协调各个子系统的工作
- **Bundle Management**：Bundle 创建、路径管理、资源清理
- **Shim Management**：Shim 启动、进程监控、状态同步
- **Logging System**：log.json 管理、日志轮转、错误收集
- **runc (OCI Runtime)**：符合 OCI 标准的底层容器运行时，负责实际的容器操作

### 6.2 Bundle 生命周期管理机制

**主要文件：** `runtime/v2/bundle.go`

Bundle 是 Runtime V2 中容器工作目录的抽象，每个容器都有独立的 Bundle，其中包含了 `log.json` 文件。

#### 6.2.1 Bundle 结构定义

```go
type Bundle struct {
    ID        string  // 容器 ID
    Path      string  // Bundle 路径（包含 log.json）
    Namespace string  // 命名空间
}
```

#### 6.2.2 Bundle 创建流程

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

### 6.3 Shim 进程管理

**主要文件：** `runtime/v2/shim.go`

Shim 进程是 Runtime V2 架构的核心组件，负责管理单个容器的生命周期，包括 `log.json` 文件的创建和维护。

#### 6.3.1 Shim 接口定义

```go
type ShimInstance interface {
    ID() string
    Namespace() string
    Bundle() string
    Close() error
    Delete(context.Context) (*types.Exit, error)
}
```

#### 6.3.2 关键组件

- `ShimInstance`: Shim 实例接口
- `loadShim`: Shim 加载逻辑
- `shim`: Shim 实现结构体

### 6.4 Runtime V2 日志系统集成

**主要文件：** `runtime/v2/logging/logging.go`

Runtime V2 的日志系统提供了统一的日志管理接口，`log.json` 文件是其重要组成部分。

#### 6.4.1 日志配置结构

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

#### 6.4.2 日志驱动实现

- Unix 系统：`runtime/v2/logging/logging_unix.go`
- Windows 系统：`runtime/v2/logging/logging_windows.go`

#### 6.4.3 log.json 在 Runtime V2 中的角色

在 Runtime V2 架构中，`log.json` 文件扮演着关键的诊断和监控角色：

- **Bundle 级别管理**：每个 Bundle 都有独立的 log.json 文件
- **Shim 进程集成**：通过 shim 进程管理日志的生命周期
- **标准化路径**：遵循 Runtime V2 的标准目录结构
- **错误传播**：为上层提供标准化的错误信息接口

### 6.5 核心源码文件分布与功能映射表

| 组件 | 文件路径 | 主要功能 |
|------|----------|----------|
| 日志路径配置 | `pkg/process/init.go` | 设置 log.json 路径和格式 |
| 日志读取解析 | `pkg/process/utils.go` | 读取和解析 log.json 内容 |
| Runc 集成 | `vendor/github.com/containerd/go-runc/runc.go` | go-runc 库的核心实现 |
| Runc 结构定义 | `vendor/github.com/containerd/go-runc/runc_unix.go` | Runc 结构体定义 |
| Bundle 管理 | `runtime/v2/bundle.go` | Bundle 结构和路径管理 |
| Shim 管理 | `runtime/v2/shim.go` | Shim 实例管理 |
| 日志系统 | `runtime/v2/logging/logging.go` | Runtime V2 日志配置 |
| Manager 管理 | `runtime/v2/manager.go` | Runtime V2 管理器 |

---

## 7. 总结

`log.json` 文件是 containerd 日志系统的核心组件，通过结构化的 JSON 格式记录 runc 运行时信息。其设计具有以下优势：

1. **结构化存储**：便于程序化处理和分析
2. **错误聚焦**：专注于错误信息的捕获和诊断
3. **性能优化**：采用流式处理和按需读取
4. **架构集成**：与 Runtime V2 深度集成
5. **扩展支持**：支持自定义日志驱动和配置

**关键限制和注意事项：**

1. **无自动轮转**：runc 的 log.json 不支持自动日志轮转，需要外部管理
2. **持续增长**：长期运行的容器会产生大量日志，需要定期清理
3. **存储规划**：需要为运行时日志预留足够的存储空间
4. **监控必要**：生产环境中需要监控日志文件大小，防止磁盘空间耗尽

该系统为 containerd 提供了强大的故障诊断和调试能力，是容器运行时管理的重要基础设施。通过合理的架构设计和高效的实现方式，结合适当的日志管理策略，`log.json` 在保证性能的同时，为开发者和运维人员提供了丰富的调试信息和故障排查能力。

---
