# Agent Sandbox 的演进与设计范式

> **摘要**：在生成式 AI 时代，大语言模型驱动的 Agent 正逐步接管复杂的工程任务。随着 Agent 能力的增强，如何为其提供一个安全、可控且高效的执行环境（Sandbox）成为了基础设施建设的核心挑战。本文将通过对比 OpenShell 的重型容器化方案与 Sandlock、OpenShell Lite 的轻量级进程沙箱，探讨 Agent Sandbox 的核心设计理念，并揭示从“硬件级隔离”向“策略优先”演进的技术趋势。

![agent-sandbox-design](./assets/agent-sandbox.png)

## 目录

- [1. Agent 安全模型的认知重构](#1-agent-安全模型的认知重构)
- [2. OpenShell：基于容器与网络的纵深防御](#2-openshell基于容器与网络的纵深防御)
- [3. 赋能轻量级沙箱的新内核技术：Landlock 解析](#3-赋能轻量级沙箱的新内核技术landlock-解析)
- [4. Sandlock：回归进程级管控的轻量化革命](#4-sandlock回归进程级管控的轻量化革命)
- [5. OpenShell Lite：云原生时代的轻量级协同沙箱](#5-openshell-lite云原生时代的轻量级协同沙箱)
- [6. 策略优先：下一代 Agent Sandbox 的设计范式](#6-策略优先下一代-agent-sandbox-的设计范式)
- [7. 结语](#7-结语)
- [参考文献](#参考文献)

---

## 1. Agent 安全模型的认知重构

在传统云计算架构中，沙箱主要用于防御主动尝试逃逸的恶意租户。然而，AI Agent 引入了全新的交互模式，其安全威胁和防御重点发生了根本性转移，这要求基础设施团队重新审视隔离方案的设计初衷。

### 1.1 威胁模型的转移：从主动攻击到被动注入

AI Agent 本身并非恶意攻击者，而是按指令执行代码的语言模型。在实际的生产环境中，Agent 的指令集由开发者编写，真正的安全威胁面来源于外部内容的 Prompt 注入。当 Agent 抓取网页、读取文档或调用 API 时，可能无意中摄入了包含恶意指令的文本。这类攻击通过篡改模型上下文，诱导 Agent 执行如 `curl` 窃取数据、`rm` 删除文件等危险操作。因此，防御的核心不在于抵御高维度的内核提权，而在于限制普通进程的系统调用和网络访问。

### 1.2 隔离的迷思：物理隔离不等于逻辑安全

行业内普遍倾向于采用微虚拟机（如 Firecracker）或重型容器来封装 Agent，以期获得军事级的隔离效果。然而，Agent 在执行合法任务时通常需要访问敏感资源（如读取 `~/.ssh` 中的私钥以拉取代码）。如果仅在硬件或容器层进行粗粒度隔离，一旦敏感凭证进入沙箱内部，便完全暴露在潜在的 Prompt 注入攻击下。真正的逻辑安全依赖于细粒度的访问控制策略，而非单一的硬件级屏障。

---

## 2. OpenShell：基于容器与网络的纵深防御

OpenShell 代表了当前主流的重量级 Agent Sandbox 设计思路 [3]，它通过 Kubernetes（K3s）、Docker 容器以及 HTTP CONNECT 代理，构建了一套边界清晰、防御层次丰富的安全体系。

### 2.1 架构与组件协同

OpenShell 的架构由控制面（Gateway）和数据面（Sandbox Pod）组成，所有组件被封装在一个 K3s 集群中。Gateway 负责沙箱的生命周期管理与安全策略分发；Sandbox Pod 则作为独立容器运行具体的 Agent 进程。Agent 进程运行在受限的 Network Namespace 中，其所有出站网络流量均被强制路由至 HTTP CONNECT 代理，实现了从底层网络到应用层的全面接管。

### 2.2 HTTP 代理与 OPA 策略引擎

网络管控是 OpenShell 纵深防御的核心。其 HTTP CONNECT 代理内置了 OPA（Open Policy Agent）策略引擎，所有网络请求必须经过严格的 Rego 规则校验。
该机制不仅实现了基于域名和端口的 L4 拦截，还能通过自动的 TLS 终止（TLS MITM）进行 L7 HTTP 协议解析。这种设计允许平台为不同域名配置细粒度的访问策略（例如仅允许对 `api.github.com` 发起 `GET` 请求），并在代理层完成 API 凭证的动态注入，确保敏感密钥绝对不落盘、不进入 Agent 的文件系统。

```yaml
# OpenShell 策略示例：仅允许对 GitHub API 发起只读请求
network:
  rules:
    - host: api.github.com # 允许访问的外部域名
      port: 443 # 目标端口
      protocol: rest # 启用的应用层协议解析
      access: read-only # 访问级别：只允许 GET/HEAD/OPTIONS 方法
```

### 2.3 重量级架构的优劣权衡

OpenShell 的容器化集群架构提供了极高的安全性与完整的功能特性，但也引入了显著的性能与运维成本。完整的 K3s 和容器启动流程使得沙箱的冷启动时间较长，难以满足高并发、短生命周期的函数级调用需求。此外，网络命名空间、veth 虚拟网卡以及 TLS 代理的大量运用，不仅增加了网络延迟，也对宿主机的权限配置（如 Docker 的能力限制）提出了更高要求。

---

## 3. 赋能轻量级沙箱的新内核技术：Landlock 解析

不论是 Sandlock 还是 OpenShell Lite，能够实现免特权、毫秒级启动的底层基石，都是 Linux 较新的 LSM（Linux Security Module）机制——Landlock [4]。它允许任何非特权进程安全地限制自身及其未来子进程的权限，而无需系统管理员级别的全局配置。

### 3.1 细粒度的文件系统隔离

在早期的 Linux 中，实现文件系统隔离通常需要依赖特权模式下构建的 `chroot` 或通过 User Namespace 配合 Mount Namespace 来实现，这不仅操作繁琐，还容易引入提权漏洞。Landlock 提供了一种基于规则集（Ruleset）的轻量级替代方案。

通过 `landlock_create_ruleset` 创建规则集后，开发者可以使用 `landlock_add_rule` 为特定的文件目录层次结构绑定具体的访问权限（如 `LANDLOCK_ACCESS_FS_READ_FILE`、`LANDLOCK_ACCESS_FS_MAKE_DIR` 或 `LANDLOCK_ACCESS_FS_EXECUTE` 等）。最后通过 `landlock_restrict_self` 将规则集应用于当前线程。由于这一限制对子进程是继承的，因此非常适合在拉起 Agent 进程前，在预执行（pre-exec）阶段精准划定其能够触碰的文件边界。

### 3.2 网络规则的逐步演进

自 ABI v4 起，Landlock 引入了对网络访问的限制能力。不同于基于文件路径的隔离，网络规则主要作用于 TCP 端口级别。它允许沙箱通过 `LANDLOCK_ACCESS_NET_BIND_TCP` 限制应用监听端口，以及通过 `LANDLOCK_ACCESS_NET_CONNECT_TCP` 限制应用发起出站连接。

当前阶段的 Landlock 尚不支持直接在内核层拦截 IP 地址或域名。但这种设计恰恰体现了其“机制与策略分离”的原则：Landlock 提供极其高效的、基于端口的纯内核态阻断，而将需要复杂 L7 协议解析的域名过滤任务，留给应用层代理或云原生组件来完成。

---

## 4. Sandlock：回归进程级管控的轻量化革命

面对重型沙箱的性能瓶颈与过度设计，Sandlock [1, 2] 提出了一种截然不同的解法：摒弃容器与虚拟机，利用 Linux 内核的原生安全特性，打造极致轻量的进程级沙箱。

### 4.1 极简架构与毫秒级启动

Sandlock 采用无守护进程设计，通过 Rust 编写的 CLI 或 FFI 接口直接拉起被沙箱化的应用进程。由于剥离了容器镜像构建、cgroups 资源分配和虚拟网络创建等复杂步骤，Sandlock 将启动时间压缩至约 5ms，资源开销极低。这种极速冷启动能力使其能够为每个具体的工具调用（Per-Tool）动态创建专属沙箱，极大地缩小了潜在的爆炸半径。

### 4.2 基于 Linux 内核特性的细粒度控制

通过组合 Linux 最新的安全模块，Sandlock 在不依赖 Root 权限的前提下实现了全方位的进程约束。
它利用 Landlock LSM 实现了细粒度的文件系统隔离，配合内置的 COW（写时复制）机制保护工作目录不被污染；借助 seccomp-bpf 过滤危险的系统调用（如 `ptrace`、`bpf`），切断权限提升的路径；并通过 seccomp 用户态通知机制（User Notification）实现了轻量级的资源限制与虚拟网络管控。这种纯内核态的管控手段既安全又高效。

```bash
# Sandlock 命令示例：限制网络访问并应用 COW 工作目录
# --net-allow-host: 允许访问指定域名
# -r /usr -r /lib: 挂载只读目录
# -w /tmp/workdir: 挂载 COW（写时复制）工作目录
sandlock run --net-allow-host api.openai.com -r /usr -r /lib -w /tmp/workdir -- python3 agent.py
```

### 4.3 XOA 模式：结构级别的注入防御

Sandlock 在架构上支持 Execute-Only Agent（XOA）模式，从根本上消除了 Prompt 注入的威胁。在 XOA 模式下，大语言模型仅负责生成执行代码，完全不接触不可信的外部数据。生成的代码被送入完全独立的 Sandlock 进程中执行，执行结果通过内核管道直接返回给终端用户，不再回流至模型的上下文窗口中。这种结构级别的隔离，使得即使代码处理了恶意的外部输入，也无法通过上下文污染操纵模型。

---

## 5. OpenShell Lite：云原生时代的轻量级协同沙箱

在充分吸取了 OpenShell 的重型防护经验与 Sandlock 的轻量化设计后，OpenShell Lite 提出了在云原生（Kubernetes）环境下的 Agent Sandbox 新范式。虽然该项目目前**尚未开源**，但其核心的跨平台构建体系、基于无特权 Docker 容器的边界拦截测试（如拦截越权文件读取、探测系统调用、内存超限），以及支持 Python `@sandboxed` 装饰器的极简交互 Demo 均已开发完成并验证通过。它明确了自身的功能边界，不再试图在沙箱内部解决所有的安全隔离问题，而是将防御策略与底层基础设施深度协同。

### 5.1 明确的功能边界与 K8s 的协同防御

在云原生架构中，Kubernetes（K8s）及其生态组件（如 Cilium、Istio）已经提供了非常强大的网络隔离能力（如 Network Policy 和 Egress Gateway）。OpenShell Lite 的设计初衷便是充分利用这些外部安全机制，避免“重复造轮子”。它去除了 OpenShell 中厚重的 HTTP CONNECT 代理和 OPA 引擎，专注于 K8s 无法精细化管控的领域——进程级别的系统调用过滤（Seccomp-BPF）和本地文件系统隔离（Landlock）。通过这种职责分离，OpenShell Lite 能够在保持极简架构（无守护进程、无特权模式）的同时，实现毫秒级启动，完美适配标准的 Docker/K8s 最小权限环境。

### 5.2 Landlock API 在网络限制上的取舍

OpenShell Lite 的轻量化网络管控依赖于 Linux Landlock API，但这在设计上面临着固有的局限性。目前，Landlock 的网络控制能力（自 ABI v4 起）仅支持基于 TCP 端口的 `bind` 和 `connect` 拦截，无法直接在内核态基于 IP 地址或域名（Domain）进行过滤。域名解析通常发生在用户态（如 DNS 查询），内核无法预知某个 IP 对应的具体域名。

为了实现基于域名或 IP 的精细化网络策略，传统的做法是引入 L7 透明代理（如 OpenShell）或使用 Seccomp 用户态通知（如 Sandlock）来劫持流量。然而，这些方案会显著增加系统复杂度并带来性能损耗。OpenShell Lite 选择拥抱这一局限：在沙箱层面仅限制危险的端口和网络协议，将复杂的域名和 IP 白名单拦截任务交由 K8s Network Policy 等云原生基础设施完成。这种取舍不仅保持了沙箱的极致轻量，还确保了整体安全架构的清晰与高效。

### 5.3 OpenShell Lite 与 Sandlock 的路线对比

尽管 OpenShell Lite 和 Sandlock 均基于 Landlock 和 Seccomp 实现了免特权的轻量级进程沙箱，但两者在架构边界与生态协同上走向了不同的分支：Sandlock 倾向于“大包大揽”，通过用户态流量劫持在单机内构建包含 L7 解析的闭环防护；而 OpenShell Lite 则明确“协同防御”定位，剥离应用层代理，仅保留内核级管控，将复杂的网络策略（如 IP/域名白名单）卸载给 K8s Network Policy 等云原生基础设施，以此换取极简架构与零性能损耗。

| 特性对比           | Sandlock                                    | OpenShell Lite                               |
| :----------------- | :------------------------------------------ | :------------------------------------------- |
| **设计定位**       | 大包大揽的独立沙箱（All-in-one）            | 边界清晰的协同沙箱（Cloud-Native）           |
| **网络隔离方案**   | Seccomp 用户态通知（拦截并解析 IP/HTTP）    | Landlock 端口过滤 + K8s Network Policy       |
| **功能复杂度**     | 极高（内置 HTTP ACL、TLS MITM、端口虚拟化） | 极简（仅做进程约束，剥离代理引擎）           |
| **生态依赖**       | 无依赖，可独立运行在任意 Linux 宿主机       | 强依赖云原生网络设施（如 K8s）以实现完整防护 |
| **运维与性能开销** | 中等（用户态流量劫持存在上下文切换损耗）    | 极低（纯内核态管控，无用户态代理瓶颈）       |

通过对比可以看出，Sandlock 更像是一个“单机版的微型 OpenShell”，它通过内核特性的奇技淫巧，在不依赖外部网络设施的情况下实现了媲美容器的隔离效果。而 OpenShell Lite 则做减法，它承认进程沙箱在网络 L7 解析上的短板，将专业的事交给专业的组件（K8s）去做，从而换取了更纯粹的架构和极致的性能。

---

## 6. 策略优先：下一代 Agent Sandbox 的设计范式

无论是 OpenShell 在应用层的 OPA 拦截，还是 Sandlock 与 OpenShell Lite 在内核层的管控，都在向同一个设计哲学收敛：Policy over Isolation（策略优先）。

### 6.1 从资源隔离到权限管控

传统的隔离机制（Isolation）侧重于为程序分配一块独立、密封的计算资源；而策略管控（Policy）则专注于定义程序能够对外部世界执行哪些具体操作。Agent Sandbox 的设计焦点正从“如何把 Agent 关在笼子里”转向“如何精确授予 Agent 完成任务所需的最小权限”。通过精准的白名单策略，即使 Agent 发生幻觉或被注入，其破坏力也会被严格限制在预设的策略边界内。

### 6.2 细粒度策略的实践价值

细粒度的安全策略为 AI 应用的生产落地提供了可操作的保障。在文件系统层面，可以将配置目录设为只读，仅开放特定的工作目录用于读写；在网络层面，可以配合底层网络插件拦截所有内网 IP 的访问（防止 SSRF 攻击），并对允许访问的外部域名实施访问控制。这种策略不仅防御了外部攻击，同样有效防止了 Agent 因模型理解偏差而导致的误操作。

---

## 7. 结语

Agent Sandbox 的演进历程，是行业对 AI Agent 安全本质逐渐加深理解的缩影。OpenShell 以完善的容器化与代理架构，展现了如何在一个完整的网络闭环中实现极致的安全管控；Sandlock 证明了纯内核特性在轻量化与极速启动上的巨大潜力；而 OpenShell Lite 则指明了与云原生基础设施协同防御的务实道路。“策略优先”的设计理念已经成为共识，未来的 Agent 基础设施必将在极致性能与细粒度安全之间找到完美的平衡，为大模型应用的繁荣提供坚实的基石。

---

## 参考文献

- [1] multikernel. "sandlock: The lightest AI sandbox." GitHub. https://github.com/multikernel/sandlock.
- [2] 王聪. "Sandlock：最轻量级的 AI Agent 沙箱." Linux内核之旅. https://mp.weixin.qq.com/s/ICxIj6ydpftSLwgjA9zZAg
- [3] NVIDIA. "OpenShell: Safe, private runtime for autonomous AI agents." GitHub. https://github.com/NVIDIA/OpenShell.
- [4] man7.org. "landlock(7) - Linux manual page." https://man7.org/linux/man-pages/man7/landlock.7.html.
