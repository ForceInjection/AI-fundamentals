# DOCA 编程入门

本文档旨在为开发者提供一份全面的 NVIDIA DOCA 编程入门指南。内容涵盖了 DOCA 软件框架的核心组件、安装配置流程，并结合 LMCache 项目的实际需求，详细解析了零拷贝传输、控制平面卸载、计算卸载等典型应用场景的编程实践，帮助开发者快速掌握利用 BlueField DPU 加速数据中心工作负载的关键技术。

## 1. BlueField 与 DOCA 简介

### 1.1 BlueField DPU 概述

NVIDIA BlueField 网络平台（DPU 和 SuperNIC）是适用于现代数据中心基础设施的先进计算平台。它将基础设施服务域与工作负载域隔离开来，从而显著提高应用程序和服务器的性能、安全性和效率。

- **核心价值**：
  - **卸载 (Offload)**：将网络、存储和安全任务从 CPU 转移到 DPU，释放 CPU 算力。
  - **加速 (Accelerate)**：利用 DPU 内置的硬件加速引擎（如数据压缩、加密、正则匹配）处理特定工作负载。
  - **隔离 (Isolate)**：实现基础设施服务与业务应用的物理隔离，提供零信任安全环境。

### 1.2 DOCA 软件框架

NVIDIA DOCA (Data Center on a Chip Architecture) 是解锁 BlueField DPU 潜力的关键软件框架。它为开发者提供了一套行业标准的开放 API、驱动程序、库、工具和示例应用程序。

- **DOCA SDK**：包含用于开发加速应用程序的库和驱动程序。支持 RDMA 加速、网络流处理 (Flow)、存储虚拟化 (Emulation)、正则表达式匹配 (RegEx) 等功能。
- **DOCA Runtime**：包含用于在数据中心大规模部署、配置和编排容器化服务的工具。
- **DOCA-Host**：安装在主机服务器上的软件包，提供与 BlueField DPU 通信所需的驱动和接口。

通过 DOCA，开发者可以构建软件定义、云原生的加速服务，满足高性能计算和 AI 云的需求。

---

## 2. DOCA 软件组件

![DOCA Software Stack](doca-software.jpg)

DOCA 软件栈由一系列分层的库和驱动程序组成，旨在为 DPU 开发提供统一且高效的接口。其核心组件可以分为以下三类：

### 2.1 基础库 (Base Libraries)

基础库为所有 DOCA 应用程序提供了底层的支撑功能，确保开发的一致性和便捷性。

- **`doca_common`**: DOCA 生态系统的基石。
  - **日志记录 (`DOCA_LOG`)**: 提供分级日志系统，支持将日志输出到控制台或文件，方便调试和运维。
  - **错误处理 (`doca_error_t`)**: 定义了统一的错误码规范，简化了跨模块的错误传播和处理。
  - **基础数据结构**: 提供了链表、哈希表等经过优化的高性能数据结构。
- **`doca_argp`**: 命令行参数解析库。
  - 允许开发者轻松定义和解析应用程序的启动参数（如 PCI 地址、配置文件路径等），减少样板代码。
- **`doca_buf`**: 高性能内存缓冲区管理。
  - 支持复杂的内存布局（如 Scatter-Gather Lists, SGL），是数据平面处理的核心数据结构。
  - 提供引用计数机制，确保零拷贝场景下的内存安全。

### 2.2 核心运行时库 (Core Runtime Libraries)

运行时库是 DOCA 的核心，封装了 BlueField DPU 的硬件加速能力，涵盖网络、存储、计算和控制平面。

- **数据传输与存储**
  - **DOCA DMA (`doca_dma`)**:
    - 提供主机 (Host) 与 DPU 之间、以及 DPU 内部内存的高效直接内存访问。
    - 支持异步操作模式，允许 CPU 在数据搬运期间处理其他任务，显著降低 CPU 占用。
  - **DOCA Compress (`doca_compress`)**:
    - 利用 DPU 内置的硬件压缩引擎（支持 Deflate, LZ4 等算法）。
    - 相比 CPU 软压缩，提供数倍的吞吐量提升，同时释放 CPU 算力。
- **网络与流处理**
  - **DOCA Flow (`doca_flow`)**:
    - 提供基于硬件流表的数据包处理管道。
    - 支持复杂的动作链（Match-Action），如修改包头、封装/解封装 (Tunneling)、转发、丢弃等。
    - 是构建软件定义网络 (SDN) 和防火墙应用的基础。

- **控制平面与通信**
  - **DOCA Comch (`doca_comch`)**:
    - 建立主机与 DPU 之间可靠的控制平面通信通道。
    - 屏蔽了底层的 PCIe/Socket 细节，提供类似于 Socket 的消息收发接口。
    - 适用于配置下发、状态同步和心跳检测等场景。

- **异构计算**
  - **DOCA DPA (`doca_dpa`)**:
    - 数据路径加速器 (Data Path Accelerator) 接口。
    - 允许开发者编写自定义的内核代码 (Kernel)，运行在 DPU 的轻量级 RISC-V 核心上。
    - 适用于极低延迟的数据包处理或计算密集型的近数据处理任务。

### 2.3 服务与应用 (Services & Applications)

DOCA 还提供了一系列预构建的服务和参考应用，帮助开发者快速落地。

- **DOCA App Shield (`doca_apsh`)**: 利用 DPU 对主机内存进行带外 (Out-of-Band) 监控，实现入侵检测和恶意软件分析。
- **DOCA Firefly**: 提供高精度的时间同步服务 (PTP)，适用于金融交易和电信网络。
- **DOCA Telemetry**: 收集 DPU 的运行指标和性能数据，集成到 Prometheus/Grafana 等监控系统。

---

## 3. 示例代码构建指南

本章节主要介绍如何获取和编译 `doca-samples` 示例代码。在此之前，请确保您已经拥有 NVIDIA BlueField DPU 硬件，并且已经按照官方文档安装了 DOCA SDK（这是编译和运行示例的前提条件）。

> **安装资源**：
>
> - [NVIDIA DOCA Installation Guide for Linux](https://docs.nvidia.com/doca/sdk/doca-installation-guide-for-linux/index.html): 详细的安装指南，包含 Host 和 DPU 端的配置。
> - [NVIDIA DOCA Developer Guide](https://docs.nvidia.com/doca/sdk/doca-developer-quick-start-guide/index.html): 开发环境搭建与 SDK 使用说明。

### 3.1 获取示例代码

DOCA 提供了丰富的示例代码，托管在 GitHub 上。首先，克隆 `doca-samples` 仓库：

```bash
git clone https://github.com/NVIDIA-DOCA/doca-samples.git
cd doca-samples
```

### 3.2 编译环境准备

DOCA 使用 `meson` 和 `ninja` 作为构建系统。请确保您的开发环境中已安装这些工具。

### 3.3 编译应用程序

参考应用程序位于 `applications` 目录下。编译步骤如下：

1. 进入应用程序目录：

   ```bash
   cd applications
   ```

2. 配置构建目录（默认使用 Debug 模式）：

   ```bash
   meson /tmp/build
   ```

   _注意：默认编译为 Debug 模式，包含调试符号且未优化。如果需要 Release 模式，请查阅 Meson 文档配置优化选项。_

3. 开始编译：

   ```bash
   ninja -C /tmp/build
   ```

4. 编译完成后，二进制文件将位于 `/tmp/build/<application_name>/` 目录下。

### 3.4 开发者配置

如果需要开启 DOCA 的追踪日志 (Trace Log) 以进行深度调试，可以在配置 Meson 时添加参数：

```bash
meson /tmp/build -Denable_trace_log=true
```

## 4. 开发环境与调试指南

### 4.1 编程语言支持

DOCA 提供了多语言的开发支持，以适应不同的应用场景。

#### 4.1.1 C 语言 (Core)

DOCA SDK 的核心库（如 `doca_dma`, `doca_flow`, `doca_comch`）均提供标准的 C API。这是开发高性能数据平面应用（如 LMCache 存储引擎）的首选语言，能提供对硬件最细粒度的控制和最低的延迟。

#### 4.1.2 Python (Management & Scripts)

虽然核心数据路径通常使用 C 开发，但 DOCA 提供了部分 Python 绑定和工具，适用于管理平面、自动化脚本和快速原型验证。

- **PyDOCA**: 部分 DOCA 库提供 Python 绑定。
- **辅助工具**: 如 `doca_devemu` 示例中包含 `rpc_nvmf_doca.py` 脚本，用于通过 RPC 配置 SPDK 后端。

#### 4.1.3 DOCA DPL (P4)

**DPL (DOCA Pipeline Language)** 是一种基于 P4-16 的领域特定语言，专用于编写 BlueField DPU 的数据平面流水线。

- **适用场景**: 软件定义网络 (SDN)、自定义包处理逻辑。
- **编译器 (`dplp4c`)**: 将 P4 代码编译为 DPU 硬件可执行的配置。

  ```bash
  dplp4c.sh --target doca my_program.p4
  ```

- **开发流程**: 编写 P4 代码 -> 使用 `dplp4c` 编译 -> 通过 DPL Runtime Service 加载到 DPU。

### 4.2 性能评测工具 (`doca_bench`)

`doca_bench` 是一个通用的性能基准测试工具，用于评估 DOCA API 在不同配置下的吞吐量和延迟。

- **基本用法**:

  ```bash
  doca_bench --csv-output-file /tmp/results.csv --latency-bucket-range 0,100 ...
  ```

- **关键参数**:
  - `--csv-output-file <path>`: 将测试结果输出为 CSV 文件，便于后续分析。
  - `--csv-append-mode`: 追加模式，适合批量运行测试并汇总结果。
  - `--latency-bucket-range <start,width>`: 自定义延迟直方图的桶范围，用于精细化分析长尾延迟。

### 4.3 Flow 调优工具 (`doca_flow_tune`)

`doca_flow_tune` 用于分析和优化 `doca_flow` 应用程序的规则下发和匹配性能。它包含 DPU 端的 Server 和 Host 端的 CLI 工具。

- **主要模式**:
  1. **Monitor (`monitor`)**: 实时监控软件 KPI 和硬件计数器。

     ```bash
     doca_flow_tune monitor background
     ```

  2. **Analyze (`analyze`)**: 分析 Flow 管道结构，检测潜在瓶颈。

     ```bash
     doca_flow_tune analyze export --file-name pipeline_dump.json
     ```

  3. **Visualize (`visualize`)**: 将导出的 JSON 描述转换为 Mermaid 图表，直观展示流表层级。

     ```bash
     doca_flow_tune visualize --pipeline-desc pipeline_dump.json --file-name flow_graph.mmd
     ```

     生成的 `.mmd` 文件可以使用 Mermaid Live Editor 渲染为流程图。

### 4.4 DPA 调试 (`doca-dpa-gdb`)

针对运行在 RISC-V 核心上的 DPA 程序，DOCA 提供了专用的 GDB Server 工具。

- **功能**: 支持断点、单步执行、查看寄存器和内存。
- **限制**:
  - 不支持捕获 Fatal Errors。
  - 不支持访问 Window 内存区域。
  - 需配合 Host 端的 GDB 客户端使用。
- **使用方式**:
  在 DPU 上启动 Server，然后使用 GDB 远程连接进行调试。

### 4.5 错误处理与日志

- **错误处理**: 所有 DOCA API 返回 `doca_error_t`。务必检查返回值：

  ```c
  if (result != DOCA_SUCCESS) {
      DOCA_LOG_ERR("Error: %s", doca_error_get_descr(result));
  }
  ```

- **日志配置**:
  - `doca_log_backend_create_standard()`: 创建标准输出日志后端。
  - `doca_log_backend_set_sdk_level()`: 设置 SDK 内部日志级别（推荐在调试时设为 `DOCA_LOG_LEVEL_DEBUG`）。

---

## 5. 典型场景编程入门

本章介绍如何利用 DOCA 组件解决实际架构中的性能瓶颈，涵盖零拷贝传输、控制卸载、计算卸载等关键场景。

### 4.1 零拷贝数据传输 (`doca_dma`)

**应用场景**：
对应 LMCache 的 **Host-DPU 内存协同 (Memory Mapping)**。在集群共享或 PD 分离场景中，解决 "Memory Wall" 问题。通过 `doca_dma`，DPU 可以直接读取 Host 的 Pinned Memory 或 GPU Memory (需结合 GPUDirect)，消除 Host CPU 在数据搬运中的开销。

**技术原理**：
DOCA DMA 利用 BlueField DPU 内置的 DMA 引擎，在不消耗 Host CPU 周期的情况下执行内存数据搬运。

- **Memory Registration**: 通过 `doca_mmap` 将虚拟地址映射到物理地址并锁定页面（Pinning），确保 DMA 引擎可以直接寻址。
- **PCIe TLP**: 在 Host 与 DPU 之间，数据以 TLP (Transaction Layer Packet) 的形式通过 PCIe 总线直接传输。
- **异步执行**: 任务提交后立即返回，CPU 可继续处理其他逻辑，待硬件搬运完成后通过回调或轮询获取通知。

**核心步骤**：

1. **Host 侧**：使用 `doca_mmap_export_pci` 导出内存区域。
2. **DPU 侧**：接收导出描述符，调用 `doca_mmap_create_from_export` 建立映射。
3. **数据传输**：DPU 提交 DMA Copy 任务，数据直接在 Host 内存与 DPU 内存/网络之间流转。

**代码示例** (基于 `samples/doca_dma/dma_local_copy/dma_local_copy_sample.c` 修改)：

```c
/* 注册内存范围并启动 mmap (DPU 侧操作) */
static doca_error_t register_memory_range_and_start_mmap(struct doca_mmap *mmap, char *buffer, size_t length)
{
    doca_error_t result;

    /* 设置内存范围：若是 Host 内存，此处需使用从 Export Descriptor 创建的 mmap */
    result = doca_mmap_set_memrange(mmap, buffer, length);
    if (result != DOCA_SUCCESS)
        return result;

    /* 启动 mmap，使内存可被 DMA 引擎访问 */
    return doca_mmap_start(mmap);
}
```

### 4.2 控制平面卸载 (`doca_comch`)

**应用场景**：
对应 LMCache 的 **Control Plane Offload**。将 `LMCacheWorker` 的心跳维护、元数据同步、连接管理等逻辑从 Host 迁移至 DPU。Host 仅需通过轻量级通道发送 `AllocRequest` 或接收 `Doorbell` 通知，从而消除 OS 噪声对 vLLM 推理线程的干扰。

**技术原理**：
DOCA Comch 基于 Client-Server 模型，在 Host 与 DPU 之间建立可靠的双向通信通道。

- **传输抽象**: 底层复用 shared memory (Host-DPU) 或 RDMA/TCP (DPU-DPU)，向上层提供统一的消息收发接口。
- **控制面隔离**: 通过将复杂的控制逻辑（如心跳、状态机）下沉到 DPU，Host 侧仅保留极简的指令发送接口，实现业务逻辑与控制逻辑的物理隔离。

**核心步骤**：

1. **建立连接**：Host (Client) 与 DPU (Server) 握手。
2. **指令下发**：Host 发送非阻塞控制消息。
3. **事件通知**：DPU 处理完成后，通过 Comch 反馈结果。

**代码示例** (摘自 `samples/doca_comch/comch_ctrl_path_client/comch_ctrl_path_client_sample.c`)：

```c
/* Host 侧：发送控制指令的回调处理 */
static void send_task_completion_callback(struct doca_comch_task_send *task,
                                          union doca_data task_user_data,
                                          union doca_data ctx_user_data)
{
    /* 确认指令已发送至 DPU */
    struct comch_ctrl_path_objects *sample_objects = (struct comch_ctrl_path_objects *)ctx_user_data.ptr;
    sample_objects->result = DOCA_SUCCESS;
    DOCA_LOG_INFO("Task sent successfully");

    /* 清理任务资源 */
    doca_task_free(doca_comch_task_send_as_task(task));
}
```

### 4.3 计算卸载与压缩 (`doca_compress`)

**应用场景**：
对应 LMCache 的 **Compute Offload**。将 KV Cache 的压缩 (`CacheGen`) 和解压任务卸载至 DPU 专用硬件加速器。这避免了在 Host CPU 上进行密集计算，也避免了占用宝贵的 GPU 显存。

**技术原理**：
DOCA Compress 调用 DPU 内部专用的硬件加速引擎执行数据压缩/解压。

- **硬件卸载**: 相比 CPU 软实现，硬件引擎具有更高的吞吐量和更低的能耗。
- **零拷贝流水线**: 结合 `doca_dma`，数据可以从 Host 内存直接读入 DPU，压缩后再写入网络或存储，全程无需 Host CPU 参与数据处理。

**核心步骤**：

1. **资源分配**：初始化 Compress 上下文。
2. **能力查询**：查询硬件支持的最大 Buffer 大小 (`max_buf_size`)。
3. **任务提交**：构建 Deflate/LZ4 压缩任务并提交。

**代码示例** (摘自 `samples/doca_compress/compress_deflate/compress_deflate_sample.c`)：

```c
/* 初始化压缩资源与查询能力 */
doca_error_t compress_deflate(struct compress_cfg *cfg, char *file_data, size_t file_size) {
    uint64_t max_buf_size;
    doca_error_t result;
    struct compress_resources resources = {0};
    struct program_core_objects *state;

    /* ... 资源分配逻辑 ... */
    state = resources.state;

    /* 查询硬件支持的最大解压/压缩 Buffer 大小 */
    result = doca_compress_cap_task_decompress_deflate_get_max_buf_size(
                doca_dev_as_devinfo(state->dev), &max_buf_size);

    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to query compress max buf size: %s", doca_error_get_descr(result));
        return result;
    }
    /* LMCache 需确保 KV Block 大小不超过 max_buf_size */
}
```

### 4.4 存储虚拟化与仿真 (`doca_devemu`)

**应用场景**：
对应 LMCache 的 **Local Reuse (Zero-Syscall)**。利用 DPU 模拟 NVMe 设备，Host 视其为标准物理磁盘，实则由 DPU 侧的 SPDK 接管 I/O。这使得 Host 可以通过标准 NVMe 驱动直接读写数据，绕过文件系统开销，实现极致的低延迟存储访问。

**技术原理**：
DOCA DevEmu 允许 DPU 模拟 PCIe 设备（Function），Host OS 将其识别为标准硬件。

- **Doorbell 截获**: Host 写入模拟设备的 Doorbell 寄存器时，操作被 DPU 硬件截获并触发 DPU 侧的事件通知。
- **软件定义 I/O**: DPU 侧的软件（如 SPDK）处理这些 I/O 请求，可以灵活地将数据重定向到本地 NVMe、远程存储 (NVMe-oF) 或内存缓存中，对 Host 完全透明。

**核心步骤**：

1. **设备模拟**：在 DPU 上通过 `doca_devemu` 创建虚拟 PCIe 设备。
2. **DB 处理**：DPU 监听 Doorbell (DB) 事件，截获 Host 的 I/O 请求。
3. **后端处理**：调用后端存储服务（如 SPDK 或 RDMA）处理数据。

**代码示例** (摘自 `samples/doca_devemu/devemu_pci_device_db/dpu/host/devemu_pci_device_db_dpu_sample.c`)：

```c
/* 初始化 DPA 线程以处理模拟设备的 Doorbell */
doca_error_t init_dpa_db_thread(struct devemu_resources *resources)
{
    doca_error_t ret;

    /* 创建 DPA 线程 */
    ret = doca_dpa_thread_create(resources->dpa, &resources->dpa_thread);
    if (ret != DOCA_SUCCESS) return ret;

    /* 设置线程处理函数 (db_handler)，用于响应 Host 的 I/O 请求 */
    ret = doca_dpa_thread_set_func_arg(resources->dpa_thread, db_handler, 0);

    /* 启动线程 */
    return doca_dpa_thread_start(resources->dpa_thread);
}
```

### 4.5 近数据计算 (`doca_dpa`)

**应用场景**：
对应 LMCache 的 **Near-Data Processing**。在 PD 分离或存储池场景中，利用 DPU 的 RISC-V 核心 (DPA) 执行自定义的 KV 量化、Token 过滤或 Embedding 检索算法。实现“数据不动，计算下沉”。

**技术原理**：
DOCA DPA (Data Path Accelerator) 是嵌入在 NIC 数据路径中的一组轻量级 RISC-V 核心。

- **低延迟执行**: DPA 核心紧邻网络接口和内存控制器，能够以极低的延迟响应网络事件或访问内存。
- **Kernel Offload**: Host 或 DPU Arm 核心可以触发 DPA 上预加载的 "Kernel" 函数。这些 Kernel 异步运行，释放了通用处理器的算力，特别适合高吞吐、简单的计算任务。

**核心步骤**：

1. **Kernel 定义**：使用 `__dpa_global__` 定义 DPA 核函数。
2. **Kernel Launch**：Host 通过 `doca_dpa_kernel_launch_update_set` 触发执行。
3. **同步等待**：使用 Sync Event 等待计算完成。

**代码示例** (摘自 `samples/doca_dpa/dpa_kernel_launch/host/dpa_kernel_launch_sample.c`)：

```c
/* Host 侧：发起 DPA Kernel 执行 */
doca_error_t kernel_launch(struct dpa_resources *resources) {
    /* ... 创建 Sync Events ... */

    /* 启动 DPA Kernel */
    /* hello_world 为 DPA 侧定义的核函数句柄 */
    result = doca_dpa_kernel_launch_update_set(resources->pf_dpa_ctx,
                           wait_event,  /* 等待条件 */
                           wait_thresh,
                           comp_event,  /* 完成通知 */
                           comp_event_val,
                           num_dpa_threads,
                           &hello_world); /* 自定义 Kernel */

    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to launch DPA kernel");
    }
    return result;
}
```
