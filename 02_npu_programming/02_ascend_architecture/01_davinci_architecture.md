# 02 — 昇腾硬件架构与 CANN 软件栈

## 1. 硬件架构: Ascend 910B3

### 1.1 概述

Ascend 910B3 是华为昇腾系列 AI 处理器，采用 **达芬奇 (Da Vinci)** 架构，7 nm 制程。本服务器配备 8 张 910B3 卡，通过 HCCS (Huawei Cache Coherence System) 全互联。

关键硬件参数 (实测数据)：

| 参数                | 值                                                                                        |
| ------------------- | ----------------------------------------------------------------------------------------- |
| 型号                | Ascend 910B3                                                                              |
| HBM 容量/卡         | 64 GB (65536 MB)                                                                          |
| HBM 频率            | 1600 MHz                                                                                  |
| HBM 厂商 ID         | 0x57 (Samsung)                                                                            |
| 卡间互联            | HCCS，8 卡全互联 (full mesh)                                                              |
| HCCS 单链路 lane 数 | 4                                                                                         |
| 典型功耗            | ~90-97 W（空闲）/ ~231 W（FP16 算力满载）/ ~273 W（实测训练满载）/ ~300 W（TDP 额定最大） |
| 空闲温度            | 34-40°C                                                                                   |
| 驱动版本            | 24.1.0.3                                                                                  |

服务器上实测的卡间拓扑（`npu-smi info -l`）：

```text
      NPU0  NPU1  NPU2  NPU3  NPU4  NPU5  NPU6  NPU7
NPU0   X    HCCS  HCCS  HCCS  HCCS  HCCS  HCCS  HCCS
NPU1  HCCS   X    HCCS  HCCS  HCCS  HCCS  HCCS  HCCS
NPU2  HCCS  HCCS   X    HCCS  HCCS  HCCS  HCCS  HCCS
NPU3  HCCS  HCCS  HCCS   X    HCCS  HCCS  HCCS  HCCS
NPU4  HCCS  HCCS  HCCS  HCCS   X    HCCS  HCCS  HCCS
NPU5  HCCS  HCCS  HCCS  HCCS  HCCS   X    HCCS  HCCS
NPU6  HCCS  HCCS  HCCS  HCCS  HCCS  HCCS   X    HCCS
NPU7  HCCS  HCCS  HCCS  HCCS  HCCS  HCCS  HCCS   X
```

8 卡之间每两张卡均有 HCCS 专有链路直连，不经过 PCIe 或 NUMA。这是训练集群的理想拓扑。

> 通过 `ascend-dmi --info --detail` 可获得更完整的硬件规格（DIE ID、AI CPU 数量、ECC 状态、PCIe LnkCap 等），详见 [02_ascend_dmi_reference.md](../05_tools/02_ascend_dmi_reference.md) 第 3 节。

---

### 1.2 达芬奇 (Da Vinci) 架构核心设计

昇腾 AI Core 是达芬奇架构的计算核心单元。每个 AI Core 内部包含三种计算单元，分工明确：

```text
┌─────────────────────────────────────┐
│              AI Core                │
│  ┌──────────┐ ┌────────┐ ┌───────┐  │
│  │  Cube    │ │ Vector │ │ Scalar│  │
│  │  Unit    │ │  Unit  │ │  Unit │  │
│  │(矩阵运算) │ │(向量运算)│ │(标量) │  │
│  └──────────┘ └────────┘ └───────┘  │
│  ┌──────────┐ ┌──────────────────┐  │
│  │ L1 Buffer│ │   L0 Buffer      │  │
│  └──────────┘ └──────────────────┘  │
│  ┌────────────────────────────────┐ │
│  │        Unified Buffer          │ │
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘
```

| 计算单元        | 职责                                   | CUDA 类比          |
| --------------- | -------------------------------------- | ------------------ |
| **Cube Unit**   | 矩阵乘加 (MAC) 运算，处理卷积/矩阵乘   | Tensor Core        |
| **Vector Unit** | 向量运算：激活函数、归一化、元素级操作 | CUDA Core (向量化) |
| **Scalar Unit** | 标量计算、控制流、地址计算             | CUDA Core (标量)   |

**关键设计理念**：与 NVIDIA 的 SIMT (Single Instruction Multiple Threads) 不同，达芬奇架构将计算按类型分离到不同硬件单元，每个单元针对特定计算模式优化。Cube Unit 专注矩阵运算（AI 训练/推理的核心负载），Vector/Scalar Unit 处理其余操作。这种异构设计使得矩阵密集负载下的面积/能效比更高。

### 1.3 内存层次

```text
HBM (64 GB, High Bandwidth Memory)
  └── L2 Cache (共享, 多 AI Core 共用)
        └── L1 Buffer (AI Core 内)
              └── L0 Buffer (CU 专用, 每种计算单元独享)
```

- **HBM**：64 GB HBM2e，带宽实测值约 1.54 TB/s（`ascend-dmi --bw -t d2d` 实测 1538 GB/s）
- **L2 Cache**：芯片级共享缓存，容量 MB 级别
- **L1 Buffer**：AI Core 内部缓存
- **L0 Buffer**：最接近计算单元，Cube/Vector/Scalar 各自独享

对比 A100 的 40/80 GB HBM2e 和 2 TB/s 带宽，910B3 的 HBM 容量更大（64 GB）但带宽略低，适合大批量数据和模型并行场景。

> 注：CANN 的商业版本号和内部版本号需要区分。本服务器上的 CANN 商业版本为 8.0.1（对应 `ascend-toolkit` 目录名），但 runtime/compiler/hccl 等内部组件版本号均为 7.6.0.2.220（即 `7.6` 系列）。在查阅版本兼容性矩阵时需注意这个区分。

---

## 2. CANN 软件栈

CANN 是昇腾硬件上层的完整异构计算栈，用户从上层框架到底层驱动都会与其交互。下面先从全貌看各层位置，再逐层展开职责。

### 2.1 栈全貌

CANN (Compute Architecture for Neural Networks) 是华为昇腾的异构计算架构，类似于 NVIDIA 的 CUDA 平台 + 工具链集合。

```text
┌──────────────────────────────────────────────┐
│  开发框架层                                    │
│  MindSpore / PyTorch (torch_npu) / TensorFlow│
├──────────────────────────────────────────────┤
│  应用开发层                                    │
│  AscendCL (C/Python API)                     │
├──────────────────────────────────────────────┤
│  图编译与优化层                                │
│  GE (Graph Engine) / AOE (Auto Optimizer)    │
├──────────────────────────────────────────────┤
│  算子层                                       │
│  Ascend C / TBE / AI CPU 算子 + 内置算子库     │
│  (OPP: Operator Plugin Package)              │
├──────────────────────────────────────────────┤
│  运行时层 (Runtime)                           │
│  任务调度 / 内存管理 / Stream 管理              │
├──────────────────────────────────────────────┤
│  任务调度层 (Task Scheduler / TSC)             │
├──────────────────────────────────────────────┤
│  驱动层 (Driver)                              │
│  设备管理 / 固件通信 / PCIe 驱动                │
├──────────────────────────────────────────────┤
│  硬件层                                       │
│  Ascend 910B3 / 310P / 710 等                 │
└──────────────────────────────────────────────┘
```

### 2.2 各层详解

#### 2.2.1 驱动层 (Driver)

最底层，负责：

- 与固件 (Firmware v7.5.0.5.220) 通信
- PCIe 设备管理（`/dev/davinci0` ~ `/dev/davinci7`）
- 内存映射、DMA 传输
- 通过 `npu-smi` 工具导出设备状态

当前服务器驱动版本：24.1.0.3

#### 2.2.2 运行时层 (Runtime)

对应 CUDA Runtime API 的角色。负责任务分发到 AI Core/AI CPU、设备内存管理、Stream 与 Event 管理。Runtime 下层是任务调度层（Task Scheduler, TSC），负责将计算任务派发到具体 AI Core，但对用户完全透明——框架和 AscendCL 封装了这一层。

#### 2.2.3 算子层 (Operator Layer)

CANN 提供三层算子开发体系：

| 算子开发方式                  | 抽象层级       | 适用场景                                     |
| ----------------------------- | -------------- | -------------------------------------------- |
| **内置算子 (OPP)**            | 最高（直接用） | 标准算子：Conv、BN、Activation 等 1400+ 算子 |
| **Ascend C**                  | 中（C++ 类库） | 自定义算子，用标准 C++ + Ascend C API 编写   |
| **TBE (Tensor Boost Engine)** | 低（DSL）      | 手动优化算子，使用领域特定语言               |

OPP (Operator Plugin Package) 已预置在 `/usr/local/Ascend/ascend-toolkit/latest/opp/`，包含各芯片架构（Ascend910、Ascend310P 等）的优化实现。

与 CUDA 的对比：

- CUDA 只有一套算子开发模型（CUDA C++ Kernel），附带 cuDNN/cuBLAS 等优化库
- CANN 将算子开发分层：优先用内置算子 → Ascend C 自定义 → TBE 深度优化。门槛更低，层次清晰

#### 2.2.4 图编译与优化层

- **GE (Graph Engine)**：将框架（PyTorch/MindSpore）的计算图转换为 Ascend 可执行图，执行图融合（算子合并）、内存优化、数据布局转换等编译优化。这是 CANN 区别于 CUDA 的关键组件——CUDA 将图优化留给框架（TorchDynamo/XLA），而 CANN 在驱动层面提供了统一的图编译器。
- **AOE (Ascend Optimization Engine)**：自动调优工具，选择最优算子实现和 tiling 策略

#### 2.2.5 AscendCL (Ascend Computing Language)

应用开发 API 层，提供 C/Python 接口。功能包括：

- 设备/Context/Stream 管理
- 模型加载与推理执行
- 媒体数据预处理（图像编解码、缩放等，即 DVPP 模块）
- 单算子调用

对于 PyTorch 用户，torch_npu 底层调用 AscendCL；对于 C/C++ 开发者，直接使用 AscendCL API。

#### 2.2.6 HCCL (Huawei Collective Communication Library)

集合通信库，对应 NVIDIA 的 **NCCL**。提供：

- AllReduce、AllGather、ReduceScatter、Broadcast 等集合操作
- 基于 HCCS 链路的卡间通信
- 通信原语和算法实现

本服务器 HCCL 版本：7.6.0.2.220，8 卡之间通过 HCCS 链路全互联。

#### 2.2.7 框架适配层

- **torch_npu**：PyTorch 与 CANN 的适配层。将 PyTorch 的 CUDA 后端调用转化为 AscendCL 调用。版本需与 PyTorch 和 CANN 同时兼容。
- **MindSpore**：华为自研框架，对 Ascend 有原生支持，不依赖 torch_npu。

---

## 3. CANN vs CUDA 开发工具对照

| 功能         | CANN 工具                                  | CUDA 工具               |
| ------------ | ------------------------------------------ | ----------------------- |
| 设备管理     | `npu-smi`                                  | `nvidia-smi`            |
| Profiling    | `msprof`                                   | `nsys` / `ncu` (Nsight) |
| 模型转换     | `atc` (Ascend Tensor Compiler)             | TensorRT `trtexec`      |
| 模型压缩     | AMCT                                       | TensorRT 量化           |
| 算子自动调优 | AOE                                        | cuDNN autotune          |
| 算子开发     | Ascend C / TBE + `msopgen`                 | CUDA C++                |
| 编译工具链   | BiSheng Compiler (`ccec`)                  | NVCC                    |
| 调试         | `msdebug`                                  | `cuda-gdb`              |
| 分布式通信   | HCCL                                       | NCCL                    |
| 图编译       | GE (Graph Engine)                          | CUDA Graphs             |
| 集合通信测试 | `hccl_test`                                | `nccl-tests`            |
| 开发套件路径 | `/usr/local/Ascend/ascend-toolkit/latest/` | `/usr/local/cuda/`      |

**关键环境变量对照**：

| CANN                        | CUDA                   | 作用          |
| --------------------------- | ---------------------- | ------------- |
| `ASCEND_RT_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES` | 可见设备控制  |
| `ASCEND_TOOLKIT_HOME`       | `CUDA_HOME`            | 工具套件路径  |
| `ASCEND_OPP_PATH`           | (无对应，内置)         | 算子包路径    |
| `ASCEND_HOME_PATH`          | (无对应)               | CANN 安装路径 |

---

## 4. 从服务器视角理解 CANN 目录结构

```text
/usr/local/Ascend/
├── driver/           # 驱动 (v24.1.0.3)
│   ├── device/       # 设备节点
│   ├── lib64/        # 驱动库
│   └── version.info
├── firmware/         # 固件 (v7.5.0.5.220)
├── ascend-toolkit/   # CANN 开发套件
│   ├── 8.0.1/        # 版本号
│   ├── 8.0/          # 主版本
│   ├── latest/  → 8.0.1  # 符号链接
│   └── set_env.sh    # 环境变量脚本
├── toolbox/          # 工具箱 (npu-smi 等)
└── Ascend-Docker-Runtime/
```

`set_env.sh` 的核心内容：

```bash
ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
LD_LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/lib64:...
PYTHONPATH=$ASCEND_TOOLKIT_HOME/python/site-packages:...
PATH=$ASCEND_TOOLKIT_HOME/bin:$ASCEND_TOOLKIT_HOME/compiler/ccec_compiler/bin:...
ASCEND_OPP_PATH=$ASCEND_TOOLKIT_HOME/opp
ASCEND_AICPU_PATH=$ASCEND_TOOLKIT_HOME
```

---

## 5. 编程范式差异: SIMT vs 达芬奇

SIMT 与达芬奇架构在“计算如何被拆分”上走了两条不同路线，这直接决定了 CUDA 代码能否直接迁移到 NPU。

### 5.1 NVIDIA SIMT

GPU 通过大量 CUDA Core 执行相同指令、不同数据（SIMT）。一个 Warp (32 线程) 共享控制逻辑，所有线程执行相同操作。矩阵运算需要通过 Tensor Core 专门加速——Tensor Core 是 SM (Streaming Multiprocessor) 内的独立子单元。

### 5.2 达芬奇架构

达芬奇将计算任务**按类型拆分**到不同的硬件单元：

1. 数据到 NPU 后，图编译器 (GE) 分析算子类型
2. **Cube Unit** 执行矩阵乘加 → 这是 AI 计算的核心（占据 ~90% 的计算量）
3. **Vector Unit** 并行执行激活函数、归一化等向量操作
4. **Scalar Unit** 处理标量运算和控制流

这种设计意味着：

- **优势**：矩阵运算的硬件利用率和能效比极高（同等功耗下矩阵乘吞吐更高）
- **代价**：需要图编译器将算子合理拆分到不同单元，CUDA 代码不能直接 1:1 翻译执行
- **迁移影响**：自定义 CUDA kernel 需要改写为 Ascend C/TBE 算子；标准 PyTorch 算子（已有 NPU 适配）可直接迁移

---

## 6. 参考链接

- [昇腾社区 — 硬件产品](https://www.hiascend.com/hardware/ai-chip)
- [CANN 商业版文档 — 软件架构](https://www.hiascend.com/document/detail/en/canncommercial/800/overview/overview/overview_0001.html)
- [昇腾社区 — CANN 概述](https://www.hiascend.com/document)
- [Ascend PyTorch 适配 (Gitee)](https://gitee.com/ascend/pytorch)
- [HCCL 集合通信库文档](https://www.hiascend.com/document/detail/en/canncommercial/800/devguide/hccl/hccl_0001.html)
