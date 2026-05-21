# CANN 软件栈详解

## 1. 栈全貌

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

> [!NOTE]
> CANN 的商业版本号和内部版本号需要区分。测试服务器上的 CANN 商业版本为 8.0.1（对应 `ascend-toolkit` 目录名），但 runtime/compiler/hccl 等内部组件版本号均为 7.6.0.2.220（即 `7.6` 系列）。在查阅版本兼容性矩阵时需注意这个区分。

## 2. 各层详解

### 2.1 驱动层 (Driver)

最底层，负责：

- 与固件 (Firmware v7.5.0.5.220) 通信。
- PCIe 设备管理（`/dev/davinci0` ~ `/dev/davinci7`）。
- 内存映射、DMA 传输。
- 通过 `npu-smi` 工具导出设备状态。

当前服务器驱动版本：24.1.0.3。

### 2.2 运行时层 (Runtime)

对应 CUDA Runtime API 的角色。负责任务分发到 AI Core/AI CPU、设备内存管理、Stream 与 Event 管理。Runtime 下层是任务调度层（Task Scheduler, TSC），负责将计算任务派发到具体 AI Core，但对用户完全透明——框架和 AscendCL 封装了这一层。

### 2.3 算子层 (Operator Layer)

CANN 提供三层算子开发体系：

| 算子开发方式                  | 抽象层级       | 适用场景                                     |
| ----------------------------- | -------------- | -------------------------------------------- |
| **内置算子 (OPP)**            | 最高（直接用） | 标准算子：Conv、BN、Activation 等 1400+ 算子 |
| **Ascend C**                  | 中（C++ 类库） | 自定义算子，用标准 C++ + Ascend C API 编写   |
| **TBE (Tensor Boost Engine)** | 低（DSL）      | 手动优化算子，使用领域特定语言               |

OPP (Operator Plugin Package) 已预置在 `/usr/local/Ascend/ascend-toolkit/latest/opp/`，包含各芯片架构（Ascend910、Ascend310P 等）的优化实现。

与 CUDA 的对比：

- CUDA 只有一套算子开发模型（CUDA C++ Kernel），附带 cuDNN/cuBLAS 等优化库。
- CANN 将算子开发分层：优先用内置算子 → Ascend C 自定义 → TBE 深度优化。门槛更低，层次清晰。

### 2.4 图编译与优化层

- **GE (Graph Engine)**：将框架（PyTorch/MindSpore）的计算图转换为 Ascend 可执行图，执行图融合（算子合并）、内存优化、数据布局转换等编译优化。这是 CANN 区别于 CUDA 的关键组件——CUDA 将图优化留给框架（TorchDynamo/XLA），而 CANN 在驱动层面提供了统一的图编译器。
- **AOE (Ascend Optimization Engine)**：自动调优工具，选择最优算子实现和 tiling 策略。

### 2.5 AscendCL (Ascend Computing Language)

应用开发 API 层，提供 C/Python 接口。功能包括设备/Context/Stream 管理、模型加载与推理执行、媒体数据预处理（DVPP 模块）、单算子调用。对于 PyTorch 用户，torch_npu 底层调用 AscendCL。

### 2.6 HCCL (Huawei Collective Communication Library)

集合通信库，对应 NVIDIA 的 **NCCL**。提供 AllReduce、AllGather、ReduceScatter、Broadcast 等集合操作，基于 HCCS 链路的卡间通信。测试服务器 HCCL 版本：7.6.0.2.220，8 卡之间通过 HCCS 链路全互联。

### 2.7 框架适配层

- **torch_npu**：PyTorch 与 CANN 的适配层。将 PyTorch 的 CUDA 后端调用转化为 AscendCL 调用。版本需与 PyTorch 和 CANN 同时兼容。
- **MindSpore**：华为自研框架，对 Ascend 有原生支持，不依赖 torch_npu。

## 3. CANN vs CUDA 开发工具对照

| 功能         | CANN 工具                                  | CUDA 工具                                         |
| ------------ | ------------------------------------------ | ------------------------------------------------- |
| 设备管理     | `npu-smi`                                  | `nvidia-smi`                                      |
| Profiling    | `msprof`                                   | `nsys` / `ncu` (Nsight)                           |
| 模型转换     | `atc` (Ascend Tensor Compiler)             | TensorRT `trtexec`                                |
| 模型压缩     | AMCT                                       | TensorRT 量化                                     |
| 算子自动调优 | AOE                                        | cuDNN autotune                                    |
| 算子开发     | Ascend C / TBE + `msopgen`                 | CUDA C++                                          |
| 编译工具链   | BiSheng Compiler (`ccec`)                  | NVCC                                              |
| 调试         | `msdebug`                                  | `cuda-gdb`                                        |
| 分布式通信   | HCCL                                       | NCCL                                              |
| 图编译       | GE (Graph Engine) — 含融合/内存/布局优化   | CUDA Graphs (仅 kernel 录制回放，GE 功能范围更广) |
| 集合通信测试 | `hccl_test`                                | `nccl-tests`                                      |
| 开发套件路径 | `/usr/local/Ascend/ascend-toolkit/latest/` | `/usr/local/cuda/`                                |

**关键环境变量对照**：

| CANN                        | CUDA                   | 作用          |
| --------------------------- | ---------------------- | ------------- |
| `ASCEND_RT_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES` | 可见设备控制  |
| `ASCEND_TOOLKIT_HOME`       | `CUDA_HOME`            | 工具套件路径  |
| `ASCEND_OPP_PATH`           | (无对应，内置)         | 算子包路径    |
| `ASCEND_HOME_PATH`          | (无对应)               | CANN 安装路径 |

## 4. CANN 目录结构

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

## 5. 参考链接

- [CANN 商业版文档 — 软件架构](https://www.hiascend.com/document/detail/en/canncommercial/800/overview/overview/overview_0001.html)
- [HCCL 集合通信库文档](https://www.hiascend.com/document/detail/en/canncommercial/800/devguide/hccl/hccl_0001.html)
- [昇腾社区官网](https://www.hiascend.com)
