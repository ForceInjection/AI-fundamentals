# 达芬奇架构与 Ascend 910B3

## 1. 概述

Ascend 910B3 是华为昇腾系列 AI 处理器，采用 **达芬奇 (Da Vinci)** 架构，7 nm 制程。测试服务器配备 8 张 910B3 卡，通过 HCCS (Huawei Cache Coherence System) 全互联。

关键硬件参数（实测数据）：

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

> [!TIP]
> 通过 `ascend-dmi --info --detail` 可获得更完整的硬件规格（DIE ID、AI CPU 数量、ECC 状态、PCIe LnkCap 等），详见 `05_tools/02_ascend_dmi_reference.md`。

## 2. 达芬奇 (Da Vinci) 架构核心设计

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

## 3. 内存层次

```text
HBM2e (64 GB)
  └── L2 Cache (共享, 多 AI Core 共用)
        └── L1 Buffer (AI Core 内)
              └── L0 Buffer (CU 专用, 每种计算单元独享)
```

- **HBM**：64 GB HBM2e，带宽实测值约 1.54 TB/s（`ascend-dmi --bw -t d2d` 实测 1538 GB/s）。
- **L2 Cache**：芯片级共享缓存，容量 MB 级别。
- **L1 Buffer**：AI Core 内部缓存。
- **L0 Buffer**：最接近计算单元，Cube/Vector/Scalar 各自独享。

对比 A100 的 HBM2e（40 GB 型号 1.55 TB/s，80 GB 型号 2.04 TB/s），910B3 的 64 GB HBM2e 容量介于两者之间，适合大批量数据和模型并行场景。

## 4. 编程范式差异：SIMT vs 达芬奇

SIMT 与达芬奇架构在 "计算如何被拆分" 上走了两条不同路线，这直接决定了 CUDA 代码能否直接迁移到 NPU。

### 4.1 NVIDIA SIMT

GPU 通过大量 CUDA Core 执行相同指令、不同数据（SIMT）。一个 Warp（32 线程）共享控制逻辑，所有线程执行相同操作。矩阵运算需要通过 Tensor Core 专门加速——Tensor Core 是 SM (Streaming Multiprocessor) 内的独立子单元。

### 4.2 达芬奇架构

达芬奇将计算任务**按类型拆分**到不同的硬件单元：

1. 数据到 NPU 后，图编译器 (GE) 分析算子类型。
2. **Cube Unit** 执行矩阵乘加 → 这是 AI 计算的核心（占据 ~90% 的计算量）。
3. **Vector Unit** 并行执行激活函数、归一化等向量操作。
4. **Scalar Unit** 处理标量运算和控制流。

这种设计意味着：

- **优势**：矩阵运算的硬件利用率和能效比极高（同等功耗下矩阵乘吞吐更高）。
- **代价**：需要图编译器将算子合理拆分到不同单元，CUDA 代码不能直接 1:1 翻译执行。
- **迁移影响**：自定义 CUDA kernel 需要改写为 Ascend C/TBE 算子；标准 PyTorch 算子（已有 NPU 适配）可直接迁移。

## 5. 参考链接

- [昇腾社区 — 硬件产品](https://www.hiascend.com/hardware/ai-chip)
- [昇腾社区 — CANN 概述](https://www.hiascend.com/document)
