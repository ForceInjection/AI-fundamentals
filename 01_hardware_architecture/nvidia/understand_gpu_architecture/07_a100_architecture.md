# GPU 架构实例：NVIDIA A100 (Ampere)

> 基于 A100-SXM4-80GB (CC 8.0) 服务器实测。A100 是 NVIDIA 数据中心 GPU 从 Volta 到 Hopper 的关键转折点——它引入了 TF32 精度、结构稀疏性（Sparsity）、MIG 分区等至今仍是 AI 训练推理基座的能力。本文从架构规格到实测验证完整覆盖。

---

## 1. Ampere GA100 概览

A100 基于 Ampere 架构的 **GA100** 芯片，SM 数量从 V100 的 80 个提升到 108 个。与 V100 的对比：

| 规格 | V100 (Volta) | A100 (Ampere) | 变化 |
|------|-------------|--------------|------|
| SM 数量 | 80 | **108** | +35% |
| CUDA Cores/SM | 64 FP32 | 64 FP32 + 64 INT32 → 可全部用于 FP32 | 翻倍 |
| CUDA Cores 总数 | 5120 | **6912** | +35% |
| Tensor Cores/SM | 8 (Gen1) | 4 (Gen3) | 减半但单核性能翻倍+ |
| Tensor Core 精度 | FP16 | **TF32, FP16, BF16, INT8, INT4, INT1** | 精度矩阵完整 |
| L2 Cache | 6 MB | **40 MB** | +6.7× |
| 显存 | 32/16 GB HBM2 | **80/40 GB HBM2e** | +2.5× |
| 显存带宽 | 900 GB/s | **2039 GB/s** (80GB) | +2.3× |
| NVLink | 6 links × 25 GB/s = 300 GB/s | **12 links × 25 GB/s = 600 GB/s** | +2× |
| PCIe | Gen 3 x16 | **Gen 4 x16** | +2× |
| 制程 | TSMC 12nm | **TSMC 7nm** | 密度提升 |
| MIG | 不支持 | **支持** (最多 7 实例) | 新能力 |
| Sparsity | 不支持 | **支持** (2:4 结构化稀疏) | 新能力 |

> 注：A100 80GB 版本使用 HBM2e，显存时钟 1593 MHz；40GB 版本使用 HBM2，显存时钟 1215 MHz。本文数据基于 80GB 版本。

---

## 2. GA100 SM 内部结构

每个 GA100 SM 包含 4 个处理分区（Partitions），每个分区有独立的 L0 指令缓存、Warp Scheduler 和 Dispatch Unit：

```text
GA100 SM (108 个在 A100 中)
├── 4 × 处理分区 (Processing Partition)
│   ├── 1 × Warp Scheduler (32 threads/warp)
│   ├── 16 × FP32 CUDA Core ──┐
│   ├── 16 × INT32 CUDA Core ─┤ 在同一数据路径上
│   ├── 1 × Tensor Core (Gen3) │ → 每 SM 共 4 个 Tensor Core
│   ├── 4 × LD/ST Unit          │
│   └── 1 × SFU                 │
├── 128 KB L1 Data Cache / Shared Memory (可配置)
├── 读取-Only Data Cache
└── 寄存器文件: 65536 × 32-bit
```

**关键改进**：V100 中 FP32 和 INT32 数据路径分离，不能同时使用。A100 将两者合并到同一条数据路径上——当不需要 INT32 运算时，INT32 数据路径也可执行 FP32 指令，使每 SM 的 FP32 吞吐实际上可达 V100 的 2 倍。

---

## 3. 第三代 Tensor Core

A100 的 Tensor Core 是第三代，带来两个关键新能力：

### 3.1 TF32：训练精度的"甜点"

```text
TF32 = FP32 的 8-bit 指数 + FP16 的 10-bit 尾数 = 19 bits
```

TF32 的数值范围与 FP32 相同（不会溢出），但尾数精度截断到 10 bits。对于深度学习训练，这种精度截断在实践中不影响收敛，却能让 Tensor Core 吞吐达到 FP16 的水平——**不需要修改任何训练代码，只需将输入保持为 FP32**。

```c
// CUDA 11+ 中，TF32 默认开启
// cuBLAS / cuDNN 中的 FP32 矩阵乘法自动使用 TF32 Tensor Core
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
```

### 3.2 精度矩阵

| 精度 | 吞吐 (相对 FP32) | 主要用途 |
|------|-----------------|---------|
| TF32 | 8× | 训练（无代码改动） |
| FP16 | 8× | 训练 + 推理 |
| BF16 | 8× | 训练（与 TPU 兼容） |
| INT8 | 16× | 推理量化 |
| INT4 | 32× | 轻量推理 |
| FP64 | 2× | HPC（A100 FP64 ratio = 2:1，V100 相同） |

---

## 4. HBM2e 显存与 L2 Cache

A100 80GB 版本配备 5 个 HBM2e 堆栈，每个提供 1024-bit 接口：

```text
显存位宽: 5 stacks × 1024 bits = 5120 bits
显存时钟: 1593 MHz
理论带宽: 2 × 1.593 GHz × 5120 / 8 = 2039 GB/s
```

**L2 Cache** 从 V100 的 6 MB 暴增到 **40 MB**——这是 A100 最被低估的升级。40 MB 的 L2 意味着多数中等大小矩阵的中间结果可以被完整缓存，大幅减少 HBM 访问。

> 实测验证：在 A100 上运行 `transpose` (4 MB 矩阵)，optimized copy 达到 ~1188 GB/s (~58% 理论峰值)。64 MB 矩阵在 RTX 5090 (96 MB L2) 上达到峰值 1341 GB/s——A100 的 40 MB L2 对于 4 MB 矩阵同样提供了可观的加速。

---

## 5. A100 实测验证

以下数据来自一台 8 × A100-SXM4-80GB (NVSwitch Gen2) 服务器实测：

```bash
# deviceQuery 关键输出
nvidia-smi --query-gpu=index,name,memory.total,clocks.current.sm,clocks.current.memory --format=csv
# 3, NVIDIA A100-SXM4-80GB, 81920 MiB, 1410 MHz, 1593 MHz
```

| 属性 | deviceQuery 值 | 说明 |
|------|---------------|------|
| Compute Capability | 8.0 | Ampere |
| SMs | 108 | × 64 CUDA Cores/SM = 6912 |
| Max Threads/SM | 2048 | V100 也是 2048 |
| Max Blocks/SM | 32 | 比 RTX 5090 的 24 更多 |
| Shared Memory/SM | 167936 bytes | 可配置为 164 KB |
| Registers/SM | 65536 | |
| Async Copy Engines | 3 | RTX 5090 只有 2 |
| Warp Size | 32 | 所有 NVIDIA GPU 一样 |
| ECC | Enabled | 数据中心级 |
| NVLink | 12 links × 25 GB/s | 总双向 600 GB/s |
| PCIe | Gen 4 × 16 | max: 5 (RTX 5090) / 4 (A100) |

**deviceQuery 完整输出参见**：[`02_gpu_programming/04_profiling/README.md` 中提及的 deviceQuery](../../../02_gpu_programming/04_profiling/README.md)

**NVLink 拓扑**：8 卡通过 NVSwitch Gen2 全互联 (NV12)，每对 GPU 间 600 GB/s 双向。详见 [NVLink 诊断与实操](../../nvlink/nvlink_diagnostics.md)。

---

## 6. MIG：多实例 GPU

A100 是首款支持 **MIG (Multi-Instance GPU)** 的 NVIDIA GPU——将一张物理 GPU 切分为最多 7 个独立的 GPU 实例，每个实例有专用的 SM、显存、L2 Cache 和内存带宽：

```text
A100-80GB MIG 配置示例:
  7 × 1g.10gb  (10 GB, ~14 SMs)    — 最多实例
  3 × 2g.20gb  (20 GB, ~28 SMs)    — 均衡
  1 × 4g.40gb  (40 GB, ~56 SMs)    — 最大单实例
  1 × 7g.80gb  (80 GB, full GPU)   — 默认 (非 MIG)
```

**适用场景**：多租户推理服务、小模型训练、CI/CD 资源隔离。训练通常不用 MIG（需要整卡算力），推理场景中 MIG 可以避免多个服务竞争同一 GPU。

```bash
# 启用 MIG 模式
nvidia-smi -i 0 -mig 1

# 列出 MIG 实例
nvidia-smi mig --list-gpu-instances
```

---

## 7. 结构化稀疏 (2:4 Sparsity)

A100 的 Tensor Core 支持 **2:4 结构化稀疏**——在每 4 个连续权重值中，最多 2 个可以为零。硬件直接跳过零值的计算，使有效吞吐翻倍：

```text
权重向量: [w0, 0, w2, 0, w4, w5, 0, w7]
            └──2:4──┘  └──2:4──┘
稀疏后有效计算: 50% 的非零值 → Tensor Core 跳过零 → 2× 加速
```

配合 cuSPARSELt 库或 PyTorch 的 `torch.sparse` 使用，对推理场景尤其有效。

---

## 8. A100 vs 消费级 GPU

A100 与文档中频繁对比的 RTX 5090 的关键差异：

| 能力 | A100 | RTX 5090 | 影响 |
|------|------|---------|------|
| FP64 性能比 | **2:1** | 64:1 | A100 适合 HPC 双精度 |
| NVLink | **600 GB/s** | 不支持 | 多卡训练必须 NVLink |
| MIG | 支持 | 不支持 | 多租户 GPU 分区 |
| ECC | 默认开启 | 可选 | 数据中心可靠性 |
| Async Engines | 3 | 2 | 并发 D2H/H2D 能力 |
| Max Threads/SM | 2048 | 1536 | A100 SM 容更多线程 |
| L2 Cache | 40 MB | 96 MB | RTX 5090 L2 更大（但无 NVLink） |

---

## 参考

- [NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) — Ampere 架构白皮书
- [NVIDIA A100 产品规格](https://www.nvidia.com/en-us/data-center/a100/)
- [GPU 架构深入理解](README.md) — 本目录其他 GPU 型号分析
- [NVLink 技术入门](../../nvlink/nvlink_intro.md) — NVLink 3.0 细节
- [NVLink 诊断与实操](../../nvlink/nvlink_diagnostics.md) — A100 NVLink 实测
