# Tensor Core GEMM 性能实测

> 基于 A100-SXM4-80GB (CC 8.0, Gen3 Tensor Core) 实测。运行官方 `cudaTensorCoreGemm` / `bf16TensorCoreGemm` samples，测量 FP16 (52.5 TFLOPS) 与 BF16 (90.9 TFLOPS) 的真实矩阵乘法性能，并与理论峰值和 HBM 带宽上限形成三角对标。

---

## 1. Tensor Core 性能模型

Tensor Core 的性能受三个硬件上限约束：

```text
                    ┌────────────────────┐
                    │  Tensor Core 算力   │ ← 峰值 TFLOPS 上限
                    │  156 TFLOPS (dense)│
                    └─────────┬──────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │  HBM 带宽     │   │ Shared Mem   │   │ 寄存器文件    │
   │  2039 GB/s   │   │ 带宽 ~12 TB/s │   │  65536/SM    │
   └──────────────┘   └──────────────┘   └──────────────┘
```

> **实际 TFLOPS = min(计算上限, 带宽能喂饱的数据量 × 每字节算数密度)**

对矩阵乘法（GEMM）：每从 HBM 读取 1 字节数据，可以做 O(N) 次乘加 → 算术密度极高 → 瓶颈在 Tensor Core 算力本身，不在带宽。

> 前置阅读：[HBM 显存带宽测试](../04_profiling/03_hbm_bandwidth_test.md)、[NVIDIA A100 架构详解](../../01_hardware_architecture/nvidia/understand_gpu_architecture/07_a100_architecture.md)

---

## 2. 理论峰值速查

A100 (GA100, CC 8.0) 的 Tensor Core 规格：

| 精度 | 稠密 TFLOPS | 稀疏 (2:4) TFLOPS | FMA/TC/clock |
| ---- | ----------- | ----------------- | ------------ |
| FP64 | 9.7         | —                 | —            |
| TF32 | **156**     | 312               | 256          |
| FP16 | **156**     | 312               | 256          |
| BF16 | **156**     | 312               | 256          |
| INT8 | 312         | 624               | 512          |
| INT4 | 624         | 1248              | 1024         |

**公式**：

```text
峰值 TFLOPS = 108 SM × 4 TC/SM × FMA率/TC × 1.41 GHz × 2 (FMA=2 ops)
           = 108 × 4 × 256 × 1.41 × 2 / 1000 = 156 TFLOPS (BF16/FP16/TF32 dense)
```

> 对比 RTX 5090 (GDDR7, 无 NVLink)：虽然算力更高（~105 TFLOPS FP16 dense），但多卡时完全依赖 PCIe。

---

## 3. A100 实测

### 3.1 FP16 Tensor Core GEMM

使用官方 `cudaTensorCoreGemm` sample（4096³ 矩阵，Tile Size 256×128×32）：

```bash
cd cuda-samples/Samples/3_CUDA_Features/cudaTensorCoreGemm
nvcc -arch=sm_80 -I../../../Common -o cudaTensorCoreGemm cudaTensorCoreGemm.cu -lcudart
CUDA_VISIBLE_DEVICES=3 ./cudaTensorCoreGemm
```

```text
GPU Device 0: "Ampere" with compute capability 8.0

M: 4096 (16 x 256)
N: 4096 (16 x 256)
K: 4096 (16 x 256)
Required shared memory size: 64 Kb
Computing... using high performance kernel compute_gemm

Time: 2.616 ms
TFLOPS: 52.53
```

### 3.2 BF16 Tensor Core GEMM

`bf16TensorCoreGemm` 使用 `cp.async`（异步拷贝）将 HBM→Shared Memory 与计算流水线化，8192³ 矩阵：

```bash
cd cuda-samples/Samples/3_CUDA_Features/bf16TensorCoreGemm
nvcc -arch=sm_80 -I../../../Common -o bf16TensorCoreGemm bf16TensorCoreGemm.cu -lcudart
CUDA_VISIBLE_DEVICES=3 ./bf16TensorCoreGemm
```

```text
M: 8192 (16 x 512)
N: 8192 (16 x 512)
K: 8192 (16 x 512)
Required shared memory size: 72 Kb
Computing using high performance kernel = 0 - compute_bf16gemm_async_copy

Time: 12.101 ms
TFLOPS: 90.86
```

### 3.3 结果对标

| 精度                 | 矩阵尺寸 | 实测 TFLOPS    | vs 稠密峰值 (156 TFLOPS) | 瓶颈分析                                |
| -------------------- | -------- | -------------- | ------------------------ | --------------------------------------- |
| FP16                 | 4096³    | **52.5**       | 33.7%                    | 矩阵偏小，SM 未充分利用                 |
| BF16                 | 8192³    | **90.9**       | 58.3%                    | 接近实际可用上限（~60% 峰值是良好结果） |
| — (FP32 cuBLAS 参考) | —        | ~15-19 (非 TC) | —                        | 无 Tensor Core 的 FP32 累积路径         |

**为什么 8192³ 比 4096³ 利用率高？**

```text
4096³ 矩阵:
  └── 每个 SM 分到的 tiles 较少 → Waves/SM 低
  └── Shared memory tile 需要 64 KB → 每个 SM 最多 2.6 个 blocks

8192³ 矩阵:
  └── 每个 SM 分到更多 tiles → Waves/SM 高 → Occupancy 更高
  └── cp.async 流水线充分运转 → HBM 延迟被完全隐藏
```

**58% 的峰值利用率是正常的**。Tensor Core 在实际 GEMM 中能达到 50-70% 的稠密峰值即是优秀结果——剩余损耗来自 tile 边缘效应、shared memory 同步、以及 pipeline 启动/排空开销。

---

## 4. 代码精要：从 GEMM Kernel 看 Tensor Core 编程

以 `cudaTensorCoreGemm` 的关键路径为例：

```c
// 1. 声明 Fragment —— 编译器将 fragment 映射到寄存器
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag;

// 2. 初始化累加器
nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

// 3. 沿 K 维度循环，每次加载 16×16 tile
for (int k = 0; k < K; k += 16) {
    nvcuda::wmma::load_matrix_sync(a_frag, A_tile + k_offset, lda);
    nvcuda::wmma::load_matrix_sync(b_frag, B_tile + k_offset, ldb);
    nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);  // ← Tensor Core 指令
}

// 4. 写回结果
nvcuda::wmma::store_matrix_sync(C_tile, acc_frag, ldc, nvcuda::wmma::mem_row_major);
```

**WMMA (Warp Matrix Multiply-Accumulate)** 是 CUDA 提供的 Tensor Core 高级 API——无需手写 MMA PTX 指令。`nvcuda::wmma::mma_sync` 一行代码即触发 16³ Tensor Core 操作（A100 上：1024 FP16 FMA/tile）。

> 对于追求极致性能的场景，BF16 sample 使用的 `cp.async` + 手动 Shared Memory 管理可以提供更好的流水线效果——这就是 `bf16TensorCoreGemm` 能达到 90.9 TFLOPS 的原因。

---

## 5. TFLOPS / 带宽 / 算数密度三角

把 GEMM 的算力数据与带宽数据放在一起：

| 指标            | 值               | 说明               |
| --------------- | ---------------- | ------------------ |
| BF16 GEMM 实测  | **90.9 TFLOPS**  | 8192³ 矩阵         |
| HBM2e 带宽      | 2039 GB/s        | 理论峰值           |
| HBM2e 实测 D2D  | ~1188 GB/s (4MB) | `transpose` sample |
| NVLink P2P 带宽 | 239 GB/s (单向)  | `simpleP2P` sample |
| PCIe H2D 带宽   | ~28 GB/s (Gen 4) | `pcie_bw_test`     |

**GEMM 是算力瓶颈，不是带宽瓶颈**：

```text
8192³ BF16 GEMM:
  数据读取: 3 × 8192² × 2 bytes = 402 MB (A, B, partial C)
  计算量:   2 × 8192³ = 1.1T FLOPs
  算数密度: 1.1T / 402 MB ≈ 2740 FLOP/byte

  HBM 带宽 (2039 GB/s) 能支撑: 2039 × 2740 = 5587 TFLOPS >> 156 TFLOPS
  → 算力瓶颈，带宽绰绰有余
```

对比：Attention 机制的算数密度仅 ~1-10 FLOP/byte → **带宽瓶颈**。这就是为什么 GEMM 优化关注 SM 利用率和 Tensor Core 指令效率，而 Attention 优化关注 KV Cache 带宽压缩。

---

## 6. 相关文档

- [`03_hbm_bandwidth_test.md`](../04_profiling/03_hbm_bandwidth_test.md)：片内带宽 —— Tensor Core 算力与之形成"算力 vs 带宽"对标
- [`08_p2p_bandwidth.md`](../04_profiling/08_p2p_bandwidth.md)：片间 P2P 带宽 —— GEMM 用于 TP 时依赖 P2P 交换中间结果
- [`07_a100_architecture.md`](../../01_hardware_architecture/nvidia/understand_gpu_architecture/07_a100_architecture.md)：A100 SM 和 Tensor Core 的硬件设计

## 参考

- [NVIDIA cuda-samples: cudaTensorCoreGemm](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/cudaTensorCoreGemm)
- [CUDA WMMA API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [NVIDIA A100 架构白皮书](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
