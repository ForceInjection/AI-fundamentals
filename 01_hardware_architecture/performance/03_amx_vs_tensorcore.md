# CPU AMX vs GPU Tensor Core：混合计算场景分析

> 基于 Intel Xeon Platinum 8470Q (Sapphire Rapids, 支持 AMX) + NVIDIA RTX 5090 (Blackwell Tensor Core) 实际环境。

---

## 1. 什么是 AMX

**AMX (Advanced Matrix Extensions)** 是 Intel 从 Sapphire Rapids (第 4 代 Xeon) 起引入的矩阵运算扩展指令集，对标 NVIDIA Tensor Core。

### 1.1 核心组件

| 组件                     | 说明                                    |
| ------------------------ | --------------------------------------- |
| **Tile Register**        | 8 个 2D 寄存器 (1KB each)，存储矩阵数据 |
| `tdpbf16ps`              | BF16 矩阵乘 + FP32 累加                 |
| `tdpbusd`                | INT8 矩阵乘 + INT32 累加                |
| `tdpbuud`                | UINT8 矩阵乘 + UINT32 累加              |
| `tileload` / `tilestore` | Tile 寄存器与内存之间的数据搬运         |

### 1.2 本环境确认

```bash
lscpu | grep -E "amx_bf16|amx_tile|amx_int8"
# Flags: ... amx_bf16 amx_tile amx_int8 ...
```

三个 AMX 相关 flag 全部存在，确认 Sapphire Rapids 完整支持 AMX。

---

## 2. AMX vs Tensor Core 对比

### 2.1 硬件规格

| 维度      | Intel AMX (SPR)      | NVIDIA Tensor Core (RTX 5090) |
| --------- | -------------------- | ----------------------------- |
| 架构      | Sapphire Rapids      | Blackwell                     |
| FP32 核心 | AVX-512 (1×512b FMA) | CUDA Cores (170 SM × 128)     |
| 矩阵加速  | AMX tile (8×1KB)     | 5th Gen Tensor Core           |
| BF16 算力 | ~1.5 TFLOPS/socket   | 数百 TFLOPS (未公开)          |
| INT8 算力 | ~3 TOPS/socket       | 数百 TOPS                     |
| 内存      | DDR5, ~480 GB/s      | GDDR7, 1792 GB/s              |
| 功耗      | 350W/socket          | 575W                          |
| 延迟      | 极低 (无 PCIe 开销)  | 需经 PCIe 传输数据            |

### 2.2 适用场景

| 场景                    | 推荐              | 原因                      |
| ----------------------- | ----------------- | ------------------------- |
| 大规模矩阵乘 (GEMM)     | GPU Tensor Core   | 算力高 2-3 个数量级       |
| 小 batch 推理 (< 4)     | CPU AMX           | 避免 PCIe 往返延迟        |
| 实时性要求高            | CPU AMX           | 确定性延迟，无 PCIe       |
| 训练大批次              | GPU Tensor Core   | 高吞吐                    |
| 混合管道 (预处理→推理)  | AMX + Tensor Core | CPU 预处理 + GPU 核心计算 |
| 已有 CPU 管线，GPU 离线 | CPU AMX           | 减少架构改动成本          |

---

## 3. AMX 编程模型

### 3.1 通过 Intel AMX Intrinsics (C)

```c
#include <immintrin.h>

void amx_matmul_bf16() {
    // 1. 配置 tile 寄存器
    _tile_loadconfig(&config);

    // 2. 加载矩阵到 tile 寄存器
    _tile_loadd(0, buf_a, stride_a);  // A → tile 0
    _tile_loadd(1, buf_b, stride_b);  // B → tile 1
    _tile_loadd(2, buf_c, stride_c);  // C → tile 2

    // 3. 矩阵乘: tile[2] += tile[0] × tile[1]
    _tile_dpbf16ps(2, 0, 1);

    // 4. 存储结果
    _tile_stored(2, buf_c, stride_c);

    // 5. 释放 tile 寄存器
    _tile_release();
}
```

### 3.2 通过 PyTorch (高层封装)

```python
import torch

# CPU 上使用 AMX（PyTorch 2.0+ 自动利用）
x = torch.randn(1024, 1024, dtype=torch.bfloat16)
w = torch.randn(1024, 1024, dtype=torch.bfloat16)

# PyTorch 在 SPR 上自动调度到 AMX
y = torch.matmul(x, w)
```

### 3.3 编译要求

```bash
# C 代码编译 (需 GCC 12+ 或 Intel Compiler)
gcc -march=sapphirerapids -mamx-bf16 -mamx-tile -mamx-int8 \
    -o amx_test amx_test.c
```

---

## 4. 混合场景：CPU AMX + GPU Tensor Core 协同

### 4.1 典型 Pipeline

```text
数据输入 → CPU AMX 预处理 (tokenize/embed) → GPU Core (attention/FFN) → CPU 后处理
```

### 4.2 异步流水线伪代码

```c
// 阶段 1: CPU AMX 做 embedding 查表 + 投影
// (在 CPU pinned memory 中完成)
compute_embedding_amx(input, cpu_buffer);

// 阶段 2: 异步 H2D
cudaMemcpyAsync(gpu_buffer, cpu_buffer, size,
                cudaMemcpyHostToDevice, stream);

// 阶段 3: GPU Tensor Core 做 attention
launch_attention_kernel(gpu_buffer, stream);
```

### 4.3 NUMA 注意事项

CPU AMX 在哪个 socket 上执行，应与 GPU 的 NUMA 位置对齐。本环境中 GPU 位于 NUMA node 1，AMX kernel 线程应绑定到 node 1 的核（CPU 52-103, 156-207）。

```bash
taskset -c 52-103,156-207 ./hybrid_amx_cuda_program
```

---

## 5. 何时使用 AMX，何时用 GPU

```text
        小 batch ➡️ CPU AMX
        ↓
推理    大 batch ➡️ GPU Tensor Core
        ↓
        实时 < 1ms ➡️ CPU AMX
        ↓
训练    微调 ➡️ GPU Tensor Core
        ↓
        预训练 ➡️ GPU Tensor Core (唯一选择)
```

核心决策因素：

1. **batch size**：AMX 优势体现在小 batch，大 batch 时 GPU 的吞吐优势压倒一切
2. **实时性**：AMX 的延迟可预测（~μs 级），GPU 需经 PCIe + kernel launch（~10-50μs）
3. **现有架构**：如果管线已经在 CPU 上完成 tokenize/prefill，加 AMX 推理比引入 GPU 改动小

---

## 6. 延伸阅读

AMX 的详细编程和优化超出本文范围（本文侧重 GPU 基础设施），推荐资源：

- [Intel AMX Programming Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [PyTorch CPU 性能调优](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch)
