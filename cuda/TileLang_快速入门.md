# TileLang 快速入门

## 1. 背景介绍

### 1.1 什么是 TileLang

TileLang（Tile Language）是一种专为高性能 GPU/CPU 内核开发而设计的简洁领域特定语言（DSL）。它采用 Pythonic 语法，底层基于 [TVM](https://tvm.apache.org/) 编译器基础设施，旨在让开发者专注于生产力，同时不牺牲获得最先进性能所需的底层优化。

> 项目地址：<https://github.com/tile-ai/tilelang/>
> 文档地址：<https://tile-ai.github.io/tilelang/>

### 1.2 TileLang 的核心优势

1. **简洁的语法**：采用 Python 风格的语法，降低学习成本
2. **高性能**：基于 TVM 编译器基础设施，提供底层优化能力
3. **跨平台支持**：支持 NVIDIA GPU（CUDA）、AMD GPU（HIP）和 CPU
4. **丰富的算子支持**：内置 GEMM、FlashAttention、LinearAttention 等高性能算子
5. **自动优化**：支持自动流水线、内存布局优化、L2 缓存友好的重排等特性

### 1.3 适用场景

TileLang 特别适合以下场景：

- 高性能矩阵运算（GEMM、量化 GEMM）
- 注意力机制实现（FlashAttention、LinearAttention）
- 自定义 GPU 内核开发
- 深度学习算子优化

### 1.4 支持的硬件平台

TileLang 支持多种硬件平台，已在以下设备上经过测试和验证：

- **NVIDIA GPU**：H100（支持 Auto TMA/WGMMA）、A100、V100、RTX 4090、RTX 3090、RTX A6000
- **AMD GPU**：MI250（支持 Auto MatrixCore）、MI300X（支持 Async Copy）
- **CPU**：x86_64 架构处理器（支持 AVX2/AVX-512 指令集）

> 以上信息出自官方代码库，从目前的新闻看，TileLang 已经被国产 GPU 支持。

---

## 2. 快速入门：矩阵乘法对比

### 2.1 传统 CUDA 实现

首先，让我们看看传统的 CUDA 矩阵乘法实现：

```cpp
// CUDA 矩阵乘法内核
__global__ void matmul_cuda(float* A, float* B, float* C, int M, int N, int K) {
    // 计算线程索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (row < M && col < N) {
        float sum = 0.0f;

        // 计算点积
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// 优化版本：使用共享内存的分块矩阵乘法
#define TILE_SIZE 16

__global__ void matmul_cuda_tiled(float* A, float* B, float* C, int M, int N, int K) {
    // 共享内存声明
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // 线程和块索引
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // 计算全局索引
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // 分块计算
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // 加载数据到共享内存
        if (row < M && tile * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (col < N && tile * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // 计算部分乘积
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // 写回结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 主机代码
void launch_matmul_cuda(float* A, float* B, float* C, int M, int N, int K) {
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_cuda_tiled<<<gridSize, blockSize>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

### 2.2 TileLang 实现

现在让我们看看 TileLang 的实现方式。以下代码展示了一个简洁的矩阵乘法实现：

> **代码来源**：[example_gemm.py](https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm.py)

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])
    return gemm

# 使用示例
def main():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)

    import torch
    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()
    c = kernel(a, b)

    # 验证正确性
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("正确性验证通过")
```

**关键特性说明**：

- `@tilelang.jit(out_idx=[-1])`：JIT 编译装饰器，指定最后一个参数为输出
- `T.Pipelined`：自动实现多级流水线优化
- `T.copy`：自动并行化数据传输
- `T.gemm`：调用最优的矩阵乘法实现

### 2.3 对比分析

| 特性           | CUDA                           | TileLang                                   |
| -------------- | ------------------------------ | ------------------------------------------ |
| **代码行数**   | ~80 行                         | ~30 行                                     |
| **内存管理**   | 手动管理共享内存、同步         | 自动管理，声明式语法                       |
| **优化复杂度** | 需要手动实现分块、流水线等优化 | 内置自动优化（`T.Pipelined`、`T.copy` 等） |
| **可读性**     | 底层细节较多，可读性较差       | 高层抽象，语义清晰                         |
| **错误处理**   | 需要手动边界检查和错误处理     | 编译器自动处理                             |
| **跨平台**     | 仅支持 NVIDIA GPU              | 支持 CUDA、HIP、CPU                        |
| **性能**       | 手动优化可达到极致性能         | 自动优化达到接近手动优化的性能             |

### 2.4 TileLang 的关键优势

1. **简洁性**：TileLang 用 30 行代码实现了 CUDA 需要 80+ 行的功能
2. **自动优化**：
   - `T.Pipelined`：自动实现多级流水线优化
   - `T.copy`：自动并行化数据传输
   - `T.gemm`：自动调用最优的矩阵乘法实现
3. **内存管理**：自动处理共享内存分配和同步
4. **类型安全**：编译时类型检查，减少运行时错误

### 2.5 高级特性对应

对于有 CUDA 开发经验的用户，以下是 CUDA 高级特性与 TileLang 的对应关系：

| CUDA 特性 | TileLang 对应 | 说明 |
|-----------|---------------|------|
| **动态并行** | 自动并行化 | TileLang 编译器自动分析数据依赖，生成最优并行策略 |
| **统一内存** | 自动内存管理 | 无需手动 `cudaMalloc`/`cudaFree`，编译器自动处理内存分配 |
| **流和事件** | 自动调度优化 | `T.Pipelined` 自动实现异步执行和同步优化 |
| **共享内存** | `T.alloc_shared` | 声明式共享内存分配，自动优化访问模式 |
| **寄存器优化** | `T.alloc_fragment` | 自动寄存器分配和数据局部性优化 |
| **Warp 级原语** | 内置算子 | `T.gemm`、`T.reduce` 等自动调用最优 Warp 级实现 |

**核心理念**：TileLang 将 CUDA 的底层控制抽象为高级语义，让开发者专注于算法逻辑而非硬件细节。

---

## 3. 编译运行

### 3.1 系统要求

#### 3.1.1 PyPI 安装要求

- **操作系统**：Ubuntu 20.04 或更高版本
- **Python**：3.8 或更高版本
- **CUDA**：11.0 或更高版本（NVIDIA GPU）

#### 3.1.2 源码编译要求

- **操作系统**：Linux
- **Python**：3.7 或更高版本
- **CUDA**：10.0 或更高版本（NVIDIA GPU）
- **LLVM**：< 20（如果使用捆绑的 TVM 子模块）
- **ROCm**：5.0 或更高版本（AMD GPU，可选）

### 3.2 安装步骤

#### 3.2.1 基础依赖安装

```bash
# 更新系统包
sudo apt-get update
sudo apt-get install -y python3-setuptools gcc libtinfo-dev zlib1g-dev \
                        build-essential cmake libedit-dev libxml2-dev

# 安装 Python 包管理器依赖
pip install --upgrade pip setuptools wheel

# 安装核心依赖
pip install Cython>=3.0.0 numpy>=1.23.5 tqdm>=4.62.3 \
            typing_extensions>=4.10.0 cloudpickle ml_dtypes psutil torch
```

#### 3.2.2 TileLang 安装

**方法一：PyPI 安装（推荐）**：

```bash
# 安装最新稳定版本
pip install tilelang

# 或安装开发版本
pip install git+https://github.com/tile-ai/tilelang
```

**方法二：源码编译安装**：

```bash
# 克隆仓库
git clone https://github.com/tile-ai/tilelang.git
cd tilelang

# 安装构建依赖
pip install -r requirements-build.txt

# 编译安装
pip install -e . -v
```

**方法三：Nightly 版本**：

```bash
# 安装最新开发版本（包含最新特性）
pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu121/
```

> **注意**：更多安装方式请参考官方文档：[Installation Guide](https://github.com/tile-ai/tilelang/blob/main/docs/get_started/Installation.md)

### 3.3 编译原理与基础语法

#### 3.3.1 JIT 编译流程

TileLang 采用 **JIT（Just-In-Time）编译模式**，在运行时动态编译高性能内核。这种设计具有以下优势：

**编译模式特点**：

- **延迟编译**：只在首次调用时编译，避免不必要的编译开销
- **参数特化**：根据具体的张量形状和数据类型生成优化代码
- **缓存机制**：编译结果会被缓存，后续调用直接复用
- **多目标支持**：同一份代码可编译到不同硬件平台

**基本编译流程**：

```python
# 1. 定义内核函数
@tilelang.jit(target="cuda")  # 指定编译目标
def my_kernel(...):
    # TileLang 高级语法

# 2. 首次调用触发编译
result = my_kernel(input_data)  # 此时进行 JIT 编译

# 3. 后续调用直接执行
result2 = my_kernel(input_data2)  # 复用已编译的内核
```

这种 JIT 编译模式使得 TileLang 能够在保持高级语法简洁性的同时，生成与手写 CUDA 代码相当的高性能内核。

#### 3.3.2 TileLang 到 CUDA 的编译过程

TileLang 通过 **TVM 编译器基础设施**将高级语法转换为高性能的 CUDA 代码。以下是 CUDA 特定的编译技术细节：

**CUDA 代码生成架构**：

- **专用代码生成器**：`codegen_cuda.cc` 负责将 TVM TIR 转换为 CUDA C++ 代码
- **内存层次优化**：自动管理全局内存、共享内存和寄存器的数据流
- **线程块配置**：根据硬件特性自动选择最优的网格和线程块大小

**TVM TIR 优化策略**：

1. **循环优化策略**
   - **循环分块（Loop Tiling）**：将大循环分解为缓存友好的小块
   - **循环重排（Loop Reordering）**：调整循环嵌套顺序优化内存访问
   - **循环融合（Loop Fusion）**：合并相邻循环减少内存往返

2. **内存访问模式优化**
   - **数据布局变换**：自动选择最优的数据排列方式（行优先/列优先）
   - **预取优化**：在计算前预加载数据到共享内存
   - **内存合并**：确保 Warp 内线程的内存访问模式最优

3. **算子融合机制**
   - **垂直融合**：将生产者-消费者算子合并，减少中间结果存储
   - **水平融合**：合并并行的独立算子，提高硬件利用率
   - **自动调度**：基于硬件特性自动选择最优的融合策略

**CUDA 特定优化技术**：

1. **内存访问优化**
   - 合并访问（Coalesced Access）：确保连续线程访问连续内存
   - 共享内存分块（Shared Memory Tiling）：减少全局内存访问
   - 寄存器重用：最大化寄存器利用率

2. **计算优化**
   - 循环展开（Loop Unrolling）：减少分支开销
   - 指令级并行（ILP）：充分利用 GPU 流水线
   - Tensor Core 利用：自动生成 MMA 指令（支持的硬件上）

**生成的 CUDA 代码示例**：

```cpp
extern "C" __global__ void __launch_bounds__(256) main_kernel(
    float* __restrict__ A,
    float* __restrict__ B,
    float* __restrict__ C
) {
    // 共享内存声明
    __shared__ float A_shared[128][32];
    __shared__ float B_shared[32][128];
    
    // 寄存器数组
    float C_local[8][8];
    
    // 优化的内存访问和计算逻辑
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        // 向量化加载和计算
    }
}
```

**调试和性能分析工具**：

```python
# 查看生成的 CUDA 源码
kernel = tilelang.compile(your_function, target="cuda")
cuda_source = kernel.get_kernel_source()
print("Generated CUDA kernel:\n", cuda_source)

# 编译过程调试回调
from tilelang.engine.callback import register_cuda_postproc_callback

@register_cuda_postproc_callback
def debug_cuda_code(code, target):
    print("Final CUDA code:", code)
    return code
```

**性能特点**：

- **零开销抽象**：高级语法不会引入额外的运行时开销
- **硬件感知优化**：根据目标 GPU 架构（如 SM 版本）进行特定优化
- **与手写 CUDA 相当的性能**：在许多场景下可达到或超越手写 CUDA 代码的性能

#### 3.3.3 基础语法结构

TileLang 使用 `@tilelang.jit` 装饰器来标记需要编译的函数：

```python
import tilelang
import tilelang.language as T

# 基本用法
@tilelang.jit(out_idx=[-1])  # 指定最后一个参数为输出
def kernel_function(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def actual_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # 1. 定义内核网格和线程配置
        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
            # 2. 分配共享内存和寄存器
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # 3. 计算逻辑
            # ...

    return actual_kernel
```

**编译选项说明**：

```python
@tilelang.jit(
    out_idx=[-1],           # 输出参数索引
    target="cuda",          # 目标平台：cuda, rocm, cpu
    debug=False,            # 调试模式
)
```

### 3.4 安装验证与第一个程序

#### 3.4.1 安装验证

创建测试文件 `test_installation.py`：

> **代码来源**：参考 [example_elementwise_add.py](https://github.com/tile-ai/tilelang/blob/main/examples/elementwise/example_elementwise_add.py)

```python
import tilelang
import tilelang.language as T
import torch

print(f"TileLang 版本: {tilelang.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

# 简单的向量加法测试
@tilelang.jit(out_idx=[-1])
def vector_add(M, N, block_M, block_N, dtype="float32"):
    @T.prim_func
    def add_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                row = by * block_M + i
                col = bx * block_N + j
                if row < M and col < N:
                    C[row, col] = A[row, col] + B[row, col]
    return add_kernel

# 测试执行
M, N = 1024, 1024
kernel = vector_add(M, N, 128, 128)

a = torch.randn(M, N).cuda()
b = torch.randn(M, N).cuda()
c = kernel(a, b)

# 验证结果
ref_c = a + b
torch.testing.assert_close(c, ref_c, rtol=1e-5, atol=1e-5)
print("TileLang 安装验证成功！")
```

运行验证：

```bash
python test_installation.py
```

#### 3.4.2 完整的矩阵乘法示例

以下是一个完整的矩阵乘法示例，展示了 TileLang 的编译和运行流程：

> **代码来源**：参考 [quickstart.py](https://github.com/tile-ai/tilelang/blob/main/examples/quickstart.py)

```python
import tilelang
import tilelang.language as T
import torch

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def matmul_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # 分配共享内存和寄存器
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)

            # 流水线化的矩阵乘法计算
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return matmul_kernel

# 使用示例
def main():
    M, N, K = 1024, 1024, 1024
    block_M, block_N, block_K = 128, 128, 32

    # 编译内核（一次性）
    kernel = matmul(M, N, K, block_M, block_N, block_K)

    # 创建测试数据
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    # 执行内核
    c = kernel(a, b)

    # 验证正确性
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("矩阵乘法验证通过")

if __name__ == "__main__":
    main()
```

### 3.5 内存管理与错误处理

#### 3.5.1 内存管理策略

```python
# 自动内存管理（推荐）
c = kernel(a, b)  # 自动分配输出张量

# 手动内存管理（高性能场景）
c = torch.empty(M, N, device="cuda", dtype=torch.float16)
kernel(a, b, c)  # 复用预分配的张量
```

#### 3.5.2 常见错误处理

```python
try:
    kernel = matmul(M, N, K, block_M, block_N, block_K)
    c = kernel(a, b)
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("正确性验证通过")
except RuntimeError as e:
    if "out of memory" in str(e):
        torch.cuda.empty_cache()
        print("建议：减少矩阵大小或分块大小")
    else:
        print(f"内核执行失败: {e}")
except Exception as e:
    print(f"执行失败: {e}")
```

**常见错误类型**：

- **CUDA out of memory**：减少分块大小或清理 GPU 缓存
- **编译错误**：检查内核函数定义和参数类型
- **运行时错误**：验证输入张量的设备和数据类型

---

## 4. 总结

本文通过快速入门指南，让开发者可以快速理解 TileLang 的优势，掌握基本的使用方法。

---
