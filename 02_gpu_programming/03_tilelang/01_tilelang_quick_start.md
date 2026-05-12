# TileLang 快速入门

TileLang 是一种面向现代异构加速器（如 NVIDIA GPU、AMD GPU）的高性能算子开发语言与编译器基础设施。它基于 Python DSL 提供了接近原生的表达能力，并通过底层 TVM 编译器引擎自动处理复杂的硬件资源分配与指令调度，使开发者能够在避免深陷 CUDA 底层细节的前提下，快速编写出性能媲美甚至超越手写 CUDA 的定制化算子。本文将带领大家从环境搭建开始，逐步掌握 TileLang 的核心编程范式与最佳实践。

## 1. 为什么需要 TileLang？

手写高性能 CUDA Kernel 是一项门槛极高的工作——开发者不仅要设计算法，还要同时处理共享内存分配、线程同步、Bank Conflict 规避、寄存器布局和指令流水线等大量底层细节。一个高质量的 GEMM Kernel 动辄数百行代码，其中算法逻辑往往只占一小部分，其余全是硬件适配。

TileLang 的出现，正是为了把开发者从这些繁琐的硬件细节中解放出来。

### 1.1 手写 CUDA 的隐性成本

下面是一个典型的分块矩阵乘法 Kernel 需要处理的事项：

| 关注点        | 手写 CUDA 需要做什么                                    |
| ------------- | ------------------------------------------------------- |
| 共享内存      | 手动 `__shared__` 声明、加载、边界填充、`__syncthreads` |
| 寄存器布局    | 手动分配合适数量的寄存器变量，避免 spill                |
| 指令选择      | 手写 WMMA/MMA 指令，管理 Fragment 布局                  |
| 流水线        | 手写 multi-buffering，管理 `cp.async` 和 commit group   |
| 跨平台        | 为不同 GPU 架构编写不同的 Kernel 变体                   |
| Bank Conflict | 手动 padding shared memory                              |

这些工作量大、易出错，且与具体硬件强绑定。当新架构（如 Hopper → Blackwell）出现时，大量优化代码需要重写。

### 1.2 编译器驱动的算子开发

TileLang 的思路与 LLVM 类似：将算法描述与硬件适配解耦。开发者用 Pythonic DSL 描述**做什么**（数据分块、计算逻辑），编译器负责**怎么做**（内存分配、指令调度、流水线优化）。

- **底层**：基于 [TVM](https://tvm.apache.org/) 编译器基础设施，复用成熟的 TIR 优化和 CodeGen
- **语法**：Python 原生风格，`@tilelang.jit` 装饰器标记编译入口
- **平台**：支持 CUDA（NVIDIA）、HIP（AMD）、CPU（x86 AVX2/AVX-512）

> 项目地址：<https://github.com/tile-ai/tilelang/>
> 文档地址：<https://tile-ai.github.io/tilelang/>

---

## 2. 环境搭建

TileLang 依赖 Python 3.8+ 与 CUDA 11.0+ 基础环境，支持通过 PyPI 快速分发或源码编译，以适配 NVIDIA、AMD 等多种异构计算后端。

### 2.1 系统要求

**PyPI 安装**：Ubuntu 20.04+、Python 3.8+、CUDA 11.0+

**源码编译**：Linux、Python 3.7+、CUDA 10.0+、LLVM < 20、CMake + GCC

**已验证的硬件**：H100 (Auto TMA/WGMMA)、A100、V100、RTX 4090/3090/A6000；AMD MI250 (Auto MatrixCore)、MI300X (Async Copy)；x86_64 CPU (AVX2/AVX-512)。国产 GPU 也已开始支持。

### 2.2 安装

**推荐方式 — PyPI**：

```bash
pip install tilelang

# 或开发版本
pip install git+https://github.com/tile-ai/tilelang
```

**源码编译**：

```bash
git clone https://github.com/tile-ai/tilelang.git
cd tilelang
pip install -r requirements-build.txt
pip install -e . -v
```

**Nightly**：

```bash
pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu121/
```

> 更多方式见 [官方安装指南](https://github.com/tile-ai/tilelang/blob/main/docs/get_started/Installation.md)。

### 2.3 验证安装

通过 JIT 编译一个简单的二维向量加法 Kernel，可完整验证 TileLang 的编译链路与硬件后端调用逻辑。

> 代码参考：[example_elementwise_add.py](https://github.com/tile-ai/tilelang/blob/main/examples/elementwise/example_elementwise_add.py)

```python
import tilelang
import tilelang.language as T
import torch

# 打印基础环境信息
print(f"TileLang 版本: {tilelang.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

# 使用 jit 装饰器进行即时编译，out_idx 指定输出张量所在参数索引
@tilelang.jit(out_idx=[-1])
def vector_add(M, N, block_M, block_N, dtype="float32"):
    # 声明一个计算原语
    @T.prim_func
    def add_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # 划分 Grid 和 Block，每个 Block 包含 128 个线程
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # 自动并行化二维网格迭代
            for i, j in T.Parallel(block_M, block_N):
                # 计算全局线程索引
                row = by * block_M + i
                col = bx * block_N + j
                # 边界检查并执行逐元素加法
                if row < M and col < N:
                    C[row, col] = A[row, col] + B[row, col]
    return add_kernel

# 测试参数初始化
M, N = 1024, 1024
kernel = vector_add(M, N, 128, 128)

# 准备测试数据并运行 Kernel
a = torch.randn(M, N).cuda()
b = torch.randn(M, N).cuda()
c = kernel(a, b)

# 与 PyTorch 原生实现进行精度对比
ref_c = a + b
torch.testing.assert_close(c, ref_c, rtol=1e-5, atol=1e-5)
print("TileLang 安装验证成功！")
```

```bash
python test_installation.py
```

---

## 3. 核心概念与第一个算子

TileLang 采用声明式编程范式，通过 `@tilelang.jit` 与 `T.Kernel` 等核心原语，将复杂的硬件资源分配与多级调度逻辑抽象为紧凑的 Python 语法结构。

### 3.1 基础语法结构

TileLang 用 `@tilelang.jit` 装饰器标记编译入口。内核函数包含三个要素：

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])             # 输出参数索引；可选 target="cuda"/"rocm"/"cpu"
def my_kernel(M, N, K, block_M, block_N, block_K):
    @T.prim_func                         # 标记为原始函数
    def actual_kernel(
        A: T.Tensor((M, K), dtype),      # 类型注解即形状声明
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):  # 定义 Grid + Block
            A_shared = T.alloc_shared((block_M, block_K), dtype)  # 共享内存
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)  # 寄存器
            # 计算逻辑 ...
    return actual_kernel
```

| 原语                      | 作用                                |
| ------------------------- | ----------------------------------- |
| `T.Kernel(gx, gy, t)`     | 定义 Block 网格与每 Block 线程数    |
| `T.alloc_shared(shape)`   | 声明共享内存（相当于 `__shared__`） |
| `T.alloc_fragment(shape)` | 声明寄存器片段（Tile 级别变量）     |
| `T.Parallel(m, n)`        | 自动并行化迭代                      |
| `T.Pipelined(n, stages)`  | 多级软件流水线                      |
| `T.copy(src, dst)`        | 自动并行数据传输                    |
| `T.gemm(A, B, C)`         | 调用最优矩阵乘实现                  |

### 3.2 实战：矩阵乘法

**传统 CUDA 需要处理什么**：一个分块矩阵乘法 Kernel 通常需要手动管理共享内存加载、边界检查、`__syncthreads()` 同步、循环展开等。朴素实现 30 行，优化版本轻松超过 80 行，且大部分代码与算法逻辑无关。

**TileLang 实现** — 约 30 行，语义清晰：

> 代码参考：[example_gemm.py](https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm.py)

```python
import tilelang
import tilelang.language as T
import torch

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # 划分 Grid 和 Block，并指定每 Block 的线程数
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # 声明位于共享内存中的张量
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            # 声明位于寄存器中的本地累加片段
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            # 初始化寄存器片段为 0
            T.clear(C_local)
            # 开启 3 级流水线循环，自动处理异步加载与计算重叠
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # 将全局内存中的数据分块拷贝到共享内存
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                # 调用矩阵乘法指令，编译器将自动映射为 Tensor Core 指令
                T.gemm(A_shared, B_shared, C_local)

            # 计算完成后，将寄存器片段中的结果写回全局内存
            T.copy(C_local, C[by * block_M, bx * block_N])
    return gemm

# 编译并使用
M, N, K = 1024, 1024, 1024
kernel = matmul(M, N, K, 128, 128, 32)

# 构建 PyTorch 输入数据
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = kernel(a, b)

# 精度校验
ref_c = a @ b
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("矩阵乘法验证通过")
```

关键 API：

- `T.Pipelined(n, num_stages=3)`：自动生成 3 级流水线，隐藏全局内存延迟
- `T.copy`：自动并行数据传输，处理合并访问与边界填充
- `T.gemm`：编译器自动映射到目标架构最优的 Tensor Core/Warp 级指令

### 3.3 CUDA → TileLang 对照表

TileLang 编译器隐式接管了传统 CUDA 编程中繁琐的共享内存同步、边界检查与指令调度机制，开发者仅需关注高层数据流声明。

| CUDA 概念               | TileLang 原语      | 变化                                 |
| ----------------------- | ------------------ | ------------------------------------ |
| `__shared__` 手动管理   | `T.alloc_shared`   | 声明式，编译器自动处理加载/同步      |
| `__syncthreads()`       | — (隐式)           | 编译器自动插入，消除遗漏风险         |
| 手动边界检查            | — (隐式)           | 编译器根据形状自动生成               |
| `cudaMalloc`/`cudaFree` | — (自动)           | 张量生命周期由 PyTorch/TileLang 管理 |
| WMMA/MMA 指令           | `T.gemm`           | 一行调用，编译器选最优指令           |
| `cp.async` + commit     | `T.copy` 或隐式    | 编译器自动插入异步拷贝和流水线       |
| 手动循环展开            | — (自动)           | TVM TIR 自动应用分块/融合/展开       |
| CUDA Streams            | `T.Pipelined`      | 多级流水线自动处理异步执行           |
| 寄存器分配              | `T.alloc_fragment` | 声明式，编译器优化 spill             |

> **核心理念**：TileLang 不改变你思考算法的方式，它只是把"如何高效映射到硬件"这件事交给了编译器。

---

## 4. 进阶：编译原理与调试

TileLang 依托 TVM TIR (Tensor Intermediate Representation) 基础设施，将 Python DSL 实时降级并应用 Tiling、Fusion 等硬件感知优化，最终生成特化的 PTX 或机器码（仅关注业务集成的开发者可跳过本章）。

### 4.1 JIT 编译流程

TileLang 采用 **JIT 编译模式**：首次调用时编译，编译结果缓存。

```python
@tilelang.jit(target="cuda")
def my_kernel(...): ...

result = my_kernel(input)   # 首次调用 → 编译 + 执行
result2 = my_kernel(input2)  # 后续调用 → 缓存命中，直接执行
```

编译流程：`TileLang DSL` → `TVM TIR` → `优化后的 TIR` → `CUDA C++` → `PTX` → `可执行 Kernel`

关键特性：

- **参数特化**：编译器根据首次调用的形状和 dtype 生成特化代码
- **缓存机制**：编译结果缓存到磁盘，跨进程复用
- **多目标**：同一份 DSL 可编译到 CUDA / HIP / CPU

### 4.2 TVM TIR 到 CUDA 的转换

TileLang DSL 并非直接翻译为 CUDA C++，而是先降级为 TVM 的中间表示 TIR，经过与硬件无关的通用优化后，再由 `codegen_cuda.cc` 生成特化的 CUDA 源码。理解这一流程有助于在性能不达预期时判断瓶颈出在算法描述层还是代码生成层。

编译器自动应用的优化：

| 类别     | 优化策略                                        |
| -------- | ----------------------------------------------- |
| 循环     | 分块 (Tiling)、重排 (Reorder)、融合 (Fusion)    |
| 内存访问 | 合并访问 (Coalesced)、共享内存分块、预取        |
| 计算     | 循环展开、指令级并行 (ILP)、Tensor Core MMA     |
| 算子融合 | 垂直融合（生产者-消费者）、水平融合（独立算子） |

**性能特点**：零开销抽象（DSL 无运行时开销）、硬件感知（根据 SM 版本选择最优策略）、在多数场景下达到与手写 CUDA 相当的性能。

### 4.3 查看生成代码与调试

通过 `tilelang.compile` 导出 CUDA 源码或注册 `postproc_callback` 钩子函数，开发者可以下钻至指令层级进行性能瓶颈排查与逻辑验证。

```python
# 查看生成的 CUDA 源码
kernel = tilelang.compile(your_function, target="cuda")
print(kernel.get_kernel_source())

# 编译过程回调
from tilelang.engine.callback import register_cuda_postproc_callback

@register_cuda_postproc_callback
def debug_cuda_code(code, target):
    print("Final CUDA code:", code)
    return code
```

---

## 5. 实践建议

在生产环境中引入 TileLang 时，需严格评估其 JIT 编译延迟对系统的影响，并结合具体架构特性选择最优的算子下发与显存管理策略。

### 5.1 适合与不适合的场景

**适合 TileLang**：

- 你需要快速开发一个性能接近手写 CUDA 的 Kernel，但不想花几天调试 Bank Conflict 和寄存器 spill
- 你需要一个跨 NVIDIA/AMD 的算子，不想维护两套代码
- 你在做算子研发的快速迭代阶段，后期可替换为手写 CUDA

**不适合 TileLang（建议手写 CUDA/使用 Triton）**：

- 你需要极致性能的最后一两个百分点（如 cublas 竞品），必须控制每条指令
- 你的算法需要非标准 Warp 级原语或特殊硬件特性，而 TileLang 尚未暴露
- 你的工作流中 JIT 编译延迟不可接受（如高频推理服务的热更新）
- 你的团队已有成熟的手写 CUDA 资产，迁移成本高于收益

| 对比维度     | TileLang         | Triton            | 手写 CUDA         |
| ------------ | ---------------- | ----------------- | ----------------- |
| 性能天花板   | 接近手写         | 接近手写          | 最高              |
| 开发效率     | 高               | 高                | 低                |
| AMD GPU      | 支持 (HIP)       | 有限支持          | 不支持（需 ROCm） |
| 跨架构兼容   | 强（TIR 抽象层） | 中                | 弱                |
| JIT 启动延迟 | TVM 编译 ~秒级   | Triton 编译 ~秒级 | 无（AOT）         |
| 学习曲线     | 中（需理解 DSL） | 中                | 高                |

### 5.2 内存管理

TileLang 完全复用 PyTorch 的显存分配器，支持隐式张量创建与显式预分配复用，合理的张量生命周期管理可有效消除 CUDA OOM 与分配开销。

```python
# 自动管理（推荐）
c = kernel(a, b)       # 自动分配输出张量

# 手动管理（复用预分配张量）
c = torch.empty(M, N, device="cuda", dtype=torch.float16)
kernel(a, b, c)
```

**常见错误**：

| 错误               | 原因                | 解决                          |
| ------------------ | ------------------- | ----------------------------- |
| CUDA out of memory | 形状或分块过大      | 减小 `block_M/N/K` 或清理缓存 |
| 编译错误           | 参数类型/形状不匹配 | 检查 `@T.prim_func` 签名      |
| 运行时错误         | 张量不在正确设备上  | 确保 `.cuda()` 且 dtype 匹配  |

### 5.3 延伸阅读

- [SIMT vs Tile-Based：CUDA 编程范式的演进与对比](../02_cuda/04_simt_vs_tile_based.md) — 理解 Tile-Based 编程范式的设计哲学
- [GPU 编程导论](../02_cuda/01_gpu_programming_introduction.md) — CUDA Grid/Block/Warp/Thread 基础
- [TileLang 官方文档](https://tile-ai.github.io/tilelang/)
- [TileLang 示例仓库](https://github.com/tile-ai/tilelang/tree/main/examples)
