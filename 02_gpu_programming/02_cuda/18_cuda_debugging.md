# CUDA 调试实战——从 cuda-gdb 到 compute-sanitizer

你已经写了 17 篇 CUDA 代码——但总有那么一刻：kernel 跑完了，结果全 NaN。没有 error code，没有 crash，只有一片安静的 `nan`。或者更糟：有时正确有时错误，多跑几次才复现——典型的 race condition。

CUDA 的调试工具链不像 CPU 侧那么成熟——没有 `printf` 那么方便的断点内省（实际上有 `printf` 但 thread 间输出乱序）、没有 IDE 一键 debug。但一旦掌握 `cuda-gdb` + `compute-sanitizer` + 正确的错误处理模式，定位 GPU Bug 的效率不输 CPU。

本文覆盖三个工具和一个模式：`cuda-gdb`（断点调试）、`compute-sanitizer`（自动检测越界/race/未初始化）、`cudaError_t`（错误处理模式），以及常见 Bug 的排查流程。

> **前置要求**：你已写过 CUDA kernel，知道 Shared Memory、Warp、`__syncthreads` 的基本概念。
>
> **验证代码**：本文所有命令已在 A100 (CUDA 12.8) 上验证。配套验证程序：[18_debug_demo.cu](code/18_debug_demo.cu)——故意引入 5 种常见 Bug（SMEM 越界、Race、未初始化、NaN、Null Pointer），可用于练习 `compute-sanitizer` 的各种模式。

---

## 1. `cuda-gdb`——像 gdb 一样调试 GPU 代码

### 1.1 基本用法

`cuda-gdb` 是 NVIDIA 对 gdb 的扩展，支持在 GPU kernel 内设置断点、查看寄存器、检查 Shared Memory。

```bash
# 用 -G 编译（debug 模式——关闭优化，保留符号）
nvcc -G -g -o my_program my_kernel.cu

# 启动 cuda-gdb
cuda-gdb ./my_program
```

```gdb
# 在 kernel 内部设断点
(cuda-gdb) break my_kernel

# 条件断点——只在特定 thread 触发
(cuda-gdb) break my_kernel if threadIdx.x == 0 && blockIdx.x == 0

# 运行
(cuda-gdb) run

# 触发后——查看当前 warp、block 信息
(cuda-gdb) cuda thread
(cuda-gdb) cuda block

# 切换到特定线程
(cuda-gdb) cuda thread (0,0,0)  # threadIdx = (0,0,0)

# 查看寄存器
(cuda-gdb) info registers

# 查看 Shared Memory
(cuda-gdb) info cuda sharedmem

# 查看 Global Memory 中的变量
(cuda-gdb) p d_array[0]@16  # 打印 d_array 的前 16 个 float

# 单步（当前 warp 内所有活跃线程同步执行）
(cuda-gdb) stepi   # 单条 PTX/SASS 指令
(cuda-gdb) nexti   # 跳过函数调用
```

### 1.2 关键限制

- **`-G` 关闭了编译器优化**——debug 模式下代码行为可能与 release 不同（时序变化可能让 race condition 消失）
- **单步是 warp-level 的**——`stepi` 推进整个 warp 的 PC，而不是一个线程
- **不能单步切换线程**——断点触发时，warp 内所有活跃线程停在同一位置
- **调试信息有代价**——debug build 比 release 慢 10-100×

### 1.3 典型场景：NaN 定位

```gdb
(cuda-gdb) break my_kernel if threadIdx.x == 0
(cuda-gdb) run
# ... 断点命中 ...
(cuda-gdb) p sum
# $1 = 0.0  ← 初始值正常

(cuda-gdb) continue  # 继续到下次命中断点
# ... 几次迭代后 ...
(cuda-gdb) p sum
# $2 = nan  ← 从这里开始 sum 变成了 NaN

# 检查上一个操作的输入
(cuda-gdb) p a_val
# $3 = inf  ← 输入是 inf，导致 sum = 0 + inf → nan
```

**排查 NaN 的 SOP**：

1. 找到第一个变成 NaN 的变量
2. 向上追溯——哪个操作导致了 NaN？
3. 检查那个操作的输入——是否有 inf、-inf、或本身就是 NaN？

90% 的 NaN 来源：除以零（`x/0 → inf`）、`inf - inf → NaN`、`0 * inf → NaN`、未初始化数据。

### 1.4 检查 Shared Memory 内容

```gdb
(cuda-gdb) break my_kernel
(cuda-gdb) run
(cuda-gdb) info cuda sharedmem
# 打印当前 block 的 Shared Memory 内容（十六进制 + float 解释）
```

尤其在 Bank Conflict 排查时有用——可以看到 Shared Memory 中数据的实际布局。

---

## 2. `compute-sanitizer`——自动检测越界、Race、未初始化

`compute-sanitizer` 是 NVIDIA 的 GPU 内存错误检测工具——相当于 CPU 上的 Valgrind / AddressSanitizer。它不需要重新编译（!），支持 `-lineinfo` 的 release build 即可。

### 2.1 四种检测模式

```bash
# 模式 1: 越界（Out-of-bounds）—— 默认模式
compute-sanitizer ./my_program

# 模式 2: Race condition
compute-sanitizer --tool racecheck ./my_program

# 模式 3: 未初始化访问
compute-sanitizer --tool initcheck ./my_program

# 模式 4: 内存泄漏
compute-sanitizer --tool memcheck ./my_program
```

### 2.2 越界检测（默认）

最常用的模式。A100 (CUDA 12.8) 实测：默认模式**能检测 Global Memory 非法地址（如 Null Pointer deref），但对 Shared Memory 越界和 Global Memory 轻微越界的检测能力有限**。

```bash
nvcc -lineinfo -o my_program my_kernel.cu  # release + line info
compute-sanitizer ./my_program
```

A100 实测输出（Null Pointer deref）：

```text
========= COMPUTE-SANITIZER
========= Trace/breakpoint trap
=========     at null_ptr()+0x20 in 18_debug_demo.cu:52
=========     by thread (0,0,0) in block (0,0,0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: main [0x8ee6] in debug_demo
=========
========= Program hit cudaErrorLaunchFailure (error 719) due to
=========     "unspecified launch failure" on CUDA API call to
=========     cudaDeviceSynchronize.
```

> **重要发现**：A100 实测中，`compute-sanitizer` 默认模式**不会**报告 Shared Memory 的轻微越界（如 `sdata[62]` 在只分配 32 个元素的数组上）。这是因为 GPU 的 Shared Memory 没有 MMU 保护——越界写入会悄悄覆盖相邻内存（可能是其他 Shared Memory 变量或未使用区域），不触发硬件异常。因此**不能仅靠 compute-sanitizer 来保证 Shared Memory 越界安全**——代码审查和静态分析（检查所有 `threadIdx.x * stride + offset` 的上限）仍然必要。

Global Memory 越界同理——写到一个合法但非预期的地址（如 `d[tid + N]` 越过分配边界但仍在 GPU 可寻址范围内）可能不被报告。

排查 SOP：

1. `compute-sanitizer` 跑一次（默认模式）——能抓到非法地址和 severe OOB
2. 但不能 100% 依赖它——逐行检查所有 `threadIdx.x` 相关的索引计算，确认 `tid * stride + offset < ARRAY_SIZE`
3. 特别注意 `tid + s` 模式中的 `s` 最大值（Reduction kernel 中的常见 bug）
4. 怀疑 SMEM 越界时——在 cuda-gdb 中用 `info cuda sharedmem` 检查实际数据

### 2.3 Race Condition 检测

```bash
compute-sanitizer --tool racecheck ./my_program
```

检测条件：

- 两个线程访问同一 Shared Memory 或 Global Memory 地址
- 至少一个是写操作
- 两者之间没有 `__syncthreads()` 之类的同步

A100 实测输出（`race_condition` kernel，256 线程缺 `__syncthreads`）：

```text
========= COMPUTE-SANITIZER
========= Error: Race reported between Write access at
=========     race_condition(float *)+0xb0 in 18_debug_demo.cu:28
=========     and Read access at
=========     race_condition(float *)+0xc0 in 18_debug_demo.cu:30
=========     [1024 hazards]
=========
========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)
```

**典型 race 场景**：

```cuda
// RACE! 线程 A 写 sdata[0]，线程 B 在同一个 cycle 读 sdata[0]
// 没有 __syncthreads() 保护
sdata[tid] = input[tid];          // 所有线程并发写
float x = sdata[(tid + 1) % 32];  // ← RACE: 线程 0 读 sdata[1] 可能还未被线程 1 写入
```

修复：在共享内存写和读之间加 `__syncthreads()`。

### 2.4 未初始化访问检测

```bash
compute-sanitizer --tool initcheck ./my_program
```

检测任何对未初始化 Shared Memory 或 Local Memory 的读操作。这在高 Occupancy 场景下特别隐蔽——Shared Memory 可能残留前一个 block 的数据（不是清零的）。

### 2.5 性能代价

| 工具                             | 速度下降 | 何时用                                   |
| -------------------------------- | -------- | ---------------------------------------- |
| `compute-sanitizer` (默认，越界) | ~2-5×    | 每次提 PR 前跑一次                       |
| `--tool racecheck`               | ~5-20×   | 怀疑有 race 时、Shared Memory 代码变更后 |
| `--tool initcheck`               | ~10-50×  | 出现随机正确/错误时                      |
| `cuda-gdb` (-G debug build)      | ~10-100× | 精确定位 NaN、单步排查逻辑               |

---

## 3. `cudaError_t`——错误处理模式

### 3.1 同步 vs 异步 Error

CUDA 的错误分为同步和异步两类：

```cuda
// 同步错误 —— API 调用当场返回错误码
cudaError_t err = cudaMalloc(&d_ptr, size);
if (err != cudaSuccess) { /* 立即处理 */ }

// 异步错误 —— kernel launch 不返回 kernel 内部的错误
my_kernel<<<grid, block>>>(d_ptr, N);
cudaError_t err = cudaGetLastError();  // ← 只检查 launch 参数错误
if (err != cudaSuccess) { /* 非法 grid/block 大小、不存在的 kernel */ }

// kernel 内部的错误（如除零）——需要同步后才能捕获
cudaDeviceSynchronize();
err = cudaGetLastError();  // ← 现在才能捕获 kernel 内部错误
```

**关键规则**：`cudaGetLastError()` 在 device 同步之前**不能**返回 kernel 内部错误。必须 `cudaDeviceSynchronize()` 或 `cudaStreamSynchronize()` 之后再检查。

### 3.2 生产环境的错误检查宏

```cuda
// 同步 API 检查
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while(0)

// Kernel launch + 同步 + 错误检查
#define CUDA_LAUNCH(kernel, grid, block, ...) do {                 \
    kernel<<<grid, block>>>(__VA_ARGS__);                          \
    CUDA_CHECK(cudaGetLastError());                                \
    CUDA_CHECK(cudaDeviceSynchronize());                           \
} while(0)
```

### 3.3 常见错误码速查

| 错误码                           | 含义                      | 常见原因                          |
| -------------------------------- | ------------------------- | --------------------------------- |
| `cudaErrorInvalidConfiguration`  | 非法 grid/block 配置      | 每维线程数超过 maxThreadsPerBlock |
| `cudaErrorMemoryAllocation`      | OOM                       | 显存不足                          |
| `cudaErrorLaunchFailure`         | kernel 执行异常           | 除零、越界、非法指令              |
| `cudaErrorIllegalAddress`        | 非法内存地址              | 解引用 null 或越界 device pointer |
| `cudaErrorAssert`                | kernel 内 `assert()` 触发 | 调试模式下的断言失败              |
| `cudaErrorPeerAccessUnsupported` | P2P 不支持                | 未启用 peer access 或硬件不支持   |

---

## 4. 常见 Bug 排查 SOP

### 4.1 NaN 排查

```text
Step 1: cuda-gdb 断点到怀疑的 kernel，单步找到第一个 NaN
Step 2: 检查 NaN 来源:
  ├─ 除以零? → inf → 后续 op 产生 NaN
  ├─ inf - inf? → NaN
  ├─ 0 * inf? → NaN
  └─ 读未初始化数据? → 任意值 → 用 compute-sanitizer --tool initcheck
Step 3: 如果是除零——检查分母来源、加上 epsilon 保护
```

### 4.2 Hang（程序卡死）

```text
Step 1: nvidia-smi 看 GPU 状态
  → 有进程但 GPU 利用率为 0? → 可能是 kernel hang
Step 2: Ctrl+C 后 cuda-gdb attach
  → (cuda-gdb) info cuda threads → 看哪个 warp 停在哪个位置
  → 可能是无限循环或死锁（缺少 __syncthreads 导致 warp divergence 后无法收敛）
Step 3: 常见原因:
  ├─ 条件块内缺少 __syncthreads()
  ├─ __syncthreads() 在 if/else 中被跳过 → 一个 warp 等另一个
  └─ 无限循环（循环变量使用 threadIdx 计算但条件永真）
```

### 4.3 有时正确有时错误

这是一个典型的 **Race Condition**：

```text
Step 1: compute-sanitizer --tool racecheck → 大概率直接定位
Step 2: 如果 racecheck 没找到，手动检查:
  ├─ 所有 Shared Memory 写读之间有没有 __syncthreads()?
  ├─ Global Memory 是否有多个 block 写同一位置?
  └─ Atomic 操作是否正确使用?
Step 3: 如果只在 release 模式下出现(-O3):
  → 编译器的 aggressive 优化改变了指令顺序
  → 用 nvcc --ptx 对比 -G 和 -O3 的 PTX 差异
```

### 4.4 内存泄漏

```bash
compute-sanitizer --tool memcheck --leak-check full ./my_program
```

CUDA 内存泄漏与 CPU 内存泄漏行为相同——`cudaMalloc` 后没有配对的 `cudaFree`。`compute-sanitizer --tool memcheck` 在程序退出时报告所有未释放的 device 内存。

---

## 5. 总结——诊断工具速查

| 我想做什么             | 工具                                             | 命令                          |
| ---------------------- | ------------------------------------------------ | ----------------------------- |
| 断点进 kernel 查看变量 | `cuda-gdb`                                       | `nvcc -G -g`，然后 `cuda-gdb` |
| 检测越界访问           | `compute-sanitizer`                              | `compute-sanitizer ./prog`    |
| 检测 race condition    | `compute-sanitizer --tool racecheck`             |                               |
| 检测未初始化访问       | `compute-sanitizer --tool initcheck`             |                               |
| 检查 launch 参数错误   | `cudaGetLastError()` after launch                |                               |
| 捕获 kernel 内部错误   | `cudaDeviceSynchronize()` + `cudaGetLastError()` |                               |
| 查看时间线（宏观瓶颈） | `nsys` (Nsight Systems)                          | `nsys profile ./prog`         |
| 查看单 kernel 微观测   | `ncu` (Nsight Compute)                           | `ncu --section ...`           |

> `nsys` 是系统级分析工具，用于宏观时间线（CPU-GPU 交互、是否真的并发），`ncu` 是 `compute-sanitizer` 的兄弟工具——两者互补。前述系列文章中的"用 ncu 检查"参考 [#17 性能调优方法论](17_roofline_optimization.md) §2。

### 与其他文章的衔接

| 本文涉及的                  | 详见                                                                   |
| --------------------------- | ---------------------------------------------------------------------- |
| ncu 诊断 SOP                | [#17 性能调优方法论](17_roofline_optimization.md) §2                   |
| Shared Memory Race 场景     | [#13 Shared Memory & Bank Conflict](13_shared_memory_bank_conflict.md) |
| Warp Divergence 导致的 hang | [#14 Warp-level Programming](14_warp_level_programming.md) §1.2        |
| P2P 错误处理                | [#15 Multi-GPU](15_multi_gpu_programming.md)                           |
| NCCL 错误处理               | [#15 Multi-GPU](15_multi_gpu_programming.md) §5.4                      |
