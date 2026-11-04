# CUDA 调试实战——从 cuda-gdb 到 compute-sanitizer

总有那么一刻：kernel 跑完了，结果全 NaN。没有 error code，没有 crash，只有一片安静的 `nan`。或者更糟：有时正确有时错误，多跑几次才复现——典型的 race condition。

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

# 查看 Shared Memory 变量——直接按变量名 print
(cuda-gdb) print sdata@32      # 打印 sdata 的前 32 个元素

# 查看 Global Memory 中的变量
(cuda-gdb) p d_array[0]@16  # 打印 d_array 的前 16 个 float

# 单步（当前 warp 内所有活跃线程同步执行）
(cuda-gdb) stepi   # 单条 PTX/SASS 指令
(cuda-gdb) nexti   # 跳过函数调用
```

### 1.2 关键限制

- `-G` 关闭了编译器优化：debug 模式下代码行为可能与 release 不同（时序变化可能让 race condition 消失）
- 单步是 warp-level 的：`stepi` 推进整个 warp 的 PC，而不是一个线程
- 不能单步切换线程：断点触发时，warp 内所有活跃线程停在同一位置
- 调试信息有代价：debug build 比 release 慢 10-100×

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
(cuda-gdb) print sdata@32
# 按变量名直接打印 Shared Memory 数组内容
(cuda-gdb) info cuda threads
# 查看各 warp/thread 的执行状态（cuda-gdb 的 cuda 子命令可用 info cuda 查看）
```

尤其在 Bank Conflict 排查时有用——可以看到 Shared Memory 中数据的实际布局。

---

## 2. `compute-sanitizer`——自动检测越界、Race、未初始化

`compute-sanitizer` 是 NVIDIA 的 GPU 内存错误检测工具——相当于 CPU 上的 Valgrind / AddressSanitizer。无需重编即可运行，带 `-lineinfo` 的 release build 即可拿到源码级回溯。

### 2.1 四个检测工具

```bash
# 工具 1: memcheck —— 内存错误（越界/非法地址/对齐），默认工具
compute-sanitizer ./my_program                 # 不带 --tool 即 memcheck
compute-sanitizer --tool memcheck --leak-check full ./my_program

# 工具 2: racecheck —— Shared Memory race condition
compute-sanitizer --tool racecheck ./my_program

# 工具 3: initcheck —— 未初始化读
compute-sanitizer --tool initcheck ./my_program

# 工具 4: synccheck —— CUDA API 同步用法错误
compute-sanitizer --tool synccheck ./my_program
```

### 2.2 越界检测（memcheck，默认）

最常用的工具。A100 (CUDA 12.8) 实测：能检测 Global Memory 非法地址（如 Null Pointer deref）。

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

> **边界条件**：memcheck 对 Shared Memory 的标准报错是 "Invalid __shared__ write ... out of bounds"，检测的是访问**越过 kernel 的 Shared Memory 分配窗口**的情况。实测中 `sdata[62]`（数组声明 32 个元素，但 kernel 的动态 SMEM 窗口更大）这类仍在分配窗口内的数组级越界可能不报——越界覆盖的是本 kernel 自己的其他 SMEM 空间。所以 Shared Memory 的数组级越界要靠代码审查（检查所有 `threadIdx.x * stride + offset` 的上限），不能只靠 sanitizer。

Global Memory 越界同理——越出分配且落在未映射地址会被报；写进恰好合法的其他分配则很难区分。

排查 SOP：

1. `compute-sanitizer` 跑一次（默认 memcheck）——能抓到非法地址和越过分配窗口的 OOB
2. 但不能 100% 依赖它——逐行检查所有 `threadIdx.x` 相关的索引计算，确认 `tid * stride + offset < ARRAY_SIZE`
3. 特别注意 `tid + s` 模式中的 `s` 最大值（Reduction kernel 中的常见 bug）
4. 怀疑 SMEM 越界时——在 cuda-gdb 中 `print sdata@N` 检查实际数据

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

检测对未初始化 device（global）内存的读。注意 Shared Memory 不在 initcheck 的覆盖范围内——SMEM 残留前一个 block 数据的问题要靠显式初始化解决。

### 2.5 性能代价

| 工具                             | 速度下降（经验值，随驱动/负载浮动） | 何时用                                   |
| -------------------------------- | ------------------------------------ | ---------------------------------------- |
| memcheck（默认）                 | ~2-5×                                | 每次提 PR 前跑一次                       |
| `--tool racecheck`               | ~5-20×                               | 怀疑有 race 时、Shared Memory 代码变更后 |
| `--tool initcheck`               | ~10-50×                              | 出现随机正确/错误时                      |
| `cuda-gdb` (-G debug build)      | ~10-100×                             | 精确定位 NaN、单步排查逻辑               |

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
| `cudaErrorLaunchFailure`         | kernel 执行异常           | 整型除零、非法地址、非法指令（浮点除零得 inf/NaN，不报错） |
| `cudaErrorIllegalAddress`        | 非法内存地址              | 解引用 null 或越界 device pointer                          |
| `cudaErrorAssert`                | kernel 内 `assert()` 触发 | 调试模式下的断言失败                                       |
| `cudaErrorPeerAccessUnsupported` | 硬件不支持 P2P            | 在不支持 P2P 的硬件上调 enable；未启用 peer access 的拷贝不返回此码（回退 host staging，见 [#15](15_multi_gpu_programming.md)） |

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
  → 用 cuobjdump -sass 对比 -G 与 -O3 的 SASS 差异更直接（-G 主要关后端优化）
```

### 4.4 内存泄漏

```bash
compute-sanitizer --tool memcheck --leak-check full ./my_program
```

CUDA 内存泄漏与 CPU 内存泄漏行为相同——`cudaMalloc` 后没有配对的 `cudaFree`。`compute-sanitizer --tool memcheck` 在程序退出时报告所有未释放的 device 内存。

---

## 5. 总结——诊断工具速查

| 我想做什么             | 工具                                             | 命令                                   |
| ---------------------- | ------------------------------------------------ | -------------------------------------- |
| 断点进 kernel 查看变量 | `cuda-gdb`                                       | `nvcc -G -g`，然后 `cuda-gdb`          |
| 检测越界访问           | memcheck（默认）                                 | `compute-sanitizer ./prog`             |
| 检测 race condition    | `compute-sanitizer --tool racecheck`             | `compute-sanitizer --tool racecheck .` |
| 检测未初始化访问       | `compute-sanitizer --tool initcheck`             | `compute-sanitizer --tool initcheck .` |
| 检查 launch 参数错误   | `cudaGetLastError()` after launch                | 见 §3.1                                |
| 捕获 kernel 内部错误   | `cudaDeviceSynchronize()` + `cudaGetLastError()` | 见 §3.1                                |
| 查看时间线（宏观瓶颈） | `nsys` (Nsight Systems)                          | `nsys profile ./prog`                  |
| 查看单 kernel 微观测   | `ncu` (Nsight Compute)                           | `ncu --section ...`                    |

> `nsys`/`ncu` 管性能，`compute-sanitizer` 管正确性，三类工具互补。前述系列文章中的「用 ncu 检查」参考 [#17 性能调优方法论](17_roofline_optimization.md) §2。

### 与其他文章的衔接

| 本文涉及的                  | 详见                                                                   |
| --------------------------- | ---------------------------------------------------------------------- |
| ncu 诊断 SOP                | [#17 性能调优方法论](17_roofline_optimization.md) §2                   |
| Shared Memory Race 场景     | [#13 Shared Memory & Bank Conflict](13_shared_memory_bank_conflict.md) |
| Warp Divergence 导致的 hang | [#14 Warp-level Programming](14_warp_level_programming.md) §1.2        |
| P2P 错误处理                | [#15 Multi-GPU](15_multi_gpu_programming.md)                           |
| NCCL 错误处理               | [#15 Multi-GPU](15_multi_gpu_programming.md) §5.4                      |
