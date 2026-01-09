# GPU-Virtual-Service 深度剖析：CUDA 劫持机制、全链路流程与 HAMi-Core 对比

## 1. 概述

在云原生环境中，GPU 资源的昂贵稀缺与任务负载的动态变化形成了鲜明矛盾。为了最大化硬件利用率，**GPU 虚拟化**技术应运而生。`GPU-Virtual-Service` 采用 **CUDA 劫持（CUDA Hijacking）** 技术，通过 Linux 的 `LD_PRELOAD` 机制，在用户态实现了对 NVIDIA GPU 资源的细粒度切分与隔离。

本文档将从代码实现层面深入剖析 `GPU-Virtual-Service` 的核心机制，涵盖：

- **API 拦截原理**：如何透明地接管 CUDA Driver API 调用。
- **资源控制算法**：基于文件锁的显存隔离与基于 PID 的算力压制策略。
- **全链路流程**：从 Kubernetes 资源申请到底层运行时拦截的完整生命周期。

此外，本文还将从架构设计、性能权衡及稳定性等维度，对 `GPU-Virtual-Service` 与社区主流方案 HAMi-Core 进行深入的横向评测，旨在拨开浮夸的宣传迷雾，为开发者和架构师提供一份详尽的技术参考。

---

## 2. 核心架构

`GPU-Virtual-Service`模块位于用户应用与 NVIDIA 驱动（CUDA Driver / `libcuda.so`）之间，充当了一个中间代理层。

- **Hook 层 (`cuda_hooks.cpp`)**: 定义了与 CUDA Driver API 签名一致的函数，用于拦截调用。
- **资源限制器 (`CudaResourceLimiter`)**: 单例控制器，协调显存和算力限制逻辑。
- **显存管理器 (`MemoryLimiter`)**: 负责显存申请的拦截、校验与配额管理。
- **算力限制器 (`GpuCoreLimiter`)**: 通过 PID 控制器动态调节内核发射频率，实现算力压制。
- **底层交互 (`GpuManager`)**: 封装 NVML 接口，查询物理 GPU 的真实状态。

---

## 3. 实现逻辑梳理

本章节将深入代码深处，逐行拆解 `GPU-Virtual-Service` 实现虚拟化的三大技术支柱：

1. **动态拦截技术**：阐述如何通过 `dlsym` 与 `RTLD_NEXT` 机制，在用户态透明地拦截并代理 CUDA 驱动 API 调用。
2. **无状态显存隔离**：解析如何基于 NVML 实时状态查询与文件锁同步机制，实现无侵入式的显存配额管理。
3. **PID 算力压制**：展示如何应用 `PID`（比例-积分-微分）控制算法，动态调节内核发射频率以实现精确的算力限制。

### 3.1 拦截机制

系统利用 `dlsym` 配合 `RTLD_NEXT` 标志位，在运行时动态解析并获取原始 CUDA 驱动库（`libcuda.so`）中函数的真实地址。同时，通过维护一个全局映射表（`g_hookedProc`），建立了拦截函数（Hook Function）与原始函数（Original Function）之间的双向绑定关系，确保调用链的正确转发。

- **代码位置**: `direct/cuda/src/hooks/cuda_hooks.cpp`

该文件维护了一个巨大的映射表 `g_hookedProc`，列出了所有被拦截的 CUDA 驱动 API。

```cpp
// direct/cuda/src/hooks/cuda_hooks.cpp

static std::unordered_map<void *, void*> g_hookedProc = {
  PROC_ADDR_PAIR(cuDriverGetVersion),
  PROC_ADDR_PAIR(cuInit),
  PROC_ADDR_PAIR(cuGetProcAddress), // 关键：拦截动态符号查找
  PROC_ADDR_PAIR(cuMemAlloc_v2),    // 关键：拦截显存分配
  PROC_ADDR_PAIR(cuMemAllocManaged),// 关键：拦截统一内存分配
  PROC_ADDR_PAIR(cuLaunchKernel),   // 关键：拦截内核启动
  PROC_ADDR_PAIR(cuCtxCreate_v2),   // 关键：拦截上下文创建
  // ... 共计 40+ 个关键函数
};
```

系统使用宏 `FUNC_HOOK_BEGIN` 来定义拦截逻辑。以 `cuDriverGetVersion` 为例：

```cpp
// direct/common/include/hook_helper.h
#define PROC_ADDR_PAIR(function) {dlsym(RTLD_NEXT, #function), reinterpret_cast<void *>(&function)}

// direct/cuda/src/hooks/cuda_hooks.cpp
CUresult FUNC_HOOK_BEGIN(cuDriverGetVersion, int *driverVersion)
  CudaResourceLimiter::Instance().Initialize(); // 初始化资源限制器
  return original(driverVersion);               // 调用原始驱动函数
FUNC_HOOK_END
```

### 3.2 显存虚拟化与隔离

显存隔离的防线建立在“事前准入机制”之上，即在任何显存分配请求真正提交给 GPU 驱动之前，必须先通过虚拟化层的配额校验。

- **代码位置**: `direct/common/src/memory_limiter.cpp`

当用户申请显存时，系统会调用 `GuardedMemoryCheck`。该函数首先获取文件锁，然后查询 NVML 获取当前 GPU 的真实显存使用量，最后判断是否超出配额。

```cpp
// direct/common/src/memory_limiter.cpp

MemoryLimiter::Guard MemoryLimiter::GuardedMemoryCheck(size_t requested)
{
    // 1. 获取文件锁，防止多进程并发导致的配额超发
    FileLock lock(LockPath(), LOCK_EX);
    // 2. 执行内存检查
    return {std::move(lock), MemoryCheck(requested)};
}

bool MemoryLimiter::MemoryCheck(size_t requested)
{
    size_t used;
    // 3. 通过 NVML 查询当前 GPU 已使用的显存 (Source of Truth)
    int ret = xpu_.MemoryUsed(used);
    if (ret) {
        return false;
    }

    size_t quota = config_.MemoryQuota();
    // 4. 检查：申请量 + 已用量 > 配额
    if (requested + used > quota) {
        log_err("out of memory, request {} B, used {} B, quota {} B",
            requested, used, quota);
        return false;
    }
    return true;
}
```

### 3.3 算力限制

算力限制采用了 **PID 反馈控制** 策略，通过在 Kernel Launch 阶段注入动态延迟来控制 GPU 时间片占用。

- **代码位置**: `direct/cuda/src/gpu_core_limiter.cpp`

PID 控制器根据 **实际利用率** 与 **目标配额** 的差值，动态计算需要休眠（Sleep）的微秒数。

```cpp
// direct/cuda/src/gpu_core_limiter.cpp

// 初始化阶段根据算力配额设置 PID 参数
int GpuCoreLimiter::Initialize()
{
    // ...
    if (upLimit <= BOUNDARY_LIMIT) {
        // 低算力场景：响应更激进
        GpuCoreLimiter::pidController_ = {10.5, 3.9, 1, 0, 0, 2}; // Kp=10.5, Ki=3.9, Kd=1
    } else {
        // 高算力场景：响应更平滑
        GpuCoreLimiter::pidController_ = {5.5, 0.76, 1, 0, 0, 2}; // Kp=5.5, Ki=0.76, Kd=1
    }
    // ...
}

long GpuCoreLimiter::PidController::CalculateDelay(int diff)
{
  // 标准 PID 公式：Delay = Kp * error + Ki * integral + Kd * derivative
  long delay = lround(kp * (diff - prevDiff1) + ki * diff + kd * (diff - coeffDouble * prevDiff1 + prevDiff2));
  prevDiff2 = prevDiff1;
  prevDiff1 = diff;
  return delay;
}

void GpuCoreLimiter::ComputingPowerLimiter()
{
  // 在每次 cuLaunchKernel 前调用
  int delay = GetDelay(gpu_.CurrentDevice());
  if (delay != 0) {
    // 注入延迟，让出 GPU 时间片
    std::this_thread::sleep_for(std::chrono::microseconds(delay));
  }
}
```

---

## 4. 用户视角：资源申请与全链路调用流程

本章将从使用者的角度出发，阐述如何通过 Kubernetes 申请虚拟化资源，并详细解析从资源申请到最终 CUDA API 被拦截执行的完整生命周期。

### 4.1 资源申请 (Resource Request)

用户通过 Kubernetes Pod YAML 的 `resources.limits` 字段声明所需的 vGPU 资源。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vgpu-demo
spec:
  containers:
    - name: cuda-container
      image: nvidia/cuda:11.0.3-base
      resources:
        limits:
          # 申请 1 个 vGPU
          huawei.com/vgpu-number: 1
          # 申请 4GB 显存 (单位: 1Gi 块)
          huawei.com/vgpu-memory.1Gi: 4
          # 申请 50% 算力
          huawei.com/vgpu-cores: 50
```

### 4.2 全链路调用流程

整个虚拟化过程可以分为三个阶段：**调度注入**、**环境初始化**、**运行时拦截**。

#### 4.2.1 Phase 1: 调度与配置注入

1. **用户提交**: 用户提交上述 Pod YAML。
2. **调度 (Volcano)**: Volcano 调度器识别 vGPU 资源请求，选择合适的节点，并将分配决策写入 Pod Annotation。
3. **Kubelet 调用**: 节点上的 Kubelet 调用 Device Plugin 的 `Allocate` 接口。
4. **配置生成 (Device Plugin)**:
   - Device Plugin 读取 Pod Annotation 中的分配信息。
   - 在宿主机生成配置文件 `vgpu.config`，内容包含显存配额 (`UsedMem`) 和算力配额 (`UsedCores`)。
   - 通过 `AllocateResponse` 将该配置文件挂载到容器内的 `/etc/xpu/vgpu.config`。
   - 设置容器环境变量 `LD_PRELOAD=/usr/local/vgpu/libcuda_hook.so` (或其他路径)。

#### 4.2.2 Phase 2: 初始化与加载

1. **容器启动**: 容器内的应用进程（如 PyTorch）启动。
2. **动态库注入**: 操作系统加载器（ld.so）根据 `LD_PRELOAD` 优先加载 `libcuda_hook.so`。
3. **Hook 初始化**:
   - 当应用首次调用 `cuInit` 或 `cuDriverGetVersion` 时，Hook 库触发初始化。
   - `ResourceConfig::Initialize` 读取 `/etc/xpu/vgpu.config`。
   - **配额加载**: 解析 `UsedMem` 为 `memory_` (显存上限)，解析 `UsedCores` 为 `computingPower_` (算力上限)。

#### 4.2.3 Phase 3: 运行时拦截 (Runtime Interception)

1. **API 调用**: 用户程序调用 `cudaMalloc` (最终调用 `cuMemAlloc_v2`)。
2. **拦截**: `cuda_hooks.cpp` 中的 `FUNC_HOOK_BEGIN(cuMemAlloc_v2)` 捕获调用。
3. **鉴权 (ResourceLimiter)**:
   - `GuardedMemoryCheck` 申请文件锁。
   - 查询 NVML 获取当前物理卡已用显存。
   - 计算 `当前已用 + 申请量` 是否大于 `memory_` 配额。
4. **执行**:
   - 如果通过检查，调用原始 `libcuda.so` 的 `cuMemAlloc_v2`，返回成功。
   - 如果未通过，返回 `CUDA_ERROR_OUT_OF_MEMORY`。

### 4.3 关键配置传递机制

系统并不直接依赖环境变量传递配额，而是通过**挂载配置文件**的方式。

- **配置文件路径**: `/etc/xpu/vgpu.config` (容器内)
- **文件格式**:

  ```text
  UsedMem:4096    # 对应 vgpu-memory.1Gi * 1024 (MB)
  UsedCores:50    # 对应 vgpu-cores
  ```

- **代码对应**: `ResourceConfig::LoadVxpuConfig` 负责解析此文件，将其转化为内存中的配额限制。

---

## 5. 与 HAMi-Core 的代码级对比分析

`HAMi-Core` (Heterogeneous AI Computing Middleware) 是社区中广泛使用的另一款开源 GPU 虚拟化核心库。尽管两者在设计目标上殊途同归，但在底层实现路径上却存在本质差异。

### 5.1 拦截策略

GPU 虚拟化的核心在于对 CUDA Driver API 的拦截与重定向。通过 Linux 的 `LD_PRELOAD` 机制，可以在应用程序加载官方 `libcuda.so` 之前，优先加载自定义的虚拟化库。该库劫持了关键的 CUDA 函数（如 `cuInit`, `cuMemAlloc`, `cuLaunchKernel` 等），在这些函数中插入资源检查、配额限制和虚拟化逻辑，然后再调用原生的 CUDA 函数。这种方式对上层应用透明，无需修改应用代码即可实现 GPU 资源的细粒度控制。

在具体实现策略上，两者略有不同：

- **GPU-Virtual-Service**: 采用 **白名单映射** 方式。显式定义了 `g_hookedProc` 映射表，只有表中的函数才会被拦截。这使得代码结构非常清晰，易于维护。
- **HAMi-Core**: 采用了更为底层的 **`dlsym` 劫持** 策略。它拦截了 `dlsym` 函数本身，当应用尝试查找任何 CUDA 符号时，它都会介入并返回自己的 Wrapper 函数。

以下是 **HAMi-Core 代码片段 (`src/libvgpu.c`)**:

```c
// 拦截 dlsym，这是更底层的劫持
FUNC_ATTR_VISIBLE void* dlsym(void* handle, const char* symbol) {
    // ... 初始化 ...
    if (handle == RTLD_NEXT) {
        // 获取真实的符号地址
        void *h = real_dlsym(RTLD_NEXT,symbol);
        // 检查是否已经在映射表中，如果是则返回 wrapper
        // ...
        return h;
    }
    // ...
}
```

### 5.2 显存记账与同步

在多容器共享同一块物理 GPU 的场景下，如何准确追踪每个容器已使用的显存，并确保多个进程并发申请显存时的数据一致性，是虚拟化方案面临的最大挑战。**显存记账**决定了系统如何感知显存用量，而**同步机制**则确保了并发操作的原子性。

在这方面，两种方案采用了截然不同的设计哲学，这也是它们最大的区别所在。

- **GPU-Virtual-Service**: **无状态 (Stateless)** 设计。

  - **记账**: 每次申请都信任 NVML (`nvmlDeviceGetComputeRunningProcesses`) 返回的物理状态。
  - **同步**: 使用文件锁 (`flock`) 串行化检查过程。
  - **优点**: 极其稳定，规避了进程异常退出导致的“僵尸记录”问题。
  - **缺点**: NVML 调用有一定开销。

- **HAMi-Core**: **强状态 (Stateful)** 设计。
  - **记账**: 在 **共享内存 (Shared Memory)** 中维护一张全局的显存使用表。每次申请显存时，只是更新内存中的计数器，不一定查询 NVML。
  - **同步**: 使用信号量 (`sem_timedwait`) 保护共享内存。
  - **优点**: 性能极高，几乎没有系统调用开销。
  - **缺点**: 复杂度高。进程异常崩溃（如 Segfault/OOM）时，共享内存中的状态无法自动回收，需依赖复杂的 **Active OOM Killer** 或 **Watchdog** 机制清理脏数据，否则将引发显存永久泄露（Ghost Usage）。

以下是 **HAMi-Core 代码片段 (`src/allocator/allocator.c` & `src/multiprocess/multiprocess_memory_limit.c`)**:

```c
// 纯软件层面的检查，依赖 limit 和 _usage 变量
int oom_check(const int dev, size_t addon) {
    // ...
    uint64_t limit = get_current_device_memory_limit(d);
    size_t _usage = get_gpu_memory_usage(d); // 从共享内存读取

    if (new_allocated > limit) {
        // 触发 OOM，并尝试清理无效的进程槽位 (clear_proc_slot)
        if (clear_proc_slot_nolock(1) > 0)
            return oom_check(dev,addon);
        return 1;
    }
    return 0;
}
```

### 5.3 算力限制算法

**GPU-Virtual-Service 使用了 PID 控制算法来动态调整任务提交频率，以精确控制 GPU 利用率。**

PID（比例-积分-微分）控制器是一种广泛应用于工业控制系统的反馈回路机制。它通过计算设定值（目标 GPU 利用率）与测量值（当前 GPU 利用率）之间的误差，并利用误差的比例（P）、积分（I）、微分（D）三个分量来调整输出（注入的延迟时间）。

- **比例（P）**：反映当前误差的大小，误差越大，调节力度越大。
- **积分（I）**：反映过去误差的累积，用于消除稳态误差。
- **微分（D）**：反映误差的变化趋势，用于预测未来变化，减少超调。

在本项目中，PID 控制器根据 GPU 利用率的偏差，动态计算出需要在 `cuLaunchKernel` 前插入的休眠时间，从而精确控制任务提交频率。相比简单的固定延迟，PID 能更平滑地逼近目标利用率，减少震荡。

**HAMi-Core 使用了自适应令牌桶算法来动态限制任务提交频率，以精确控制 GPU 利用率。**

HAMi-Core 采用的是**自适应令牌桶（Adaptive Token Bucket）**算法。它维护一个全局的令牌桶（`g_cur_cuda_cores`），每次 Kernel 启动时消耗一定数量的令牌（通常与 Kernel 规模相关）。

以下是 **HAMi-Core 代码片段 (`src/multiprocess/multiprocess_utilization_watcher.c`)**:

```c
// src/multiprocess/multiprocess_utilization_watcher.c

// 1. 全局令牌桶状态
static volatile long g_cur_cuda_cores = 0;
static volatile long g_total_cuda_cores = 0;

// 2. 核心拦截逻辑：Kernel 启动前消耗令牌
void rate_limiter(int grids, int blocks) {
  long before_cuda_cores = 0;
  long after_cuda_cores = 0;
  long kernel_size = grids;

  // ... (省略部分检查代码)

  // 循环尝试扣减令牌
  do {
CHECK:
    before_cuda_cores = g_cur_cuda_cores;
    // 如果令牌不足，进入休眠等待
    if (before_cuda_cores < 0) {
      nanosleep(&g_cycle, NULL);
      goto CHECK;
    }
    after_cuda_cores = before_cuda_cores - kernel_size;
  } while (!CAS(&g_cur_cuda_cores, before_cuda_cores, after_cuda_cores));
}

// 3. 令牌更新原子操作
static void change_token(long delta) {
  int cuda_cores_before = 0, cuda_cores_after = 0;

  do {
    cuda_cores_before = g_cur_cuda_cores;
    cuda_cores_after = cuda_cores_before + delta;

    if (cuda_cores_after > g_total_cuda_cores) {
      cuda_cores_after = g_total_cuda_cores;
    }
  } while (!CAS(&g_cur_cuda_cores, cuda_cores_before, cuda_cores_after));
}

// 4. 计算令牌调整量（基于利用率偏差）
long delta(int up_limit, int user_current, long share) {
  // 计算利用率偏差
  int utilization_diff =
      abs(up_limit - user_current) < 5 ? 5 : abs(up_limit - user_current);

  // 计算增量：基于 SM 数量、每 SM 线程数和利用率偏差
  long increment =
      (long)g_sm_num * (long)g_sm_num * (long)g_max_thread_per_sm * (long)utilization_diff / 2560;

  /* Accelerate cuda cores allocation when utilization vary widely */
  if (utilization_diff > up_limit / 2) {
    increment = increment * utilization_diff * 2 / (up_limit + 1);
  }

  // 根据当前利用率与目标上限的关系，增加或减少 share
  if (user_current <= up_limit) {
    share = (share + increment) > g_total_cuda_cores ? g_total_cuda_cores
                                                   : (share + increment);
  } else {
    share = (share - increment) < 0 ? 0 : (share - increment);
  }

  return share;
}

// 5. 后台监控线程：周期性调整令牌
void* utilization_watcher() {
    // ... (初始化)
    int upper_limit = get_current_device_sm_limit(0);

    while (1){
        nanosleep(&g_wait, NULL); // 周期性休眠

        // ... (PID 检查等)

        // 获取当前 GPU 利用率
        get_used_gpu_utilization(userutil,&sysprocnum);

        // 如果利用率在有效范围内，计算并更新令牌
        if ((userutil[0]<=100) && (userutil[0]>=0)){
          share = delta(upper_limit, userutil[0], share);
          change_token(share);
        }
    }
}
```

后台线程定期通过 NVML 获取实际 GPU 利用率，并根据利用率与配额的偏差，动态调整令牌的补充速率（`share`）。如果利用率低于配额，增加令牌投放；反之则减少。这种方式通过反馈机制动态平衡了配额限制与突发性能需求。

**对比总结**：

| **特性**       | **GPU-Virtual-Service (PID)** | **HAMi-Core (自适应令牌桶)** |
| :------------- | :---------------------------- | :--------------------------- |
| **控制机制**   | 计算延迟时间 (Sleep Time)     | 消耗令牌 (Token Consumption) |
| **反馈调节**   | PID 公式调节延迟              | 动态调整令牌发放速率         |
| **平滑性**     | 较好，适合连续控制            | 较好，支持突发 (Burst)       |
| **实现复杂度** | 中等 (需调参 Kp, Ki, Kd)      | 中等 (需调整增量逻辑)        |

---

## 6. 总结与建议

通过对 GPU-Virtual-Service 源码的深入剖析以及与业界主流方案 HAMi-Core 的横向对比，我们可以清晰地看到不同技术路线的权衡。GPU-Virtual-Service 选择了稳定性优先的无状态设计与精确的 PID 算力控制，而 HAMi-Core 则追求极致性能与更广泛的兼容性。

| 特性           | GPU-Virtual-Service              | HAMi-Core                                      |
| :------------- | :------------------------------- | :--------------------------------------------- |
| **架构风格**   | C++ OOP，模块化，依赖 NVML       | C Procedural，底层黑客风格，依赖 Shared Memory |
| **稳定性**     | **高** (NVML 兜底，无脏数据风险) | **中** (需处理共享内存一致性问题)              |
| **性能**       | **中** (NVML 调用开销)           | **高** (纯内存操作)                            |
| **代码可读性** | **优** (逻辑分离，现代 C++)      | **一般** (逻辑耦合，大量底层操作)              |

---
