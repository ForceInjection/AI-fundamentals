# GPU 内存管理——从推理工程师的日常问题出发

本文写给**后端和推理工程师**——你已经会用 vLLM 部署模型，但 GPU 内存对你来说是个黑盒：`cudaMalloc` 为什么在 `nvidia-smi` 显示有空闲时还报 OOM？pinned memory 为什么能让传输快一倍？NVLink 和 PCIe 到底差多少？GDS 绕过 CPU 的原理是什么？

这些问题有一个共同点：答案不在 CUDA API 手册里，而在**操作系统和硬件层面的内存管理机制**中。幸运的是，CPU (Linux) 和 GPU (CUDA) 在这些机制上有着惊人的对称性——虚拟内存、缺页中断、DMA、大页、NUMA 拓扑——你只需要一个翻译层，把 Linux 的知识映射到 GPU 世界。

本文就是这个翻译层。每一节从一个你在部署中实际遇到的困惑出发，先用 Linux 的对应概念建立直觉，再展开 GPU 侧的原理，最后回到"这对我意味着什么"。建议按顺序阅读——后面章节的概念建立在前面的基础上，但也可以直接跳到感兴趣的问题。

> 不需要 CUDA 编程背景。只要你理解 Linux 的基本概念（虚拟内存、`malloc`、`/proc`），就能跟上。

> **可交互概念图**：本文配有可交互的 SVG 概念图，覆盖 6 个图层（物理拓扑 / 内存层级 / DMA 路径 / MMU 页表 / 碎片化 / 跨进程共享），建议在阅读对应章节时打开对照查看：[gpu-memory-visual.html](gpu-memory-visual.html)。

---

## 先理解一个基础概念：虚拟内存 vs 物理内存

无论 CPU 还是 GPU，程序看到的"内存地址"都不是真实的物理地址。操作系统（或 GPU 驱动）维护一张**页表**，把程序使用的虚拟地址映射到物理内存的某个位置。这个映射工作由 **MMU**（Memory Management Unit，内存管理单元）完成。

```text
程序:  "我要读写地址 0x7f000000"
          ↓
MMU:   查页表 → 0x7f000000 映射到物理地址 0x12345000
          ↓
硬件:   访问物理内存 0x12345000
```

有了这个基础，后续所有概念——碎片化、DMA 需要物理地址、页锁定、大页——都围绕"虚拟地址如何映射到物理地址"展开。

---

## 1. 加载模型——"70GB 的模型怎么放进一张卡？"

你在 8×H100 上跑 vLLM，模型权重 70GB。启动后，`RuntimeError: CUDA out of memory`——但 `nvidia-smi` 的 `memory.free` 明明还有 10GB。

为什么？

### 1.1 GPU 显存不是一整块连续区域

CPU 程序调用 `malloc(20GB)` 时，内核只需要在虚拟地址空间中找到一段 20GB 的空闲范围，然后按需分配物理页——**物理上可以不连续**（一个 4KB 的页可能在物理地址 0x1000，下一个页在 0xFFFF000，MMU 负责把它们拼接成程序看到的连续空间）。这就是虚拟内存的精髓：程序看到的是连续的，物理上可以是碎片化的。

GPU 同样有 MMU 和页表。但 `cudaMalloc` 在处理大块分配时，需要建立**连续**的虚拟地址映射，而虚拟地址空间也可能被碎片化——先分配 50GB、再分配 1GB、再释放 1GB、再分配 20GB——这个 20GB 的分配请求可能在虚拟地址空间中找不到连续的 20GB 区域，尽管物理显存还有空闲。`cudaMalloc` 返回的不是简单的指针偏移——CUDA 驱动维护一个虚拟地址空间映射表，每次分配需要在 GPU MMU（Memory Management Unit）中建立页表条目。大块分配成功后，小块释放产生的孔洞无法被后续大块分配复用。

```bash
# CPU 端——查看进程真实内存占用
cat /proc/<pid>/smaps | grep -E "Rss|Pss"   # RSS: 实际占用的物理页

# GPU 端——查看显存碎片化程度
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

CPU 类比：Linux 的 buddy allocator 也有碎片化问题，但可以通过 `vm.compaction` 和 THP (Transparent Huge Pages) 缓解。GPU 没有自动碎片整理。

**你可以做什么**：

- `cudaMemGetInfo(&free, &total)` 拿到的 `free` 是**所有空闲字节的总和**——不代表有一块这么大的连续区域
- 分配 70GB 模型时，先 `cudaMalloc` 最大的权重 tensor，而不是先分配小 buffer——好的内存布局减少碎片
- GPU_MEM_UTIL 设得越低，vLLM 留给 KV cache 的空间越小，但也越容易出现碎片导致的 OOM

```text
CPU 空闲                             GPU "空闲"（碎片化后）
┌──────────────────────┐              ┌──────────┬───────┬──────────────┐
│                      │              │ 已用 20GB │ 10GB  │  已用 50GB   │
│   全部空闲 30GB       │              │          │ 空闲  │              │
│                      │              └──────────┴───────┴──────────────┘
└──────────────────────┘              这 10GB 被夹在两个块中间，无法做 20GB
malloc(20GB) → OK                     的分配。cudaMalloc(20GB) → OOM
                                      尽管 cudaMemGetInfo 报告 free=10GB
```

---

## 2. 数据传输——"为什么 pageable 不随 PCIe 升级而变快？"

你在两台机器上测试 CPU→GPU 传输。A100（PCIe Gen4 x16，AMD EPYC）上 pageable ~22 GB/s，pinned ~26 GB/s——差别只有 1.2 倍。同样的程序在 H100（PCIe Gen5 x16，Intel Xeon）上，pageable 还是 ~22 GB/s，pinned 却飙到 ~55 GB/s——2.5 倍的差距。

为什么 pageable 不随 PCIe 升级而变快？为什么 A100 上的 pinned 提升这么小，H100 上却这么大？

### 2.1 先理解 DMA：为什么 GPU 不通过 CPU 搬运数据

把数据从 CPU 内存搬到 GPU 显存，最原始的办法是 CPU 逐字节读、逐字节写——但没人这么做，因为太慢了。实际上用的是 **DMA**（Direct Memory Access，直接内存访问）：一个专门的硬件引擎（GPU 的 copy engine）自己从系统内存读取数据，自己写到 GPU 显存，**CPU 全程不参与数据搬运**。

但这带来了一个问题：DMA 引擎是硬件，它不经过 MMU，不懂虚拟地址。它只认**物理地址**。

### 2.2 CPU 不知道自己分配的物理内存在哪

当你调用 `malloc` 时，内核给你的是**虚拟地址**。背后的物理页可能在 swap 里，可能在 NUMA node 1 上，可能被压缩过。CPU MMU 处理这些细节，但 GPU DMA 引擎**不走 MMU**——它需要一个可以直接使用的物理地址。

```text
pageable 带宽 = 传输量 / (DMA 时间 + CPU 锁定+页表遍历时间)
                              ↑                    ↑
                       取决于 PCIe 代际        取决于 CPU 架构，与 PCIe 无关

cudaMallocHost（Pinned）：
  → 跳过页表遍历和锁定，DMA 引擎直传，带宽 = PCIe 理论 × 效率
```

实测拆解（传输 2GB，单次 `cudaMemcpy`）：

```text
                │  A100 Gen4 (AMD EPYC)  │  H100 Gen5 (Intel Xeon)
────────────────┼────────────────────────┼──────────────────────────
DMA 时间         │  2GB ÷ 26GB/s = 77ms   │  2GB ÷ 55GB/s = 36ms
CPU 开销         │  ~14ms (AMD-Vi IOMMU)  │  ~56ms (Intel VT-d)
单次总耗时       │  77 + 14 = 91ms        │  36 + 56 = 92ms
pageable 带宽    │  2GB ÷ 91ms ≈ 22 GB/s  │  2GB ÷ 92ms ≈ 22 GB/s
pinned 带宽      │  ~26 GB/s              │  ~55 GB/s
pinned/pageable  │  1.2×                  │  2.5×
```

两台的 pageable 带宽碰巧接近，但原因完全不同：AMD 的 CPU 页锁定效率高（IOMMU 硬件翻译，开销仅 14ms），但 PCIe Gen4 的 DMA 慢；Intel 的 CPU 页锁定开销大（56ms），但 DMA 快（Gen5）。两者总耗时恰好接近，导致 pageable 都落在 ~22 GB/s。

**关键结论：pageable 的瓶颈是 CPU，不是 PCIe。** pageable 带宽不是固定值——取决于你的 CPU 架构和 IOMMU 实现。但无论 CPU 多快，pageable 永远存在一笔 CPU 开销，而 pinned 直接跳过它。随着 PCIe 代际演进，DMA 时间越来越短，CPU 开销占比越来越大，pinned 相对于 pageable 的优势也会持续扩大。

CPU 类比：`mlock(ptr, size)` 锁定页面防止换出，但 `mlock` **不保证物理连续**——它只防止 swap。DMA 引擎需要的是**物理上可被 DMA 访问**的内存，这是 `cudaMallocHost` 额外保证的——内核在分配时就确保页面不会被移动或压缩。

```c
// pageable（malloc）：CPU 页表遍历 + DMA。带宽取决于 CPU 速度 + PCIe 代际
float *cpu_buf = malloc(N * sizeof(float));
cudaMemcpy(gpu_buf, cpu_buf, N * sizeof(float), cudaMemcpyHostToDevice);

// pinned（cudaMallocHost）：直传，带宽 = PCIe 理论 × 效率
//   Gen4 x16: ~26 GB/s  |  Gen5 x16: ~55 GB/s
float *pinned_buf;
cudaMallocHost(&pinned_buf, N * sizeof(float));  // 页锁定
cudaMemcpy(gpu_buf, pinned_buf, N * sizeof(float), cudaMemcpyHostToDevice);
cudaFreeHost(pinned_buf);
```

**代价**：Pinned memory 吃的是系统 RAM——分配太多会拖慢整个系统（其他进程的可用物理内存减少，内核开始换出）。生产环境建议 pinned 内存不超过总 RAM 的 10-20%。

---

## 3. 多 GPU——"NVLink 和 PCIe 到底差多少？"

你有 8 张 H100，做 TP=4 推理。GPU 0 算完一个 layer 后，需要把结果传给 GPU 1。传输走 NVLink、PCIe P2P、还是经由 CPU？差别有多大？

```bash
nvidia-smi topo -m     # 看 GPU 间连接类型
nvidia-smi topo -p2p r # 看 P2P 是否可用
```

输出示例：

```text
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
 GPU0    X      NV18    NV18    NV18    NV18    SYS     SYS     SYS
 GPU1   NV18     X      NV18    NV18    NV18    SYS     SYS     SYS
 GPU2   NV18    NV18     X      NV18    SYS     NV18    SYS     SYS
 GPU3   NV18    NV18    NV18     X      SYS     SYS     NV18    SYS
 GPU4   NV18    NV18    SYS     SYS      X      NV18    NV18    NV18
 GPU5   SYS     SYS     NV18    SYS     NV18     X      NV18    NV18
 GPU6   SYS     SYS     SYS     NV18    NV18    NV18     X      NV18
 GPU7   SYS     SYS     SYS     SYS     NV18    NV18    NV18     X
```

- `NV18`：NVLink 直连——H100 有 **18 条 NVLink 4.0 link**（每条 link = 4 对差分信号 × 双向，50 GB/s 双向），**900 GB/s 双向总量（单向 450 GB/s）**。H100 配 4 个第三代 NVSwitch，GPU 0-3 在一个 NVSwitch 域内全互联，GPU 4-7 在另一个
- `SYS`：需经过 CPU 中转——GPU→CPU→GPU 两次 DMA 串行，端到端有效带宽 ~13 GB/s (Gen4) / ~27 GB/s (Gen5)

CPU 类比：这和 NUMA 拓扑完全对应。Linux 用 `numactl --hardware` 查看 CPU socket 间的互联，GPU 用 `nvidia-smi topo -m`。

```text
CPU (NUMA)                           GPU (Multi-GPU)
────────                              ────────
Node 0 ──UPI── Node 1          GPU 0 ──NVLink── GPU 1
  │                │              │                  │
  └── 共享内存控制器 ──┘          └── PCIe Switch ────┘
```

**cudaMemcpyPeer**：GPU 间直接拷贝。不同路径的延迟差异巨大：

| 路径        | 延迟                  | 带宽                              |
| ----------- | --------------------- | --------------------------------- |
| NVLink 直连 | ~1 μs（类比 CPU UPI） | 450 GB/s 单向                     |
| PCIe P2P    | ~5 μs                 | ~32 GB/s (Gen4)                   |
| 经 CPU 中转 | ~20 μs                | ~13 GB/s (Gen4) / ~27 GB/s (Gen5) |

```cuda
// NVLink 直连：~450 GB/s 单向（900 GB/s 双向）
cudaMemcpyPeer(dst_gpu_ptr, dst_device, src_gpu_ptr, src_device, size);

// 经过 CPU：~26 GB/s (Gen4 pinned, 单跳；串行后 ~13 GB/s)
cudaMemcpy(cpu_buf, src_gpu_ptr, size, cudaMemcpyDeviceToHost);
cudaMemcpy(dst_gpu_ptr, cpu_buf, size, cudaMemcpyHostToDevice);
```

经 CPU 中转的实测与理论对比：

|                  |              Gen4 |              Gen5 |
| ---------------- | ----------------: | ----------------: |
| 单跳理论最大     |           32 GB/s |           64 GB/s |
| 单跳 pinned 实测 |     26 GB/s (81%) |     55 GB/s (86%) |
| 两跳串行理论最大 |           16 GB/s |           32 GB/s |
| 两跳串行实测     | **13 GB/s** (81%) | **27 GB/s** (84%) |

两次 DMA 的 PCIe 协议开销叠加，实际效率约为单跳效率的平方（81%² ≈ 66%，86%² ≈ 74%），与实测吻合。

在生产环境中：TP 组内的 GPU 必须 NVLink 直连。如果 TP=4 且 4 个 GPU 不在同一个 NVSwitch 域内，vLLM 会在 `initialize_from_config` 时检测到拓扑不匹配并告警。

---

## 4. 跨进程共享——"两个 vLLM 实例能不能共享显存里的 KV cache？"

你做 8 实例、TP=1 部署。实例 A 计算了 prompt 的 KV cache 存到 HBM，实例 B 处理相同的 prompt——能不能直接读实例 A 的显存？

### 4.1 进程隔离：谁也看不见谁

和 CPU 一样，每个 CUDA context 有独立的虚拟地址空间。实例 A 的 `cudaMalloc` 返回 `0x7f000000`，实例 B 的同一地址指的不是同一块物理 HBM。这就是为什么 LMCache 用**磁盘**（不是显存）做跨实例共享——每个实例独立读写同一个文件，互不干扰。

那么 CUDA 的 Unified Memory（`cudaMallocManaged`）呢？它号称可以让多个 GPU 访问同一块内存。要理解它的工作原理——以及为什么它也无法胜任 LLM 推理——必须先理解**缺页中断**。

```text
实例 A (pid=1234)                  实例 B (pid=5678)
  cudaMalloc → 0x7f000000            cudaMalloc → 0x7f000000
       ↓                                   ↓
  物理 HBM 0x10000                    物理 HBM 0x20000
  (完全不同)                          (完全不同)
```

### 4.2 先理解"缺页"：当程序访问一块还不存在的内存时

无论是 CPU 还是 GPU，程序访问一个虚拟地址时，MMU 查页表——如果发现这个地址还没有对应的物理页，就会触发一个**缺页中断**（page fault）。操作系统（或 GPU 驱动）的 page fault handler 被唤醒，分配一块物理内存，更新页表，然后程序继续执行——就像什么都没发生过。

这个过程对程序透明，但**不是免费的**：CPU 缺页处理大约 1-2μs，GPU 缺页处理（Unified Memory 场景）约 10μs。单次看起来不多，但累计起来——尤其是 LLM 推理中对延迟极其敏感——就是致命的。

### 4.3 Unified Memory：共享的幻觉

CUDA 确实提供了跨进程共享显存的机制——`cudaMallocManaged`。它的底层原理就是**缺页驱动的按需迁移**。但它的工作方式和很多人的直觉相反。

```cuda
// 实例 A 和 B 都可以访问这个指针
float *shared;
cudaMallocManaged(&shared, N * sizeof(float));

// 实例 A 写入，GPU 0 的 page fault handler 把页面迁移到 GPU 0
kernel_A<<<...>>>(shared);

// 实例 B 读取——page fault！页面从 GPU 0 迁移到 GPU B
// 延迟: ~10 μs per page × (N / 64KB pages)
kernel_B<<<...>>>(shared);
```

每 64KB 一页，每页迁移延迟 ~10μs。如果 N=4GB，那是 **65,536 次 page fault**，总延迟 **655ms**。对于需要 sub-100ms TTFT 的 LLM 推理来说，这是不可接受的。

CPU 类比：`mmap` 的文件映射也是 page fault 驱动的——第一次读文件时缺页中断把磁盘数据读入 page cache。64KB GPU 页 ≈ Linux 的 THP (2MB pages)，都是为了减少 TLB 压力和 fault 次数。

**为什么 LMCache 用磁盘而不是 UM**：Unified Memory 跨实例共享在理论上是可行的——两个 CUDA context 可以通过 IPC (`cudaIpcGetMemHandle`) 共享同一个 `cudaMallocManaged` 区域。但 page fault 的延迟对于 LLM 推理是致命的——KV cache 的每个 256-token chunk 约 14MB，跨 GPU 迁移需要 ~210 次 page fault，2ms 延迟。而 LMCache 的 L3 磁盘方案（GDS DMA）也差不多是这个量级——**但不需要 GPU 间协调**，每个实例独立读磁盘。

### 4.4 CUDA IPC：显式的跨进程共享

除了 UM，CUDA 还有一套专门的跨进程共享 API——`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`。它可以让进程 A 分配一块显存，导出为一个不透明的 handle，进程 B 通过 handle 打开并直接访问同一块物理 HBM。

```cuda
// 进程 A：导出
cudaIpcMemHandle_t handle;
cudaIpcGetMemHandle(&handle, gpu_ptr);

// 进程 A 把 handle 通过 shm/pipe 传给 B

// 进程 B：导入
void *remote_ptr;
cudaIpcOpenMemHandle(&remote_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
// 现在 remote_ptr 指向的就是进程 A 的同一块物理 HBM
```

**为什么不用于 LMCache？** IPC 共享是"真共享"——零拷贝，延迟最低。但它要求**进程间协调**：进程 B 必须知道进程 A 的 handle，必须在使用期间保持 handle 有效。如果进程 A 崩溃或释放了内存，进程 B 立即出现 segfault。LMCache 选择磁盘而非 IPC 的根本原因不是性能——是**故障隔离**。8 实例共享一块 IPC 显存，一个实例的 OOM 或 crash 会连锁影响其他 7 个实例。磁盘方案天然隔离——每个实例读自己的文件，一个崩溃不影响其他。

### 4.5 GPU Memory Pool：预分配 + Sub-allocation

vLLM 实际管理 KV cache 时，并不是每次都调用 `cudaMalloc`。它使用 CUDA Memory Pool（`cudaMemPool`）——预分配一大块显存，然后自己做 sub-allocation：

```cuda
// 创建 pool，预分配 40GB
cudaMemPoolProps props = {.allocType = cudaMemAllocationTypePinned};
cudaMemPool_t pool;
cudaMemPoolCreate(&pool, &props);
cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, ...);

// 从 pool 中分配（不是从全局显存）
void *kv_block;
cudaMallocFromPoolAsync(&kv_block, block_size, pool, stream);
```

好处：

- **无碎片**：pool 内部的分配由应用程序控制，不会出现全局显存级别的碎片
- **快**：不需要走 CUDA 驱动的全局分配器
- **可预测**：KV cache 的 block 大小固定（vLLM 中默认 16 tokens × KV 维度），天然适合 pool 管理

CPU 类比：`jemalloc` 或 `tcmalloc`——替代系统的 `malloc`，通过预分配 arena 和线程本地缓存减少锁竞争和碎片。

---

## 5. 绕过 CPU——"能不能让 NVMe 直接把数据写到 GPU？"

接上一节：LMCache L3 的 KV cache 存在磁盘上。传统路径是：

```text
NVMe →(DMA)→ CPU RAM →(CPU copy)→ GPU HBM
        ~3 GB/s              ~22 GB/s     总计 ~2.6 GB/s（串行）
```

GPUDirect Storage 把中间环节砍掉：

```text
NVMe →(DMA)→ GPU HBM
        ~3-7 GB/s（取决于 NVMe 型号和 PCIe 拓扑）
```

### 5.1 DMA 链条：为什么需要 P2P + udev？

GDS 依赖一条完整的 DMA 通道：

```text
NVMe 控制器 ──PCIe── PCIe Switch ──PCIe── GPU
     │                                        │
     └── 数据走 PCIe P2P DMA ─────────────────┘
              (不经过 CPU Root Complex)
```

这条通道要工作，需要三个条件：

1. **BIOS 层面**：PCIe ACS (Access Control Services) 必须禁用，允许 P2P 流量跨越 PCIe 分支
2. **内核层面**：`nvidia-fs.ko` 模块加载，NVMe 块设备注册到 nvidia-fs 驱动
3. **用户层面**：cuFile 库能通过 udev 查询块设备属性（`/dev/nvme1n1p1` 的 `ID_FS_USAGE` 等）

其中第 3 条就是我们在 H100 上排查了 2 小时的 `cuFile err=5030` 的根因——容器内缺少 `/run/udev` 挂载。

```bash
# 确认 GDS 可用
/usr/local/cuda/gds/tools/gdscheck -p  # 平台检查
/usr/local/cuda/gds/tools/gdscheck -f /mnt/nvme/test  # 文件注册测试

# 实时查看 DMA 吞吐
nvidia-smi dmon -s pucv  # PCIe + NVLink utilization
```

CPU 类比：`sendfile()` 系统调用——数据从磁盘文件描述符直接传到 socket 文件描述符，不经过用户态缓冲区。GDS 是 GPU 版本的 `sendfile`。

**代价**：GDS 对硬件有严格要求——需要 NVIDIA 认证的 NVMe 驱动器、特定的 PCIe 拓扑（GPU 和 NVMe 必须在同一 PCIe 交换机下）、以及 nvidia-fs 内核模块。不是所有系统都能用。

---

## 6. 大页——"2MB 的页和 4KB 的页，对 GPU 有什么影响？"

在 CPU 上启用 2MB 大页后，你发现 CPU→GPU 传输快了约 5%。为什么 GPU 传输受 CPU 页面大小影响？

### 6.1 先理解 TLB：MMU 的"快捷键"

MMU 每次查页表都要访问内存（页表本身存在 RAM 里），这太慢了。所以 CPU 和 GPU 都有一个 **TLB**（Translation Lookaside Buffer，旁路转换缓冲），相当于 MMU 的 Cache——把最近用过的"虚拟→物理"映射缓存起来。

TLB 的条目数有限（通常几十到几百条）。如果访问的内存分散在太多小页面中，TLB 不够用，MMU 就得频繁去 RAM 里查页表——这叫 **TLB miss**，每次多花几十个 CPU 周期。大页的意义就是：一个 2MB 的页只需要 1 个 TLB 条目，覆盖同样 2MB 范围如果用 4KB 页则需要 512 个条目。

### 6.2 DMA 传输需要遍历页表

GPU DMA 引擎发起传输时，内核必须锁定所有涉及的物理页面（pin），然后构建一个 **scatter-gather list**（散聚列表）给 DMA 引擎——因为 DMA 引擎需要知道"从物理地址 A 读 4KB，跳到物理地址 B 再读 4KB，跳到物理地址 C..."。页越大，这个列表越短：

```text
4KB 页（传输 1GB）：
  1GB / 4KB = 262,144 个页表条目
  内核遍历 262K 条目 → 构建 262K 条目的 SG list → 发给 DMA 引擎

2MB 大页（传输 1GB）：
  1GB / 2MB = 512 个页表条目
  遍历快 512 倍，SG list 短 512 倍
```

这个遍历发生在**每次 `cudaMemcpy` 调用**——除非你用了 pinned memory（`cudaMallocHost`），内核只需要在分配时 pin 一次，后续传输直接用。

GPU 本身也有大页概念——CUDA 的 Unified Memory 使用 64KB 页面粒度，不是标准的 4KB。这是 CUDA 驱动层面的优化，对用户透明。

```bash
# CPU 端：查看大页配置
cat /proc/meminfo | grep Huge
# HugePages_Total:    1024
# Hugepagesize:       2048 kB
```

### 6.3 使用大页分配

```c
mmap(NULL, size, PROT_READ|PROT_WRITE,
     MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, -1, 0);
```

**实践**：对于 vLLM 推理，CPU 端使用大页对 CPU→GPU 传输影响有限（因为 vLLM 传输量不大）。但**CPU 内存池**（如 LMCache 的 L2 CPU cache）使用大页可以提升存储和检索吞吐——`/proc/sys/vm/nr_hugepages` 的配置值得关注。

---

## 7. 内存层级——"这一路的延迟差了多少？"

把前面所有概念串起来，从最快的寄存器到最慢的磁盘：

| 层级 | CPU              | 延迟             | 类比 | GPU                   | 延迟                |
| ---- | ---------------- | ---------------- | :--: | --------------------- | ------------------- |
| 最快 | L1 Cache (32KB)  | ~1 ns            |  ←   | Register (256KB/SM)   | ~0 ns               |
|      | L2 Cache (256KB) | ~4 ns            |  ←   | Shared Mem (228KB/SM) | ~5 ns               |
|      | L3 Cache (32MB)  | ~12 ns           |  ←   | L1 Cache (256KB/SM)   | ~28 ns              |
|      | RAM (512GB)      | ~100 ns          |  ←   | L2 Cache (50MB)       | ~200 ns             |
|      |                  |                  |      | HBM (80GB)            | ~400 ns             |
| 最慢 | swap / NVMe      | ~10 μs / ~100 μs |  ←   | UM / GDS              | ~10 μs / ~50-100 μs |

**关键差异**：

- GPU 的 Shared Memory 是**显式管理的**——程序员决定什么数据放进去。CPU 的 L1/L2/L3 Cache 对程序员透明
- GPU 没有 swap。HBM 满了就是 OOM。CPU 可以通过 swap 换出到磁盘（慢但能用）
- GPU 通过 GDS 访问 NVMe 的延迟（50-100μs）远低于 CPU 的常规磁盘访问（100-500μs），因为 DMA 绕过了 CPU 的 VFS/block 层

---

## 8. 性能诊断——"我怎么知道瓶颈在哪？"

推理慢了。是 HBM 带宽不够？是 PCIe 拥塞？还是 GPU 在等 page fault？

### 8.1 对标工具

| 诊断维度     | CPU (Linux)                  | GPU (CUDA)                                        |
| ------------ | ---------------------------- | ------------------------------------------------- |
| 实时内存占用 | `free -h`, `vmstat 1`        | `nvidia-smi`, `nvidia-smi dmon -s mu`             |
| 进程内存细项 | `pmap -x <pid>`              | `nvidia-smi --query-compute-apps=pid,used_memory` |
| 带宽/吞吐    | `perf stat -e bus-cycles`    | `nvidia-smi dmon -s pucv`                         |
| 缺页         | `perf stat -e page-faults`   | `nsys profile --stats=true`                       |
| 拓扑         | `numactl --hardware`         | `nvidia-smi topo -m`                              |
| 长期趋势     | Prometheus + `node_exporter` | Prometheus + `dcgm-exporter`                      |
| 内核级 trace | `perf record`, `bpftrace`    | `nsys profile`, `ncu`                             |
| GPU 利用率   | —                            | `nvidia-smi dmon -s u`                            |

```bash
# 一行实时监控：显存 + 利用率 + PCIe/NVLink 吞吐
nvidia-smi dmon -s mucp -d 2

# 输出列：
#  sm   mem   enc   dec    pci_rx    pci_tx   nv_rx   nv_tx
#  65   45G     0     0     1024     2048     18000   18000
#  ↑    ↑                     ↑                 ↑
#  SM   显存                  PCIe RX/TX       NVLink RX/TX
#  利用率 用量                 (MB/s)           (MB/s)
```

**典型故障模式**：

| 症状                                 | 可能原因                               | 如何确认                                                                |
| ------------------------------------ | -------------------------------------- | ----------------------------------------------------------------------- |
| TTFT 偶尔飙高                        | Unified Memory page fault 累积         | `nsys profile --stats=true`，看 page fault count                        |
| TP 通信慢                            | GPU 间经 PCIe 而非 NVLink              | `nvidia-smi topo -m`，核对 TP 组内 GPU 连接类型                         |
| GPU 利用率低，但 CPU 闲着            | GPU 在等 HBM —— 带宽瓶颈               | `nvidia-smi dmon -s mucp`，若 SM 低但 HBM 读写饱和，则是带宽瓶颈        |
| GPU 利用率低，CPU 繁忙               | CPU 端是瓶颈（tokenization、请求调度） | `htop` 看 CPU 核心使用率；`perf top` 看热点函数                         |
| cudaMalloc OOM 但 nvidia-smi 有 free | 显存碎片化，无连续大块可用             | 重启进程并先分配 max tensor；降 GPU_MEM_UTIL                            |
| cudaMemcpy 比预期慢                  | 未使用 pinned memory                   | 改用 `cudaMallocHost`；`nvidia-smi dmon -s pucv` 查看 PCIe 吞吐是否打满 |

---

## 9. 决策树——"什么时候该操心这些？"

```text
你在做推理部署，GPU 内存相关的问题：
│
├─ 模型加载 OOM？
│   ├─ 模型太大 → TP 分片 或 降低 GPU_MEM_UTIL
│   ├─ 碎片化 → 重启进程，先分配大块
│   └─ 有其他进程占用 → nvidia-smi 查谁在占
│
├─ CPU→GPU 传输慢？
│   ├─ 数据量大（>100MB/次） → 用 cudaMallocHost (pinned)
│   ├─ 数据量小 → 不用折腾，malloc 足够
│   └─ 频繁传输 → 预分配 + 重用 buffer
│
├─ 多 GPU 通信慢？
│   ├─ NVLink 可用 → 优先！确保 GPU 在同一 NVSwitch 域
│   ├─ 只有 PCIe → 考虑 PCIe P2P
│   └─ 经 CPU 中转 → 最慢，只有别无选择时用
│
├─ 跨实例共享 KV cache？
│   ├─ 同机多 GPU → 共享磁盘 (L3)，不要用 Unified Memory
│   ├─ 跨机 → Controller 模式 或 共享文件系统
│   └─ 低延迟要求 (<1ms) → GPU 间 P2P
│
├─ 磁盘→GPU 传输？
│   ├─ 硬件支持 GDS → 用 GDS (DMA 直通)
│   ├─ 不支持 GDS → 用 O_DIRECT + POSIX 回退
│   └─ 容器部署 → 记得挂载 /run/udev + nvidia-fs 设备
│
└─ 性能诊断？
    ├─ 长期趋势 → DCGM + Prometheus
    ├─ 单次 profile → nsys
    └─ 实时监控 → nvidia-smi dmon
```

---

## 附录：快速参考——CPU ↔ GPU 命令对照

| 你要做什么         | CPU (Linux)                     | GPU (CUDA)                           |
| ------------------ | ------------------------------- | ------------------------------------ |
| 查看内存空闲       | `free -h`                       | `nvidia-smi --query-gpu=memory.free` |
| 进程内存详情       | `pmap -x <pid>`                 | `nvidia-smi --query-compute-apps`    |
| 分配不可换出的内存 | `mlock`                         | `cudaMallocHost`                     |
| 大页分配           | `mmap(MAP_HUGETLB)`             | 对用户透明（64KB UM 页）             |
| 进程间共享内存     | `shm_open` / `mmap(MAP_SHARED)` | `cudaIpcGetMemHandle`                |
| 查看拓扑           | `numactl --hardware`            | `nvidia-smi topo -m`                 |
| DMA 直通存储       | `sendfile()`（文件→socket）     | GDS / cuFile（NVMe→GPU）             |
| Page fault 统计    | `perf stat -e page-faults`      | `nsys profile --stats=true`          |
| 带宽监控           | `perf stat -e bus-cycles`       | `nvidia-smi dmon -s pucv`            |
| OOM 时怎么办       | 减小 batch / 增加 swap          | 减小 batch / TP 分片                 |
