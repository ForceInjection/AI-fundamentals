# GPU 间数据传输方法实测——从 P2P 到 Unified Memory 的 100 倍差距

你在 8×A100 上做 TP=4 推理。GPU 0 算完一个 layer，要把结果传给 GPU 1、2、3。你知道机器有 NVLink，随手写了行 `cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)`——代码能跑，但实际走的是 CPU 中转（~12 GB/s）而非 NVLink（~249 GB/s）。没有 error、没有 warning——带宽差了 **21 倍**，你直到看 `nvidia-smi dmon` 才发现 PCIe 吞吐爆了。

GPU 间数据传输不是 API 调用的问题——是**三条物理路径 × 四种编程方法 × 拓扑前提条件**的组合选择。选错路径的代价从 21× 到 119× 不等。

本文将四种方法按带宽分为三层：Tier 1（NVLink P2P，~249 GB/s）→ Tier 2（CPU relay，~12 GB/s）→ Tier 3（Zero-Copy/Unified Memory，~2-3 GB/s）。Tier 1 有两种 API 等效写法（`cudaMemcpyPeer` 和 `cudaMemcpy D2D`），性能相同。P2P 和 CPU relay 为往返测试（A→B→A 串行）中的每方向等效速率；Zero-Copy 和 UM 为单向速率。

- **可交互概念图**：本文配有可交互的概念图，覆盖 3 个图层（物理拓扑 / 测试场景 / 实测结果），建议在阅读对应章节时打开对照查看：[gpu-transfer-methods-visual.html](gpu-transfer-methods-visual.html)。
- **验证代码**：[09_gpu_transfer_methods.cu](code/09_gpu_transfer_methods.cu) — 四种方法一次测完。`CUDA_VISIBLE_DEVICES=0,1` 选择 NVLink 直连的 GPU 对即可。
- **测试环境**：A100-SXM4-80GB (NVLink NV12, GPU 0↔1)，CUDA 12.8，2026-06 实测。
- **测试场景**：往返（A→B 再 B→A，串行交替，非全双工同时收发）。每次迭代传输 128 MB/方向，10 次迭代（3 次 warmup）。报告值为**每方向等效速率**（往返总数据量 / 往返总时间；单方向用时 ≈ 总时间的一半）。
- **对比规格**：NVLink 3.0 单向理论值 = 300 GB/s（12 条 link × 25 GB/s/link/方向）。NVIDIA `simpleP2P` 单向持续传输实测 = 239 GB/s。

---

## 1. 三条物理路径，四种 CUDA 方法

GPU 间数据搬运在物理上有三条路可走。CUDA 提供了多种 API 覆盖这些路径——但 API 的名字不告诉你它走的是哪条路。

### 1.1 三条物理路径

```text
路径 A: NVLink 直连
  GPU 0 ──NVLink──→ GPU 1          ← 最快: ~249 GB/s 单向 (A100 NVLink 3.0)

路径 B: PCIe P2P
  GPU 0 ──PCIe Switch──→ GPU 1     ← 较快: ~28 GB/s (Gen4), 需 PIX 拓扑

路径 C: CPU 中转
  GPU 0 ──PCIe──→ CPU DRAM ──PCIe──→ GPU 1   ← 最慢: ~12 GB/s, 两次 DMA 穿越
```

NVLink 是 NVIDIA 专用互连，仅在数据中心 GPU（A100/H100）上可用。PCIe P2P 允许同一 PCIe Switch 下的 GPU 直接通信（不经 CPU 内存）。CPU 中转是所有 GPU 都支持的兜底方案——兼容性最好，带宽最差。

### 1.2 四方法 ↔ 三路径的映射

| 方法           | CUDA API                                     | 物理路径           | 每方向等效速率 | P2P 依赖 |
| -------------- | -------------------------------------------- | ------------------ | -------------- | -------- |
| **P2P 直连**   | `cudaMemcpyPeer` / `cudaMemcpy D2D`          | NVLink 或 PCIe P2P | **249 GB/s**   | 是       |
| CPU relay      | `cudaMemcpy` H2D + D2H                       | CPU 中转           | 11.8 GB/s      | 否       |
| Zero-Copy      | `cudaHostAlloc` + mapped pointer             | PCIe mapped DRAM   | 2.9 GB/s       | 否       |
| Unified Memory | `cudaMallocManaged` + `cudaMemPrefetchAsync` | 按需页面迁移       | 2.1 GB/s       | 否       |

> P2P 直连有两种 API：`cudaMemcpyPeer` 显式指定 src/dst device，`cudaMemcpy D2D` 让 CUDA runtime 自动选择路径。两者走**同一条物理路径**，实测带宽相同（249.00 vs 248.89 GB/s，差异在 event 开销内）。其余带宽为往返测试中每方向等效速率（Zero-Copy 和 UM 为单向速率）。

---

## 2. 三个陷阱——"为什么我的传输没有想象中快"

四种方法放在面前，直觉是"选最快的"。但三个陷阱让这个选择比看起来复杂得多。

### 2.1 陷阱一：P2P 的静默降级

你写了 `cudaMemcpyPeer`，代码能跑——但实际走的是 **CPU 中转**，不是 NVLink。

```text
期望:  GPU 0 ──NVLink──→ GPU 1      249 GB/s
实际:  GPU 0 → CPU → GPU 1           12 GB/s   ← 静默降级, 无 error!
```

原因：你在调用 `cudaMemcpyPeer` 之前**没有** `cudaDeviceEnablePeerAccess`。CUDA runtime 不会报错——它默默地用 CPU relay 完成传输。排查方法：

```bash
# 确认 P2P 可用
nvidia-smi topo -p2p r | grep "GPU0.*GPU1"

# 在代码中检查
cudaDeviceCanAccessPeer(&canPeer, 0, 1);
printf("P2P: %s\n", canPeer ? "YES" : "NO — 会静默走 CPU relay!");
```

**修复**：在程序初始化时，对所有需要的 GPU 对调用 `cudaDeviceEnablePeerAccess`。

### 2.2 陷阱二：Unified Memory 的大块搬运

`cudaMallocManaged` 是最省心的 API——分配后 CPU 和 GPU 都能直接访问同一个指针。但 128 MB 数据量下带宽仅 2.10 GB/s——比 P2P 慢 119 倍。

原因：Unified Memory 依赖**缺页中断**（page fault）驱动的按需迁移。每 64 KB 一页，每页迁移延迟约 10 μs。128 MB = **2048 次 page fault**，累计延迟 ≈ 20 ms——每次访问都触发一次 PCIe 往返。

> Unified Memory 的正确用法是**细粒度、频繁访问的小数据**（如 CPU→GPU 的参数同步），不是大块数据搬运。

### 2.3 陷阱三：API 复杂度 ≠ 性能

```text
编程复杂度:  UM < Zero-Copy < cudaMemcpy < cudaMemcpyPeer
    ↓              ↓           ↓             ↓
实际带宽:   2 GB/s     3 GB/s     249 GB/s     249 GB/s
```

最简单的 API（Unified Memory）反而是最慢的。最复杂的（cudaMemcpyPeer）是最快的。GPU 编程中，**便利性与性能成反比**——这是与 CPU 编程最大的不同。

---

## 3. 三层带宽对比——四种方法按速度分组

四种方法按带宽自然聚为三层。每一层从实测数据出发，给出代码、适用场景和注意事项。

### 3.1 Tier 1: P2P 直连（249 GB/s）

P2P 直连有两种等效的 API 写法，底层走同一条物理路径（NVLink 或 PCIe P2P）。本节展示 NVLink 路径的实测数据。

**往返测试实测**（两个 API 结果相同）：

```text
cudaMemcpyPeer (NVLink/P2P):  10.04 ms    249.00 GB/s
cudaMemcpy D2D (P2P enabled): 10.04 ms    248.89 GB/s
```

> 两个 API 走同一物理路径，实测带宽几乎相同（差异 < 0.05%，在 event 开销波动范围内）。NVLink 3.0 单向规格 = 300 GB/s（12 条 link × 25 GB/s/link/方向）。实测 249 GB/s，效率 83%。NVIDIA `simpleP2P` sample 单向持续传输实测 = 239 GB/s（80%），与本文结果一致（差异 < 5%，来自往返测试的额外开销）。参见 [08_p2p_bandwidth.md](08_p2p_bandwidth.md)。

**API 写法一：cudaMemcpyPeer —— 显式指定 src/dst device**：

```c
// 前提: 已 cudaDeviceEnablePeerAccess
cudaMemcpyPeer(dst_ptr, dst_device, src_ptr, src_device, size);
```

直接指定源和目标 GPU ID，src 和 dst 均为 device pointer，**不经 CPU 内存**。

**API 写法二：cudaMemcpy D2D —— 更简洁，P2P 开启后自动走 NVLink**：

```c
cudaDeviceEnablePeerAccess(peer_device, 0);   // 先开启 P2P
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);  // 自动走 NVLink
```

一旦 P2P 开启，普通的 `cudaMemcpy` 自动利用 NVLink——性能与 cudaMemcpyPeer 完全相同。**这是 TP 推理中最常用的模式**。

> **易错点**：`cudaDeviceEnablePeerAccess(1, 0)` 必须在正确的 device context 下调用。先 `cudaSetDevice(0)` 再 `enablePeerAccess(1, 0)`——如果 device context 搞反，调用会静默失败。

**前提条件**：

- `cudaDeviceCanAccessPeer` 返回 true
- 物理拓扑为 NVLink 或 PIX（`nvidia-smi topo -m` 确认）
- 必须先 `cudaDeviceEnablePeerAccess`

**PCIe P2P 变体**：同样的 API 在 PIX 拓扑（同 PCIe Switch）下可用，但带宽约 28 GB/s 单向（受限于 PCIe Gen4 x16 链路）。关于 PCIe P2P 的详细分析见 [PCIe 带宽实测](02_pcie_bandwidth_measurement.md)。vLLM 在生产环境中实测 NCCL P2P 可达 ~16 GB/s（含 NCCL 协议开销和真实模型权重的多 tensor 启动损耗）。

### 3.2 Tier 2: CPU relay（11.8 GB/s）

当 P2P 不可用时（SYS/NODE 拓扑），这是唯一的兜底方案。

**实测数据**：

```text
3. CPU relay (G->CPU->G):  212.57 ms    11.76 GB/s
```

**代码**：

```c
float *host_buf;
cudaMallocHost(&host_buf, size);         // pinned memory — 关键!
cudaMemcpy(host_buf, d_src, size, cudaMemcpyDeviceToHost);   // GPU→CPU
cudaMemcpy(d_dst, host_buf, size, cudaMemcpyHostToDevice);   // CPU→GPU
```

Pinned memory (`cudaMallocHost`) 是性能关键——普通 `malloc` 的 pageable memory 无法做异步 H2D/D2H，DMA 引擎每次都要遍历页表、锁定物理页。Pinned memory 在分配时即锁定，DMA 引擎直接传输。

**适用场景**：GPU 拓扑为 SYS/NODE（跨 NUMA node 或无 NVLink）。缺点：占用 CPU 内存带宽，延迟高。每方向需两次 PCIe 穿越（D2H + H2D），单次穿越受限于 PCIe Gen4 x16 单向规格 ~32 GB/s。

#### SHM/mmap 进阶：跨进程共享 relay buffer

当 relay buffer 需要在多个进程间共享（如推理服务前后端分离），可以用 `mmap` + `cudaHostRegister` 替代 `cudaMallocHost`：

```c
int fd = shm_open("/gpu_buf", O_CREAT | O_RDWR, 0666);
ftruncate(fd, size);
float *buf = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
cudaHostRegister(buf, size, cudaHostRegisterPortable);  // pin mmap'd memory
// ... cudaMemcpyAsync(buf, d_src, size, cudaMemcpyDeviceToHost, stream) ...
```

| 场景                    | 说明                                                              |
| ----------------------- | ----------------------------------------------------------------- |
| 多进程共享 relay buffer | 进程 A 将 GPU 数据拷入 SHM → 进程 B 从 SHM 读取，无需 CPU 间拷贝  |
| 文件 I/O 优化           | 对 mmap 映射的文件页做 `cudaHostRegister` 后可直接 DMA 到 GPU     |
| 内存不可迁移            | 已有 mmap/大页分配的 buffer 不想重构为 `cudaMallocHost`，注册即可 |

**A100 实测**：对于 128 MB 数据块，`mmap+cudaHostRegister` 与 `cudaMallocHost` 带宽无差异（均受限于 PCIe Gen4）。SHM 的优势不在带宽，在跨进程零拷贝。

| 方法                      | 128 MB 每方向速率 | 跨进程共享 |
| ------------------------- | ----------------- | ---------- |
| `cudaMallocHost`          | 5.26 GB/s         | 否         |
| `mmap + cudaHostRegister` | 5.40 GB/s         | **是**     |

### 3.3 Tier 3: Zero-Copy & Unified Memory（2-3 GB/s）

这两种方法的带宽远低于前两层——**不适合大块数据搬运**。但它们各自有不可替代的场景。

**实测数据**：

```text
4. Zero-Copy (mapped host memory):  431.28 ms    2.90 GB/s
5. Unified Memory (prefetch):       595.77 ms    2.10 GB/s
```

**Zero-Copy —— mapped host memory**：

```c
cudaHostAlloc(&ptr, size, cudaHostAllocPortable | cudaHostAllocMapped);
cudaHostGetDevicePointer(&d_ptr, ptr, 0);  // 同一块内存的 device 指针
```

Host 内存被映射到 GPU 地址空间，GPU kernel 可以直接通过 load/store 访问——**不需要显式 memcpy**。但每次访问都要经 PCIe 往返，延迟远高于显存。

Zero-Copy 的优势在**小数据、频繁访问**场景（如 CPU→GPU 的配置参数同步），不是大块搬运。128 MB 下带宽仅 2.90 GB/s——因为映射内存在大块传输场景下每次 GPU 访问实际仍触发 PCIe 往返，且不支持 DMA 批量传输。

**Unified Memory —— 最省心**：

```c
cudaMallocManaged(&ptr, size);      // 分配统一内存
cudaMemPrefetchAsync(ptr, size, device_id);  // 提示：请迁移到 GPU 0
```

CUDA 驱动自动管理页面迁移。添加 `cudaMemPrefetchAsync` 可以提前将数据迁到目标 GPU，减少首次访问的 page fault。编程模型是四种方法中最简单的：**分配后 CPU 和 GPU 都能直接用同一个指针**。

适用场景：**快速原型、移植遗留 CPU 代码、细粒度频繁访问的小数据**。不适合追求极致带宽的大块搬运。

---

## 4. 选型决策——速度 vs 复杂度 vs 兼容性

### 4.1 决策树

```text
需要 GPU 间搬数据？

├─ 拓扑是 NVLink 或 PIX (有 P2P)？
│  └─ ✅ P2P 直连 (cudaMemcpyPeer / cudaMemcpy D2D)
│      前提: cudaDeviceEnablePeerAccess 已调用
│      带宽: ~249 GB/s (NVLink) 或 ~28 GB/s (PCIe P2P)
│
├─ 无 P2P，但需要极限兼容性？
│  └─ ✅ CPU relay
│      带宽: ~12 GB/s (往返测试每方向等效)
│      + 多进程共享 → mmap + cudaHostRegister
│
├─ 数据量 < 1 MB 且频繁访问？
│  └─ ✅ Zero-Copy
│      GPU kernel 直接读写 mapped host memory
│
├─ 快速原型 / 移植老代码？
│  └─ ✅ Unified Memory
│      编程最简单: 一个指针 CPU/GPU 通用
│      ⚠️ 大块数据触发大量 page fault
│
└─ 追求极致带宽？
   └─ ✅ 确保 NVLink + P2P 已启用 + cudaMemcpyPeer / cudaMemcpy D2D
       验证: nvidia-smi topo -m | grep NV
```

### 4.2 三层对比总览

| Tier | 方法           | 每方向等效速率 | vs 最快 | P2P 依赖 | 编程复杂度 | 一句话                             |
| ---- | -------------- | -------------- | ------- | -------- | ---------- | ---------------------------------- |
| 1    | P2P 直连       | **249 GB/s**   | 1×      | 是       | 中         | NVLink 直连，最快（两种 API 等效） |
| 2    | CPU relay      | 11.8 GB/s      | 21×     | 否       | 中         | 兼容性最好，pinned memory 是关键   |
| 3    | Zero-Copy      | 2.9 GB/s       | 86×     | 否       | 低         | 小数据频繁访问场景                 |
| 3    | Unified Memory | 2.1 GB/s       | 119×    | 否       | 最低       | 快速原型，最省心                   |

### 4.3 核心教训

1. **P2P 不是默认的**——必须显式 `cudaDeviceEnablePeerAccess`，否则静默降级到 CPU relay（21× 慢）
2. **API 复杂度与性能成反比**——最简单的 UM 是最慢的（2.1 GB/s），最复杂的 cudaMemcpyPeer 是最快的（249 GB/s），差距 119 倍
3. **Unified Memory 是大块搬运的反模式**——page fault 机制决定了它适合细粒度、频繁访问场景
4. **SHM 的带宽不优于 pinned memory**——它的价值在跨进程零拷贝，不在带宽

---

## 5. 相关文档

- [08_p2p_bandwidth.md](08_p2p_bandwidth.md)：P2P 单向带宽 239 GB/s 的详细测试和拓扑分析
- [02_pcie_bandwidth_measurement.md](02_pcie_bandwidth_measurement.md)：PCIe H2D/D2H 单卡带宽 ~28 GB/s
- [03_hbm_bandwidth_test.md](03_hbm_bandwidth_test.md)：片内带宽 ~1453 GB/s (Kernel) / ~821 GB/s (Copy Engine)——与片间 P2P 的 249 GB/s 形成"片内/片间"层级对比
- [NVLink 技术入门](../../01_hardware_architecture/nvlink/nvlink_intro.md)
- [CUDA Peer-to-Peer Memory Access](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#peer-to-peer-memory-access)
- [CUDA Unified Memory Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)
