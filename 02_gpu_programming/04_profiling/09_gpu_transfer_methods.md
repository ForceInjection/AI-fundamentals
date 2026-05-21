# GPU 间数据传输方法实测

> 基于 A100-SXM4-80GB (NVLink NV12, GPU 3↔4) + CUDA 13.1 实测。GPU 间搬数据不止 `cudaMemcpy` 一种方式——本文对比 5 种方法在 128 MB 数据量下的带宽，从 124 GB/s 的 NVLink P2P 到 2 GB/s 的 Unified Memory，差距达 60 倍。

---

## 1. 为什么有多种搬运方式

GPU 间数据传输有三条物理路径，CUDA 提供了五种编程方法覆盖这些路径：

| 物理路径                      | CUDA 方法                                 | 实测?                          |
| ----------------------------- | ----------------------------------------- | ------------------------------ |
| NVLink (600 GB/s 双向)        | `cudaMemcpyPeer` / `cudaMemcpy D2D` (P2P) | ✅ NVLink NV12 实测            |
| PCIe P2P (~28 GB/s Gen4 实测) | 同上（同一套 API，底层自动选路径）        | ⚠️ 需 PIX 拓扑，本环境无可测对 |
| PCIe + CPU DRAM               | CPU relay (H2D + D2H)                     | ✅ 实测                        |
| PCIe mapped host memory       | Zero-Copy (`cudaHostAlloc`)               | ✅ 实测                        |
| 按需页面迁移                  | Unified Memory (`cudaMallocManaged`)      | ✅ 实测                        |

> **注意**：`cudaMemcpyPeer` / `cudaMemcpy D2D (P2P)` 是同一套 API，底层自动根据拓扑选择 NVLink 或 PCIe P2P 路径——无论哪种，数据都**不经 CPU 内存**。本文在 NVLink 对上实测了 P2P 的 NVLink 路径。PCIe P2P 路径的预期带宽约 28 GB/s 单向（一次 PCIe 穿越），是方法 3 CPU relay（需 H2D + D2H 两次穿越，~10.5 GB/s 单向）的约 **2.7 倍**。vLLM 在生产环境中实测 NCCL P2P 可达 ~16 GB/s（含 NCCL 协议开销和真实模型权重的多 tensor 启动损耗）。

### 1.1 物理拓扑与方法的对应关系

这 5 种方法分别对应不同的物理路径：

| 方法              | 物理路径                                   | 对应技术                                                                                       | 本文测试状态                                                                                     |
| ----------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1/2. P2P (NVLink) | GPU → NVLink → GPU                         | [NVLink 技术入门](../../01_hardware_architecture/nvlink/nvlink_intro.md)                       | ✅ **NV12 实测**: ~249 GB/s 单向                                                                 |
| 1/2. P2P (PCIe)   | GPU → PCIe Switch → GPU                    | [GPU 物理路径全景图 §4 PIX](../../01_hardware_architecture/assets/gpu_physical_data_paths.svg) | ⚠️ 需 PIX 拓扑，本环境无可测对 (预期 ~28 GB/s, 见 [PCIe 实测](02_pcie_bandwidth_measurement.md)) |
| 3. CPU relay      | GPU → PCIe → Root Complex → DRAM → … → GPU | [同上图 §1 GPU↔CPU Memory](../../01_hardware_architecture/assets/gpu_physical_data_paths.svg)  | ✅ 实测: 5.23 GB/s 双向                                                                          |
| 4. Zero-Copy      | GPU → PCIe → Root Complex → Mapped DRAM    | [同上图 §1](../../01_hardware_architecture/assets/gpu_physical_data_paths.svg)                 | ✅ 实测: 2.36 GB/s 双向                                                                          |
| 5. Unified Memory | GPU ↔ 按需页面迁移                         | [同上图 §1](../../01_hardware_architecture/assets/gpu_physical_data_paths.svg)                 | ✅ 实测: 2.05 GB/s 双向                                                                          |

> **参考资料对应关系**：[GPU 物理路径全景图](../../01_hardware_architecture/assets/gpu_physical_data_paths.svg) 覆盖所有 PCIe 路径（方法 1 PCIe P2P 的 §4 PIX 分类 + 方法 3/4/5 的 §1 GPU↔CPU Memory）。NVLink 路径独立于 PCIe，参见 [NVLink 技术入门](../../01_hardware_architecture/nvlink/nvlink_intro.md)。`nvidia-smi topo -m` 的六级 peer 分类（X / PIX / PXB / PHB / NODE / SYS）决定了 P2P 是否可用：**PIX/PXB 支持 P2P**，**SYS 不支持**。

---

## 2. 测试环境

| 项目     | 配置                                         |
| -------- | -------------------------------------------- |
| GPU 对   | GPU 3 ↔ GPU 4 (NV12 NVLink, 同 NUMA node 0)  |
| 数据量   | 128 MB                                       |
| 迭代次数 | 10 次 (3 次 warmup)                          |
| 测量方式 | `cudaEventRecord` 计时，双向传输 (A→B + B→A) |

---

## 3. 测试程序

完整测试程序见 [`gpu_xfer.cu`](gpu_xfer.cu)，一次编译即可测完 5 种方法。核心框架：

```c
#include <cuda_runtime.h>
#define N (128 * 1024 * 1024)   // 128 MB
#define IT 10                    // 10 iterations after 3 warmup

double run(const char* name, void (*fn)()) { ... }
void m1_peer() { cudaMemcpyPeer(d_b, 1, d_a, 0, N); ... }
void m2_d2d()  { cudaMemcpy(d_b, d_a, N, cudaMemcpyDeviceToDevice); ... }
...
```

编译运行（选择 NVLink 互连的 GPU 对）：

```bash
nvcc -arch=sm_80 -o gpu_xfer gpu_xfer.cu
CUDA_VISIBLE_DEVICES=3,4 ./gpu_xfer
```

---

## 4. A100 实测结果

### 4.1 完整输出

```text
Device 0: NVIDIA A100-SXM4-80GB
Device 1: NVIDIA A100-SXM4-80GB
P2P available: YES

Method                              Time  Bandwidth
------                              ----  --------
  1. cudaMemcpyPeer (NVLink)         10.05 ms    124.38 GB/s
  2. cudaMemcpy D2D (P2P on)         10.06 ms    124.22 GB/s
  3. CPU relay (G->CPU->G)          238.87 ms      5.23 GB/s
  4. Zero-Copy (mapped host)        529.84 ms      2.36 GB/s
  5. Unified Memory (prefetch)      608.47 ms      2.05 GB/s

=== Summary (128 MB) ===
  P2P / CPU-relay:    24x
  P2P / Zero-Copy:    53x
  P2P / Unified Mem:  60x
```

> 注：带宽为双向（A→B + B→A），单向 ≈ 249 GB/s（124.4 × 2），与 simpleP2P 实测 239 GB/s 接近（差距约 4%，来自双向测试的额外 event 开销）。

### 4.2 总对比表

| #   | 方法                      | 带宽 (双向)    | vs P2P  | 依赖 P2P | 编程复杂度                   |
| --- | ------------------------- | -------------- | ------- | -------- | ---------------------------- |
| 1   | `cudaMemcpyPeer`          | **124.4 GB/s** | 1×      | 是       | 中：需指定 src/dst device    |
| 2   | `cudaMemcpy` D2D (P2P on) | **124.2 GB/s** | 1×      | 是       | 低：普通 `cudaMemcpy` 即可   |
| 3   | CPU relay                 | 5.23 GB/s      | **24×** | 否       | 中：需 pinned host buffer    |
| 4   | Zero-Copy                 | 2.36 GB/s      | **53×** | 否       | 低：`cudaHostAlloc` + mapped |
| 5   | Unified Memory            | 2.05 GB/s      | **60×** | 否       | 最低：`cudaMallocManaged`    |

> **测试范围说明**：方法 1/2 的带宽数据来自 **NVLink NV12 路径**（GPU 3↔4）。同样的 `cudaMemcpyPeer` 代码在 PCIe P2P (PIX) 拓扑下也可用，但带宽约 28 GB/s 单向（受限于 PCIe Gen4 x16 链路），参见 [PCIe 带宽实测](02_pcie_bandwidth_measurement.md)。本服务器所有 GPU 对均为 NV12 或 SYS，无可测的 PIX 对，因此 PCIe P2P 的数据为理论预期值。

---

## 5. 方法解读

### 5.1 cudaMemcpyPeer — 显式 P2P

```c
cudaMemcpyPeer(dst_ptr, dst_device, src_ptr, src_device, size);
```

直接指定源和目标 GPU ID，数据经 NVLink（或 PCIe P2P）直接传输，**不经 CPU 内存**。前提是 `cudaDeviceCanAccessPeer` 返回 true。

### 5.2 cudaMemcpy D2D (P2P 已开启)

```c
cudaDeviceEnablePeerAccess(peer_device, 0);  // 先开启 P2P
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);  // 自动走 NVLink
```

一旦启用 P2P，`cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)` 自动选择 NVLink 路径——与 `cudaMemcpyPeer` 无性能差异，但 API 更简洁。

> **易错点**：`cudaDeviceEnablePeerAccess(1, 0)` 必须在正确的 device context 下调用。先 `cudaSetDevice(0)` 再 `enablePeerAccess(1, 0)`——我们第一版代码就在这里翻了车（在 device 1 上调用了对 device 1 的 self-access）。

### 5.3 CPU relay — 无 P2P 的兜底方案

```c
float *host_buf;
cudaMallocHost(&host_buf, size);         // pinned memory
cudaMemcpy(host_buf, d_src, size, cudaMemcpyDeviceToHost);   // GPU→CPU
cudaMemcpy(d_dst, host_buf, size, cudaMemcpyHostToDevice);   // CPU→GPU
```

当 P2P 不可用时（SYS/NODE 拓扑），这是唯一的 GPU 间数据传输路径。pinned memory (`cudaMallocHost`) 是关键——普通 `malloc` 的 pageable memory 无法做异步 H2D/D2H，带宽会再降一个数量级。

**适用场景**：GPU 拓扑为 SYS/NODE（跨 NUMA node 或无 NVLink），或需要兼容性最强的方案。缺点：占用 CPU 内存带宽，延迟高。

#### 5.3.1 SHM/mmap 进阶：跨进程共享场景

当 relay buffer 需要在**多个进程之间共享**（如推理服务的前后端分离），可以用 `mmap` + `cudaHostRegister` 替代 `cudaMallocHost`：

```c
int fd = shm_open("/gpu_buf", O_CREAT | O_RDWR, 0666);
ftruncate(fd, size);
float *buf = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
cudaHostRegister(buf, size, cudaHostRegisterPortable);  // pin mmap'd memory
// ... cudaMemcpyAsync(buf, d_src, size, cudaMemcpyDeviceToHost, stream) ...
```

这在以下场景有实际价值：

| 场景                    | 说明                                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------------------- |
| 多进程共享 relay buffer | 进程 A 将 GPU 数据拷入 SHM → 进程 B 从 SHM 读取，无需 CPU 间拷贝                                        |
| 文件 I/O 优化           | 对 mmap 映射的文件页做 `cudaHostRegister` 后可直接 DMA 到 GPU，绕过 pageable memory 的慢速 staging path |
| 内存不可迁移            | 已有 mmap/大页分配的 buffer 不想重构为 `cudaMallocHost`，注册即可获得 DMA 能力                          |

**A100 实测**：对于大数据块（128 MB），`mmap+cudaHostRegister` 与 `cudaMallocHost` 带宽无差异（均受限于 PCIe Gen4 ~28 GB/s 单向）。SHM 方案的优势不在带宽，而在跨进程共享零拷贝——同一块物理内存可被多个进程的 GPU DMA 引擎直接访问。

| 方法                      | 128 MB 双向带宽 | 跨进程共享                 |
| ------------------------- | --------------- | -------------------------- |
| `cudaMallocHost`          | 5.26 GB/s       | 否（进程私有）             |
| `mmap + cudaHostRegister` | 5.40 GB/s       | **是**（SHM 多进程可访问） |

### 5.4 Zero-Copy — mapped host memory

```c
cudaHostAlloc(&ptr, size, cudaHostAllocPortable | cudaHostAllocMapped);
cudaHostGetDevicePointer(&d_ptr, ptr, 0);  // 获取同一块内存的 device 指针
```

host 内存被映射到 GPU 地址空间，GPU 可直接通过 load/store 访问。**不需要显式 memcpy**——GPU kernel 可以直接读 `d_ptr`。但每次访问都要经 PCIe 往返，延迟远高于显存。

在 128 MB 大块传输测试中带宽仅 2.36 GB/s——因为每次 `cudaMemcpy` 实际上仍触发 PCIe 传输，且 mapped memory 不支持 DMA 批量传输。**Zero-Copy 的优势在小数据、频繁访问场景**，而非大块数据搬运。

### 5.5 Unified Memory — 最省心的方案

```c
cudaMallocManaged(&ptr, size);      // 分配统一内存
cudaMemPrefetchAsync(ptr, size, device_id);  // 提示：请迁移到 GPU 0
```

CUDA 驱动自动管理数据在 CPU 和 GPU 之间的迁移。加了 `cudaMemPrefetchAsync` 后在本次测试中带宽为 2.05 GB/s——与 Zero-Copy 类似，128 MB 数据量下页迁移开销主导。Unified Memory 的编程模型最简单：分配后 CPU 和 GPU 都能直接用同一个指针，适合**快速原型和移植遗留代码**，不适合追求极致性能的数据搬运。

---

## 6. 方法选择决策树

```text
需要 GPU 间搬数据？
├── 拓扑是 NV12/PIX (有 P2P)？
│   └── 用 cudaMemcpy D2D (方法 2) — 一行代码，最快
├── 多进程需要共享 relay buffer？
│   └── 用 mmap + cudaHostRegister (§5.3.1) — 零拷贝跨进程
├── 开销可接受 CPU relay？
│   └── 用 CPU relay (方法 3) — 兼容性最好
├── 数据量 < 1 MB 且频繁访问？
│   └── 用 Zero-Copy (方法 4) — mapped memory 低延迟
├── 快速原型 / 移植老代码？
│   └── 用 Unified Memory (方法 5) — 最省心
└── 追求极致带宽？
    └── 确保 NVLink 正常，用方法 1 或 2
```

---

## 7. 相关文档

- [`08_p2p_bandwidth.md`](08_p2p_bandwidth.md)：P2P 单向带宽 239 GB/s 的详细测试和拓扑分析——本文的 124 GB/s 是双向值
- [`02_pcie_bandwidth_measurement.md`](02_pcie_bandwidth_measurement.md)：PCIe H2D/D2H 单卡带宽 ~28 GB/s——CPU relay 的 H2D 和 D2H 各受此限
- [`03_hbm_bandwidth_test.md`](03_hbm_bandwidth_test.md)：片内 D2D 带宽 ~1188 GB/s——与片间 P2P 的 249 GB/s 形成"片内/片间"层级对比

---

## 参考

- [CUDA Peer-to-Peer Memory Access](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#peer-to-peer-memory-access)
- [CUDA Unified Memory Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)
