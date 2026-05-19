# GPU Direct Storage (GDS) 基础

> GDS 是 GPUDirect 家族的存储成员，允许 GPU 通过 PCIe 直接读写 NVMe 存储，绕过 CPU 内存中转。基于 RTX 5090 + CUDA 12.8 (GDS 1.13.1) 环境。

---

## 1. GPUDirect 家族全景

| 技术                  | 解决的问题      | 物理路径               |
| --------------------- | --------------- | ---------------------- |
| **GPUDirect P2P**     | GPU↔GPU 直通    | PCIe / NVLink          |
| **GPUDirect RDMA**    | GPU↔网卡 直通   | PCIe + InfiniBand/RoCE |
| **GPUDirect Storage** | GPU↔NVMe 直通   | PCIe + NVMe            |
| **GPUDirect Video**   | GPU↔采集卡 直通 | PCIe                   |

GDS 是最晚加入的成员 (CUDA 11.4+)。它解决一个具体问题：训练数据从 NVMe 加载到 GPU 时，传统路径是 `NVMe → CPU RAM → GPU`，GDS 是 `NVMe → GPU`，省掉 CPU 中转。

```text
传统:   NVMe ──PCIe──▶ CPU RAM ──PCIe──▶ GPU VRAM   (2 次 PCIe 传输)
GDS:    NVMe ──PCIe──────────▶ GPU VRAM              (1 次 PCIe 传输)
```

---

## 2. 前置条件

```bash
# 1. CUDA 11.4+
nvcc --version

# 2. GDS 包安装
ls /usr/local/cuda/gds/README

# 3. cuFile 库存在
ls /usr/local/cuda/targets/x86_64-linux/lib/libcufile.so*

# 4. NVMe 驱动器
lsblk | grep nvme
# 预期: 至少有一个 nvmeXnY 设备

# 5. nvidia-fs.ko 内核模块（可选，优化 direct I/O）
lsmod | grep nvidia_fs
```

### 2.1 本环境确认

```bash
ls /sys/block/ | grep nvme
# nvme0n1  nvme1n1  nvme2n1    ← 3 块 NVMe 块设备

cat /usr/local/cuda/gds/README | head -1
# GDS Version: 1.13.1

ls /usr/local/cuda/targets/x86_64-linux/lib/libcufile.so*
# libcufile.so  libcufile_rdma.so  ← cuFile + RDMA 支持
```

> GDS 要求 NVMe 格式化为文件系统并挂载（如 ext4/xfs + `mount`），仅作为块设备存在 (`/dev/nvme*`) 无法直接使用 `cuFileRead`。

---

## 3. cuFile API 基础

GDS 通过 `libcufile` 暴露 API，核心流程：

```c
#include <cufile.h>

// 1. 打开 NVMe 文件 (标准 POSIX open，需 O_DIRECT)
int fd = open("/mnt/nvme/data.bin", O_RDONLY | O_DIRECT);

// 2. 注册文件为 GPU 可直接访问
CUfileDescr_t desc = {};
desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
desc.cookie = (void*)(uintptr_t)fd;
CUfileHandle_t fh;
cuFileHandleRegister(&fh, &desc);

// 3. 注册 GPU buffer
cuFileBufRegister(gpu_buffer, size, 0);

// 4. GPU 直接读取 NVMe（异步）
cuFileRead(fh, gpu_buffer, size, offset, 0);

// 5. 清理
cuFileBufDeregister(gpu_buffer);
cuFileHandleDeregister(fh);
close(fd);
```

编译：

```bash
nvcc -I/usr/local/cuda/include \
     -L/usr/local/cuda/targets/x86_64-linux/lib \
     -lcufile \
     -o gds_read gds_read.cu
```

---

## 4. 性能对比：传统 vs GDS

| 路径             | PCIe 穿越次数 | 有效带宽  | CPU 占用      |
| ---------------- | ------------- | --------- | ------------- |
| NVMe → CPU → GPU | 2             | ~½ 理论值 | 高 (memcpy)   |
| NVMe → GPU (GDS) | 1             | ~理论值   | 低 (DMA only) |

但由于 PCIe 带宽共享（GPU 和 NVMe 通常在同一 PCIe root complex 下），实际加速取决于拓扑。如果 NVMe 和 GPU 连接到同一 PCIe switch 且支持 P2P，加速最明显。

---

## 5. 常见限制

| 限制                   | 说明                                             |
| ---------------------- | ------------------------------------------------ |
| **NVMe 必须直接挂载**  | 网络存储 (NFS/Ceph) 不支持 GDS                   |
| **需 O_DIRECT**        | 绕过 page cache，直接 DMA                        |
| **文件对齐**           | 偏移和大小需对齐到 4 KB                          |
| **nvidia-fs 内核模块** | 可选，启用后性能更好（绕过 VFS 层）              |
| **消费级 GPU**         | 支持 GDS 但性能低于数据中心 GPU（BAR1 窗口限制） |
| **PCIe topology**      | GPU 和 NVMe 需在同一 PCIe domain 且支持 P2P      |

---

## 6. 适用场景

| 场景                   | GDS 适用性  | 说明                        |
| ---------------------- | ----------- | --------------------------- |
| 大模型 checkpoint 加载 | ✅ 强推荐   | 数十 GB 文件直接加载到 GPU  |
| 训练数据流式读取       | ✅ 推荐     | 绕过 CPU RAM 中转           |
| KV Cache 磁盘卸载      | ✅ 适用     | 与 LMCache 等方案配合       |
| 小文件随机读取         | ❌ 不适用   | 每次 open/register 开销过大 |
| 已有 page cache 热点   | ⚠️ 可能更慢 | 绕过 cache 反而增加延迟     |

---

## 7. 快速测试脚本

```bash
cat > test_gds.sh << 'EOF'
#!/bin/bash
echo "=== GDS Environment Check ==="
echo "CUDA: $(nvcc --version 2>/dev/null | grep release)"
echo "GDS:  $(cat /usr/local/cuda/gds/README 2>/dev/null | head -1)"
echo "NVMe devices:"
lsblk 2>/dev/null | grep nvme || echo "  No NVMe found"
echo "libcufile:"
ls /usr/local/cuda/targets/x86_64-linux/lib/libcufile.so* 2>/dev/null || echo "  Not found"
echo "nvidia-fs module:"
lsmod 2>/dev/null | grep nvidia_fs || echo "  Not loaded (optional)"
EOF
chmod +x test_gds.sh
./test_gds.sh
```

---

## 参考

- [NVIDIA GPUDirect Storage Documentation](https://docs.nvidia.com/gpudirect-storage/)
- [cuFile API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/)
- [GPUDirect RDMA 技术详解](01_gpudirect_technology.md)
- [GPUDirect P2P 技术详解](02_gpudirect_p2p.md)
- [PCIe P2PDMA 技术介绍](../pcie/02_p2pdma_technology.md)
