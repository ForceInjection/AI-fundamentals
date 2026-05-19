# 多 PCIe Domain 与 NUMA 映射

> 基于双路 Sapphire Rapids (24 个 PCIe domain, 2 个 NUMA node)。详解 PCIe domain 与 NUMA 的对应关系、如何查询、以及跨 NUMA 访问 NVMe 的性能影响。

---

## 1. 背景：Sapphire Rapids 的 PCIe 架构

Intel Sapphire Rapids (SPR) 的 PCIe 架构与之前代际显著不同：

| 代际 | PCIe 拓扑 | 特点 |
|------|----------|------|
| Ice Lake | 单 RC per socket, 多 Root Port | 所有设备共享同一 domain |
| **Sapphire Rapids** | **多 RC per socket** | 每个 RC 有独立 domain number，设备分布更分散 |

每个 socket 有多个独立的 Root Complex，各自分配唯一的 domain number。这增加了拓扑复杂度——GPU 可能在 domain 97，NVMe 在 domain 7f。

---

## 2. 本环境 Domain 分布

```bash
# 列出所有 domain 及设备数
for d in /sys/devices/pci*; do
    count=$(ls "$d" 2>/dev/null | grep -c "^0000:")
    echo "$(basename $d): $count devices"
done
```

| Domain | 设备数 | 推测用途 | NUMA |
|--------|--------|----------|------|
| 00 | 18 | 芯片组, VGA | — |
| 7e, 7f | 37, 122 | Socket 0 I/O, NVMe 阵列 | 0 |
| 97 | **5** | **GPU** (RTX 5090) | **1** |
| fe, ff | 37, 122 | Socket 1 I/O, NVMe 阵列 | 1 |
| 其他 (15/26/37/ek) | 4-8 | 系统外设 | 分布 |

---

## 3. 查询 Domain 的 NUMA 亲和性

### 3.1 sysfs 方法

```bash
# 方法 1: domain 的 numa_node（需内核支持）
cat /sys/devices/pci0000:97/numa_node 2>/dev/null

# 方法 2: 通过 domain 内任意设备查询
cat /sys/bus/pci/devices/0000:98:00.0/numa_node    # GPU → 1
```

> 本环境 domain 级 `numa_node` 返回空——内核版本未暴露。但设备级正常。

### 3.2 查询所有 domain 内的代表性设备

```bash
for d in /sys/bus/pci/devices/0000:7f:*/0000:* 0000:97:*/0000:* 0000:ff:*/0000:*; do
    [ -d "$d" ] || continue
    bdf=$(basename "$d")
    cls=$(cat "$d/class" 2>/dev/null)
    numa=$(cat "$d/numa_node" 2>/dev/null)
    echo "$bdf | class=$cls | NUMA=$numa"
done 2>/dev/null | head -20
```

### 3.3 通过设备 BDF 规律推断

Sapphire Rapids 的 BDF 编码包含 NUMA 信息：

| BDF Pattern | Socket/NUMA |
|-------------|-------------|
| `0000:00:*` ~ `0000:7f:*` | Socket 0 (NUMA node 0) |
| `0000:80:*` ~ `0000:ff:*` | Socket 1 (NUMA node 1) |

GPU BDF = `0000:98:00.0` → bus 0x98 > 0x7f → Socket 1 → NUMA node 1。与 `numa_node` 返回的 `1` 一致。

### 3.4 NVMe 的 NUMA 位置

```bash
# 本环境 NVMe (3 块均在 domain d7 / Socket 1)
for nvme in /sys/block/nvme*/device; do
    bdf=$(readlink -f "$nvme" 2>/dev/null | grep -oP "0000:[0-9a-f:]+" | head -1)
    numa=$(cat /sys/bus/pci/devices/$bdf/numa_node 2>/dev/null)
    echo "$(basename $(dirname $nvme)): $bdf, NUMA=$numa"
done
```

本环境所有 3 块 NVMe (nvme0-2n1) 均在 domain d7，与 GPU (domain 97) 同属 Socket 1。

**这意味着**：GPU 和 NVMe 在同一 NUMA node，GDS (GPU Direct Storage) 场景下数据路径全部在本地 socket 内——无需跨 UPI（Ultra Path Interconnect，Intel 双路 CPU 之间的点对点互连总线）。

---

## 4. 跨 NUMA PCIe 访问的性能影响

### 4.1 场景分析

```
Socket 0 (NUMA 0)          Socket 1 (NUMA 1)
  ├── NVMe (domain 7f) ←?→   ├── GPU (domain 97)
  └── NVMe (domain 7f)       └── NVMe (domain d7) ← ✓ 本地
```

本环境 GPU 和 NVMe 同在 Socket 1 → GDS 无跨 NUMA 开销。

### 4.2 跨 NUMA 的 PCIe 路径

如果 GPU (Socket 1) 访问 Socket 0 的 NVMe (domain 7f)：

```
NVMe (S0) → RC (domain 7f) → UPI → RC (domain 97) → Bridge → GPU (S1)
```

经过 UPI 跨 socket 链路，NUMA distance 21 vs 10，预计增加 ~2× 延迟。但由于 NVMe I/O 延迟本就 ~10-100 μs，额外的 NUMA 延迟 (~100ns) 影响很小。

---

## 5. PCIe Domain 与 GPU 编程

### 5.1 确认设备选对 domain

CUDA 的 `cudaDeviceGetByPCIBusId` 可确保选中特定 BDF 的 GPU：

```c
int device_id;
cudaDeviceGetByPCIBusId(&device_id, "0000:98:00.0");
cudaSetDevice(device_id);
```

这在多 domain 环境中尤其重要——多张 GPU 可能分散在不同 domain。

### 5.2 Domain 与 DMA 可达性

DMA 通常可在同 domain 内自由进行。跨 domain 的 DMA（如 GPU 访问另一个 domain 的 NVMe）需要系统 firmware 的 ACPI DMAR 表支持，云环境可能受限。

---

## 6. 总结

| 问题 | 本环境答案 |
|------|-----------|
| GPU 在哪个 NUMA? | Node 1 (BDF 0x98 > 0x7f) |
| NVMe 在哪个 NUMA? | Node 1 (domain d7)，与 GPU 同 socket |
| 有跨 NUMA PCIe 访问吗? | 本环境无——GPU 和 NVMe 在同一 socket |
| BDF 编码规则 | 0x00-0x7f = S0, 0x80-0xff = S1 |
| 多 domain 影响编程吗? | 需 `cudaDeviceGetByPCIBusId` 确保选对设备 |

---

## 参考

- [单卡 GPU 拓扑与 NUMA 深入分析](02_single_gpu_topology_analysis.md)
- [PCIe 拓扑层次](../pcie/06_pcie_topology_hierarchy.md)
- [PCIe Switch 识别与验证](../pcie/07_pcie_switch_vs_bridge.md)
- [CUDA NUMA API 编程实践](../../02_gpu_programming/02_cuda/05_cuda_numa_api.md)
