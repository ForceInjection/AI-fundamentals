# 多 PCIe 根总线段与 NUMA 映射

> 基于双路 Sapphire Rapids（domain 0 内 24 个根总线段，2 个 NUMA node）。详解 PCIe 根总线段与 NUMA 的对应关系、如何查询，以及跨 NUMA 访问 NVMe 的性能影响。实测环境为双路 Sapphire Rapids + RTX 5090 + 3 块 NVMe，实测时间未记录（按 RTX 5090 上市时间推断约 2025 年），内核与驱动版本未留存记录。

---

## 1. 背景：Sapphire Rapids 的 PCIe 架构

先澄清一个常见的误读：Linux 设备名 `0000:98:00.0` 中，`0000` 是 PCI domain 号，`98` 是 bus 号。Sapphire Rapids 平台上的设备全部位于 domain 0，并不存在「GPU 在 domain 97」这回事；`pci0000:97` 指的是 domain 0 内以 bus 0x97 为根的一段总线，即一个 host bridge（RC stack）所辖的 bus 号段。

| 代际                | PCIe 拓扑                      | 特点                                         |
| ------------------- | ------------------------------ | -------------------------------------------- |
| Ice Lake            | 单 RC per socket, 多 Root Port | 设备集中在一个 bus 号段                      |
| **Sapphire Rapids** | **多 RC per socket**           | 每个 RC stack 占一段 bus 号段，设备分布更分散 |

每个 socket 有多个独立的 RC stack，各自占用一段 bus 号。本文环境的 24 个 `pci0000:xx` 目录是 24 个根总线段，而不是 24 个 PCIe domain；GPU 在 bus 0x98 段，NVMe 在 bus 0xd7 段。

---

## 2. 本环境根总线段分布

```bash
# 列出各根总线段及设备数
for d in /sys/devices/pci*; do
    count=$(ls "$d" 2>/dev/null | grep -c "^0000:")
    echo "$(basename $d): $count devices"
done
```

| 根总线段           | 设备数  | 推测用途           | NUMA  |
| ------------------ | ------- | ------------------ | ----- |
| 00                 | 18      | 芯片组, VGA        | —     |
| 7e, 7f             | 37, 122 | Socket 0 I/O       | 0     |
| 97                 | **5**   | **GPU** (RTX 5090) | **1** |
| d7                 | 3       | NVMe 阵列 (3 块)   | 1     |
| fe, ff             | 37, 122 | Socket 1 I/O       | 1     |
| 其他 (15/26/37/ek) | 4-8     | 系统外设           | 分布  |

---

## 3. 查询根总线段的 NUMA 亲和性

### 3.1 sysfs 方法

```bash
# 方法 1: host bridge 的 numa_node（需内核支持）
cat /sys/devices/pci0000:97/numa_node 2>/dev/null

# 方法 2: 通过段内任意设备查询
cat /sys/bus/pci/devices/0000:98:00.0/numa_node    # GPU → 1
```

> 本环境 host bridge 级 `numa_node` 返回空——内核版本未暴露。但设备级正常。

### 3.2 查询各段内的代表性设备

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

Sapphire Rapids 按 socket 划分 bus 号段，BDF 中的 bus 号直接给出 socket 归属：

| Bus Pattern             | Socket/NUMA            |
| ----------------------- | ---------------------- |
| `bus 0x00` ~ `bus 0x7f` | Socket 0 (NUMA node 0) |
| `bus 0x80` ~ `bus 0xff` | Socket 1 (NUMA node 1) |

GPU BDF = `0000:98:00.0` → bus 0x98 > 0x7f → Socket 1 → NUMA node 1。与 `numa_node` 返回的 `1` 一致。

### 3.4 NVMe 的 NUMA 位置

```bash
# 本环境 NVMe (3 块均在 bus 0xd7 段 / Socket 1)
for nvme in /sys/block/nvme*/device; do
    bdf=$(readlink -f "$nvme" 2>/dev/null | grep -oP "0000:[0-9a-f:]+" | head -1)
    numa=$(cat /sys/bus/pci/devices/$bdf/numa_node 2>/dev/null)
    echo "$(basename $(dirname $nvme)): $bdf, NUMA=$numa"
done
```

本环境所有 3 块 NVMe (nvme0-2n1) 均在 bus 0xd7 段，与 GPU (bus 0x98 段) 同属 Socket 1。

GPU 和 NVMe 在同一 NUMA node，GDS (GPU Direct Storage) 场景下数据路径全部在本地 socket 内，无需跨 UPI（Ultra Path Interconnect，Intel 双路 CPU 之间的点对点互连总线）。

---

## 4. 跨 NUMA PCIe 访问的性能影响

### 4.1 场景分析

```text
Socket 0 (NUMA 0)          Socket 1 (NUMA 1)
  ├── NVMe (bus 7f 段) ←?→   ├── GPU (bus 98 段)
  └── ...                    └── NVMe (bus d7 段) ← ✓ 本地
```

假设某块 NVMe 挂在 Socket 0（左列），GPU 在 Socket 1，就是一条跨 NUMA 路径。本环境 GPU 和 NVMe 同在 Socket 1（右列 ✓ 标记），GDS 无跨 NUMA 开销。

### 4.2 跨 NUMA 的 PCIe 路径

如果 GPU (Socket 1) 访问 Socket 0 的 NVMe (bus 7f 段)：

```text
NVMe (S0) → RC (bus 7f 段) → UPI → RC (bus 98 段) → Bridge → GPU (S1)
```

经过 UPI 跨 socket 链路，NUMA distance 21 vs 10，预计增加 ~2× 延迟。但由于 NVMe I/O 延迟本就 ~10-100 μs，额外的 NUMA 延迟 (~100ns) 影响很小。

---

## 5. PCIe 拓扑与 GPU 编程

### 5.1 确认设备选对总线段

CUDA 的 `cudaDeviceGetByPCIBusId` 可确保选中特定 BDF 的 GPU：

```c
int device_id;
cudaDeviceGetByPCIBusId(&device_id, "0000:98:00.0");
cudaSetDevice(device_id);
```

这在多根总线段环境中尤其重要，多张 GPU 可能分散在不同的 bus 号段。

### 5.2 跨 host bridge 的 P2P 可达性

P2P DMA 在同一个 host bridge（RC stack）辖属范围内通常畅通。跨 host bridge 的 P2P（如 GPU 访问另一个 RC stack 下的 NVMe）依赖系统 firmware 的 ACPI DMAR 表支持，云环境可能受限。

---

## 6. 总结

| 问题                   | 本环境答案                                |
| ---------------------- | ----------------------------------------- |
| GPU 在哪个 NUMA?       | Node 1 (bus 0x98 > 0x7f)                  |
| NVMe 在哪个 NUMA?      | Node 1 (bus 0xd7 段)，与 GPU 同 socket    |
| 有跨 NUMA PCIe 访问吗? | 本环境无，GPU 和 NVMe 在同一 socket       |
| BDF 编码规则           | bus 0x00-0x7f = S0, bus 0x80-0xff = S1    |
| 多根总线段影响编程吗?  | 需 `cudaDeviceGetByPCIBusId` 确保选对设备 |

---

## 参考

- [单卡 GPU 拓扑与 NUMA 分析](02_single_gpu_topology_analysis.md)
- [PCIe 拓扑层次](../pcie/06_pcie_topology_hierarchy.md)
- [PCIe Switch 识别与验证](../pcie/07_pcie_switch_vs_bridge.md)
- [CUDA NUMA API 编程实践](../../02_gpu_programming/02_cuda/05_cuda_numa_api.md)
