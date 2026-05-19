# PCIe 拓扑可视化：从 sysfs 到 nvidia-smi

> 基于 Intel Xeon Platinum 8470Q + RTX 5090 实际环境。`lspci` 需要安装 `pciutils` 且在新环境中可能不可用——本文展示通过 sysfs 和 nvidia-smi 推断 PCIe 拓扑的方法。

---

## 1. 快速路线

| 需求              | 工具                  | 说明                                       |
| ----------------- | --------------------- | ------------------------------------------ |
| 列出所有 PCI 设备 | `lspci`               | 需安装 `pciutils`                          |
| 树状拓扑          | `lspci -t`            | 显示 PCIe 树结构                           |
| GPU 链路状态      | sysfs 或 `nvidia-smi` | 零依赖                                     |
| NUMA 亲和性       | sysfs                 | `/sys/bus/pci/devices/<bdf>/numa_node`     |
| 链路速度和宽度    | sysfs                 | `current_link_speed`, `current_link_width` |

---

## 2. sysfs 读取 PCIe 信息

### 2.1 列出所有 PCI 设备

```bash
ls /sys/bus/pci/devices/
```

本环境输出（部分）：

```text
0000:00:00.0
0000:00:00.1
0000:00:00.2
...
0000:97:01.0    ← PCIe bridge
0000:98:00.0    ← GPU (RTX 5090)
```

### 2.2 GPU PCIe 关键信息

```bash
GPU_BDF="0000:98:00.0"   # Bus:Device.Function

# 厂商和设备 ID
cat /sys/bus/pci/devices/$GPU_BDF/vendor   # 0x10de (NVIDIA)
cat /sys/bus/pci/devices/$GPU_BDF/device   # 0x2b85

# 设备类别
cat /sys/bus/pci/devices/$GPU_BDF/class    # 0x030000 (VGA display controller)

# NUMA 亲和性
cat /sys/bus/pci/devices/$GPU_BDF/numa_node   # 1

# 链路状态
cat /sys/bus/pci/devices/$GPU_BDF/current_link_speed   # 2.5 GT/s (Gen 1)
cat /sys/bus/pci/devices/$GPU_BDF/current_link_width   # 16
cat /sys/bus/pci/devices/$GPU_BDF/max_link_speed       # 32.0 GT/s (Gen 5)
cat /sys/bus/pci/devices/$GPU_BDF/max_link_width       # 16
```

### 2.3 完整 PCIe 路径

```bash
readlink -f /sys/bus/pci/devices/0000:98:00.0
# 输出: /sys/devices/pci0000:97/0000:97:01.0/0000:98:00.0
```

路径解读：

```text
pci0000:97              ← PCI domain 0000, bus 97 (Root Complex)
  └── 0000:97:01.0      ← PCIe Bridge (bus 97, device 01, function 0)
        └── 0000:98:00.0 ← GPU (bus 98, device 00, function 0)
```

---

## 3. PCIe 树结构分析

### 3.1 理解 sysfs 路径

sysfs 路径体现了 PCIe 层次结构。以上述路径为例：

```text
pci0000:97          → Root Port (CPU 的 PCIe Root Complex)
  0000:97:01.0      → PCIe Switch/Bridge 的上游端口
    0000:98:00.0    → GPU (下游设备)
```

### 3.2 检查是否有 PCIe Switch

```bash
# 查看 bridge 的设备类别
cat /sys/bus/pci/devices/0000:97:01.0/class
# 0x060400 = PCI-to-PCI bridge (标准 PCIe bridge)
```

### 3.3 查找同桥下的其他设备

```bash
# 列出与 GPU 共享同一上游 bridge 的设备
GPU_PATH="/sys/devices/pci0000:97/0000:97:01.0"
ls "$GPU_PATH" | grep "0000:"
# 输出: 0000:98:00.0  (可能还有 0000:98:00.1 = GPU audio function)
```

---

## 4. 与 nvidia-smi 拓扑对照

### 4.1 sysfs 信息 → nvidia-smi 列

| sysfs                      | nvidia-smi 对应                     |
| -------------------------- | ----------------------------------- |
| `numa_node`                | `topo -m` 的 NUMA Affinity          |
| `current_link_speed/width` | `--query-gpu=pcie.link.gen.current` |
| `max_link_speed/width`     | `--query-gpu=pcie.link.gen.max`     |
| `vendor/device`            | `--query-gpu=pci.device_id`         |
| BDF (bus:device.function)  | `--query-gpu=pci.bus_id`            |

### 4.2 对照表示例 (本环境)

| 信息来源           | 值                      |
| ------------------ | ----------------------- |
| sysfs BDF          | `0000:98:00.0`          |
| nvidia-smi BDF     | `00000000:98:00.0`      |
| sysfs NUMA         | 1                       |
| nvidia-smi NUMA    | 1 (CPUs 52-103,156-207) |
| sysfs max speed    | 32.0 GT/s (Gen 5)       |
| nvidia-smi max gen | 5                       |

### 4.3 完整的 PCIe 拓扑验证脚本

```bash
#!/bin/bash
# 从 sysfs 和 nvidia-smi 交叉验证 GPU PCIe 拓扑
GPU_BDF="0000:98:00.0"

echo "=== GPU: $GPU_BDF ==="
echo "PCIe path:  $(readlink -f /sys/bus/pci/devices/$GPU_BDF)"
echo "NUMA node:  $(cat /sys/bus/pci/devices/$GPU_BDF/numa_node)"
echo "Max speed:  $(cat /sys/bus/pci/devices/$GPU_BDF/max_link_speed)"
echo "Current:    $(cat /sys/bus/pci/devices/$GPU_BDF/current_link_speed)"
echo "Max width:  $(cat /sys/bus/pci/devices/$GPU_BDF/max_link_width)"
echo "Current:    $(cat /sys/bus/pci/devices/$GPU_BDF/current_link_width)"
echo
echo "=== nvidia-smi verify ==="
nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max --format=csv
```

---

## 5. lspci 安装与替代

### 5.1 安装 lspci

```bash
apt install pciutils
lspci | grep -i nvidia
lspci -t           # PCIe 树
lspci -vvv -s 98:00.0  # GPU 详细
```

### 5.2 或使用 sysfs（零依赖）

如果无法/不想安装 lspci，sysfs 提供所有必要信息：

```bash
# 列出所有 NVIDIA 设备 (vendor 0x10de)
for dev in /sys/bus/pci/devices/*; do
    v=$(cat "$dev/vendor" 2>/dev/null)
    if [ "$v" = "0x10de" ]; then
        echo "$(basename $dev): vendor=$v device=$(cat $dev/device)"
    fi
done
```

---

## 6. 与多卡/HPC 场景对比

| 维度         | 单卡 RTX 5090          | 多卡 DGX/集群                  |
| ------------ | ---------------------- | ------------------------------ |
| PCIe 树深度  | 浅 (RC → Bridge → GPU) | 深 (RC → Switch → 多层 → GPU)  |
| sysfs 路径数 | 1                      | N 个                           |
| 拓扑关键信息 | NUMA node              | PCIe Switch affinity           |
| 推荐工具     | sysfs + nvidia-smi     | lspci -t + nvidia-smi topo -mp |

---

## 参考

- [PCIe 技术大全](01_pcie_comprehensive_guide.md)
- [PCIe P2PDMA 技术介绍](02_p2pdma_technology.md)
- [单卡 GPU 拓扑与 NUMA 深入分析](../performance/02_single_gpu_topology_analysis.md)
