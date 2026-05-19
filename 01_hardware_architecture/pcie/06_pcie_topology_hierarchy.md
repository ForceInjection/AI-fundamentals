# PCIe 拓扑层次：Root Complex、Switch、Bridge 与 Device

> 基于双路 Sapphire Rapids (24 个 PCIe domain) + RTX 5090 实测。从 sysfs 出发，逐层解析 PCIe 拓扑的四个核心组件。

---

## 1. 四层模型

```text
CPU Socket
  └── Root Complex (RC)
        ├── Root Port 0 → PCIe Bridge → Device (GPU)
        ├── Root Port 1 → Device (NVMe)
        ├── Root Port 2 → PCIe Switch → Device × N (NVMe阵列)
        └── Root Port 3 → Device (网卡)
```

| 组件                  | class code                       | 职责                                                     |
| --------------------- | -------------------------------- | -------------------------------------------------------- |
| **Root Complex (RC)** | —                                | CPU 内部，连接 CPU 与 PCIe 总线，每个 domain 对应一个 RC |
| **Root Port**         | 0x060400                         | RC 的下行端口，一个 domain 可有多个 Root Port            |
| **PCIe Switch**       | 0x060400 (upstream)              | 扩展 PCIe 端口：1 个上行 + N 个下行端口                  |
| **PCIe Bridge**       | 0x060400                         | 简单的 1:1 桥接，无端口扩展                              |
| **Endpoint**          | 0x030000 (VGA) / 0x010802 (NVMe) | 终端设备：GPU、NVMe、网卡等                              |

> Switch 和 Bridge 的 class code 相同 (0x060400)—区别在于端口数量和 sysfs 子树结构。

---

## 2. sysfs 识别各层

### 2.1 识别 PCIe Domains (Root Complex)

每个 `pci0000:XX` 目录是一个独立的 PCIe domain，对应一个 Root Complex：

```bash
ls /sys/devices/ | grep pci
```

本环境输出（24 个 domain）：

```text
pci0000:00  pci0000:15  pci0000:26  ...  pci0000:7f  pci0000:80
pci0000:97  pci0000:a7  ...  pci0000:fe  pci0000:ff
```

```bash
# 各 domain 的设备数量
for d in /sys/devices/pci*; do
    count=$(ls "$d" 2>/dev/null | grep -c "^0000:")
    echo "$(basename $d): $count devices"
done
```

本环境（部分）：

```text
pci0000:00: 18 devices    ← 芯片组/VGA
pci0000:7f: 122 devices   ← Socket 0 高密度 I/O
pci0000:97: 5 devices     ← GPU 所在 domain
pci0000:ff: 122 devices   ← Socket 1 高密度 I/O
```

### 2.2 识别 Bridge vs Switch

Bridge 下游只有 1 个设备；Switch 有多个下游端口：

```bash
# 本环境：GPU 经单 Bridge 连接
ls /sys/devices/pci0000:97/0000:97:01.0/ | grep "^0000:"
# 0000:98:00.0           ← 仅 1 个下游设备 = Bridge，非 Switch

# Bridge class code
cat /sys/devices/pci0000:97/0000:97:01.0/class
# 0x060400               ← PCI-to-PCI bridge
```

如果有 Switch，会看到：

```text
# 示例（非本环境）：Switch 有多个下游端口
ls /sys/devices/pci0000:7f/0000:7f:00.0/ | grep "^0000:"
# 0000:80:00.0  0000:81:00.0  0000:82:00.0  ...  ← 多个下游设备
```

### 2.3 追踪设备完整路径

```bash
readlink -f /sys/bus/pci/devices/0000:98:00.0
# /sys/devices/pci0000:97/0000:97:01.0/0000:98:00.0
#                ↑RC      ↑Bridge       ↑GPU
```

解析：`pci0000:97` (Root Complex) → `0000:97:01.0` (Bridge, Intel 0x352a) → `0000:98:00.0` (GPU)。

---

## 3. 本环境完整拓扑

```text
Socket 0 (NUMA node 0)
├── pci0000:00  18 devices  (chipset, VGA, legacy)
├── pci0000:7e  37 devices  (I/O subsystem)
└── pci0000:7f  122 devices (高密度 I/O)

Socket 1 (NUMA node 1)
├── pci0000:97   5 devices  (GPU domain)
│     └── 97:01.0 [Bridge, Intel 0x352a]
│           └── 98:00.0 [RTX 5090, NVIDIA 0x2b85]
├── pci0000:d7          (NVMe domain: nvme0n1/1n1/2n1)
├── pci0000:fe  37 devices  (I/O subsystem)
└── pci0000:ff  122 devices (高密度 I/O)
```

关键观察：

- **GPU 独占一个 domain** (pci0000:97)，无其他高速设备共享 RC——这是最优拓扑
- **NVMe 分布在 domain d7**（3 块 NVMe，`nvme0-2n1`，与 GPU 同在 Socket 1 / NUMA node 1）和 domain 7f/ff（高密度 I/O，各 122 devices）
- **GPU 与 3 块 NVMe 在同一个 Socket**：GDS (GPU Direct Storage) 的 NVMe↔GPU 数据路径在本地 socket 内完成，无跨 UPI 开销，参见 [GPU Direct Storage 基础](../gpudirect/03_gds_basics.md)。跨 socket 需经过 UPI（Ultra Path Interconnect，Intel 双路 CPU 之间的点对点互连总线，NUMA distance 21 vs 本地 10），会增加 ~2× 延迟

---

## 4. 识别 Root Port vs Switch Port

```bash
# Root Port 通常在 domain 根目录下 (BDF 的 device=00)
ls /sys/devices/pci0000:97/ | grep "0000:97:00"
# 0000:97:00.0  0000:97:00.1  0000:97:00.2  0000:97:00.4
# 这些是 Root Port (在 domain 的 bus 0 上)

# Bridge 在 Root Port 下游 (bus 01)
ls /sys/devices/pci0000:97/ | grep "0000:97:01"
# 0000:97:01.0  ← Bridge，连接在 bus 1 上
```

Root Port 的特征：BDF 中 bus = domain bus (通常是 0x97)，device = 0x00。

---

## 5. 查看 Bridge 的上下游

```bash
# Bridge 的上游 (指向 Root Port)
# 通过 PCIe 拓扑推断：97:01.0 的上游是 97:00.x (某个 Root Port)

# Bridge 的下游设备
ls /sys/devices/pci0000:97/0000:97:01.0/ | grep "0000:"
# 0000:98:00.0  ← GPU

# 检查下游是否有更多 bridge (级联)
cat /sys/bus/pci/devices/0000:98:00.0/class
# 0x030000  ← VGA controller: 这是 Endpoint，不再级联
```

---

## 6. 总结

| 层级         | 本环境                    | 如何识别                                |
| ------------ | ------------------------- | --------------------------------------- |
| Root Complex | 24 个 domain (pci0000:XX) | `/sys/devices/pci*` 的数量              |
| Root Port    | domain 内 BDF device=00   | `class = 0x060400` + bus=domain bus     |
| PCIe Switch  | **不存在**                | 下游有 ≥2 个 device 的 bridge           |
| PCIe Bridge  | 97:01.0 (Intel 0x352a)    | `class = 0x060400` + 下游仅 1 个 device |
| Endpoint     | 98:00.0 (RTX 5090)        | class 非 0x060400/0x060000              |

GPU 路径为最简拓扑：Root Complex → Bridge → GPU，无 Switch 引入额外延迟，无同级设备竞争带宽。

---

## 参考

- [PCIe 技术大全](01_pcie_comprehensive_guide.md)
- [PCIe 拓扑可视化](03_pcie_topology_visualization.md)
- [PCIe AER 错误监控](04_pcie_aer_monitoring.md)
