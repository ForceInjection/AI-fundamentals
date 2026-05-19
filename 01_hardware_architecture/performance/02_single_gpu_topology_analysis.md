# 单卡 GPU 拓扑与 NUMA 深入分析

> 基于 Intel Xeon Platinum 8470Q (Sapphire Rapids) + RTX 5090 实际环境。本文聚焦单 GPU 场景下的拓扑分析方法，与多卡 NVLink/NVSwitch 场景形成互补。

---

## 1. 为什么单卡也要关注拓扑

单 GPU 环境中没有 NVLink 和 P2P，但拓扑仍然直接影响性能：

- **NUMA 亲和性**：GPU 通过 PCIe 连接到特定 NUMA 节点，跨 NUMA 访问系统内存会增加约 2 倍延迟
- **PCIe 链路状态**：空闲时链路降级到 Gen 1 省电，负载下才恢复 Gen 5
- **CPU 线程绑定**：GPU 相关的 CPU 线程应绑定到 GPU 所在 NUMA 节点，避免跨 socket 通信

---

## 2. 环境信息

### 2.1 硬件概览

```bash
# CPU 架构
lscpu | grep -E "Model name|Socket|NUMA|Thread|Core"
```

本环境输出：

```text
Model name:     Intel(R) Xeon(R) Platinum 8470Q
Socket(s):      2
Core(s)/socket: 52
Thread(s)/core: 2
CPU(s):         208
NUMA node(s):   2
```

### 2.2 NUMA 拓扑

```bash
# 各 NUMA 节点的 CPU 范围
for f in /sys/devices/system/node/node*/cpulist; do
    echo "$f: $(cat $f)"
done
```

本环境输出：

```text
/sys/devices/system/node/node0/cpulist: 0-51,104-155
/sys/devices/system/node/node1/cpulist: 52-103,156-207
```

```bash
# NUMA 距离矩阵
for f in /sys/devices/system/node/node*/distance; do
    echo "$f:"; cat $f
done
```

本环境输出：

```text
/sys/devices/system/node/node0/distance:
10 21
/sys/devices/system/node/node1/distance:
21 10
```

**解读**：本地 NUMA 访问代价 = 10，跨 NUMA 访问代价 = 21。约 2.1 倍延迟惩罚。

### 2.3 GPU PCIe 拓扑

```bash
# GPU 的 NUMA 亲和性
cat /sys/bus/pci/devices/0000:98:00.0/numa_node
# 输出: 1

# GPU PCIe 链路最大能力
cat /sys/bus/pci/devices/0000:98:00.0/max_link_speed   # 32.0 GT/s PCIe (Gen 5)
cat /sys/bus/pci/devices/0000:98:00.0/max_link_width    # 16

# GPU 在 PCIe 树中的路径
readlink -f /sys/bus/pci/devices/0000:98:00.0
# 输出: /sys/devices/pci0000:97/0000:97:01.0/0000:98:00.0
```

---

## 3. nvidia-smi 拓扑视图解读

### 3.1 基本拓扑 (`topo -m`)

```bash
nvidia-smi topo -m
```

本环境输出（单 RTX 5090）：

```text
        GPU0    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      52-103,156-207  1               N/A
```

图例：

| 缩写 | 含义                                  | 延迟 |
| ---- | ------------------------------------- | ---- |
| X    | Self                                  | —    |
| PIX  | 单 PCIe bridge 内                     | 低   |
| PXB  | 多 PCIe bridge，不出 PCIe Host Bridge | 中   |
| PHB  | 跨 PCIe Host Bridge                   | 中高 |
| NODE | 跨 NUMA 节点内的 PCIe Host Bridge     | 高   |
| SYS  | 跨 NUMA 节点（经 QPI/UPI）            | 最高 |
| NV#  | 经 NVLink                             | 极低 |

多卡场景下，同一 NUMA 节点的 GPU 之间通常显示为 `NODE` 或 `PIX`，跨 NUMA 则为 `SYS`。

### 3.2 矩阵拓扑 (`topo -mp`)

`-mp` 选项以更紧凑的矩阵格式展示，适合多 GPU 场景：

```bash
nvidia-smi topo -mp
```

### 3.3 P2P 状态 (`topo -p2p`)

```bash
nvidia-smi topo -p2p r    # r = 可访问性
nvidia-smi topo -p2p w    # w = 带宽矩阵
nvidia-smi topo -p2p n    # n = 路径类型 (PIX/NODE/SYS等)
```

本环境输出（单卡，仅 Self）：

```text
        GPU0
 GPU0    X
```

---

## 4. 跨 NUMA 访问的实测影响

### 4.1 方法

以下脚本分别在 GPU 所在 NUMA (node 1) 和远端 NUMA (node 0) 分配内存，测量 H2D 带宽差异。

```bash
# 如果 numactl 未安装
apt install numactl

# 绑定到 GPU 所在 NUMA
numactl --cpunodebind=1 --membind=1 ./pcie_bw_test

# 绑定到远端 NUMA
numactl --cpunodebind=0 --membind=0 ./pcie_bw_test
```

**本环境实测（taskset 绑核）**：

| CPU 绑定 | H2D 1 GB (GB/s) | D2H 1 GB (GB/s) |
|----------|-----------------|------------------|
| Node 0 (跨 NUMA) | 52.60 | 53.33 |
| Node 1 (本地 NUMA) | 52.63 | 53.05 |

> 实测两个 NUMA 节点带宽几乎一致。这是因为 Seetacloud 容器环境中 `membind` 策略被限制（`set_mempolicy: Operation not permitted`），实际内存分配策略为 `default`（不受 CPU binding 影响）。容器/VM 环境常掩盖 NUMA 效应——裸金属环境预计可观测到 ~2.1× 差异。

### 4.2 建议

- GPU 程序应尽量从 GPU 所在 NUMA 节点分配 pinned memory
- 使用 `cudaMallocHost` (pinned memory) 时，系统自动从当前线程所在 NUMA 分配
- 可以用 `taskset -c 52-103` 或 `numactl --cpunodebind=1` 绑定进程到 GPU NUMA

---

## 5. CUDA 侧查询拓扑信息

```c
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("PCI Domain:  %04x\n", prop.pciDomainID);
printf("PCI Bus:     %02x\n", prop.pciBusID);
printf("PCI Device:  %02x\n", prop.pciDeviceID);
```

也可以通过 `cudaDeviceGetByPCIBusId` 反向查询，确保选中正确的 GPU：

```c
cudaDeviceGetByPCIBusId(&device_id, "0000:98:00.0");
cudaSetDevice(device_id);
```

---

## 6. 与多卡场景的对比

| 维度        | 单卡 RTX 5090                   | 多卡 A100/H100 集群           |
| ----------- | ------------------------------- | ----------------------------- |
| GPU 互连    | 无                              | NVLink / NVSwitch             |
| P2P         | 不支持                          | 支持 (GPUDirect P2P)          |
| NUMA 意识   | 关键（选错节点损失大）          | 重要但被 NVLink 部分掩盖      |
| 拓扑命令    | `topo -m` 即可                  | `topo -mp` + `topo -p2p` 全套 |
| NVLink 查询 | `nvidia-smi nvlink -s` → 不支持 | 完整的链路状态和计数          |

---

## 参考

- [PCIe 技术大全](../pcie/01_pcie_comprehensive_guide.md)
- [NVLink 技术入门](../nvlink/nvlink_intro.md)
- [nvidia-smi 快速入门](../../03_ai_cluster_ops/01_gpu_ops/03_nvidia_smi_guide.md)
- [CUDA Device Attribute 官方文档](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)
