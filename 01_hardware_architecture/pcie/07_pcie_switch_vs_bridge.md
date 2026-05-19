# PCIe Switch 识别与验证

> 基于双路 Sapphire Rapids 实测。GPU 经简单 Bridge 连接（无 Switch），NVMe 经高密度 domain（可能含 Switch）。本文教读者从 sysfs 分辨 Switch、Bridge 与 Root Port。

---

## 1. 为什么区分 Switch 和 Bridge

|             | PCIe Bridge       | PCIe Switch                                                   |
| ----------- | ----------------- | ------------------------------------------------------------- |
| 端口数      | 2 (1 up + 1 down) | N+1 (1 up + N down)                                           |
| 功能        | 透明转发          | 支持多播、ACS (Access Control Services)、VC (Virtual Channel) |
| 延迟        | ~零               | 每个端口 ~100-200ns                                           |
| 对 P2P 影响 | 通常支持          | 取决于 ACS 配置                                               |
| sysfs 表现  | 下游 1 个 device  | 下游 ≥2 个 device                                             |

Switch 内部有虚拟 PCI bus——每个下行端口分配独立的 bus number。Bridge 不分配新 bus。

---

## 2. 从 sysfs 区分

### 2.1 检查下游设备数量

```bash
# 已知一个 bridge/switch 的 BDF
BDF="0000:97:01.0"

# 查找 sysfs 路径
path=$(readlink -f /sys/bus/pci/devices/$BDF)
downstream=$(ls "$path" 2>/dev/null | grep -c "^0000:")
echo "Downstream devices: $downstream"
# = 1 → Bridge
# ≥ 2 → Switch
```

### 2.2 本环境实测

```bash
# GPU 下游（实际上 GPU 是 endpoint，无下游）
# 上游 Bridge: 97:01.0
ls /sys/devices/pci0000:97/0000:97:01.0/ | grep "^0000:"
# 0000:98:00.0    ← 仅 1 个 = Bridge
```

```bash
# domain 7f 有 122 devices——检查其 Root Port 下游结构
ls /sys/devices/pci0000:7f/ | grep "^0000:"
# 0000:7f:00.0  0000:7f:00.1  ... ← Root Port 函数
```

> 本环境 7f/ff 的高设备数来自 NVMe namespace 和 Root Port 的多 function，不是来自 Switch 的下游端口。

### 2.3 确认没有 Switch

```bash
# 全系统扫描：查找下游 ≥2 device 的非 Root Port bridge
for dev in /sys/bus/pci/devices/*; do
    cls=$(cat "$dev/class" 2>/dev/null)
    [ "$cls" != "0x060400" ] && continue
    bdf=$(basename "$dev")
    path=$(readlink -f "$dev")
    # 排除 Root Port (bus = domain bus)
    bus=$(echo "$bdf" | cut -d: -f2)
    domain_bus=$(echo "$bdf" | cut -d: -f1 | sed 's/0000://')
    [ "$bus" = "00" ] && continue
    count=$(ls "$path" 2>/dev/null | grep -c "^0000:")
    [ "$count" -gt 1 ] && echo "POTENTIAL SWITCH: $bdf, downstream=$count"
done
# 本环境：无输出 → 无 Switch
```

---

## 3. 确认 Switch 的补充方法

如果上述扫描找到候选 Switch，进一步验证：

### 3.1 检查 PCIe Capability

```bash
# Switch 通常有 PCIe Extended Capability: Device Serial Number, AER 等
lspci -vvv -s $BDF | grep -i "capabilities\|switch\|upstream\|downstream"
# 如果 lspci 未安装，用 setpci：
setpci -s $BDF 0x34.l  # Capability pointer
```

### 3.2 检查 PCIe Express Capability

Switch 的 PCIe Express Capability 中 `Device/Port Type` 字段会标明：

| 值  | 类型                           |
| --- | ------------------------------ |
| 0x0 | PCIe Endpoint                  |
| 0x4 | Root Port of PCIe Root Complex |
| 0x5 | Upstream Port of PCIe Switch   |
| 0x6 | Downstream Port of PCIe Switch |
| 0x7 | PCIe-to-PCI/PCI-X Bridge       |

```bash
# 通过 sysfs 间接判断：Switch 的 class 为 0x060400
# 但 "Upstream Port" 和 "Downstream Port" 的详细类型需要 lspci -vvv
```

### 3.3 ACS (Access Control Services)

Switch 下游端口通常有 ACS 能力：

```bash
# Switch 有 acs_* 文件（如果 ACS 被启用）
ls /sys/bus/pci/devices/$BDF/acs_* 2>/dev/null
# 如果存在 = 可能为 Switch（但开启 ACS 的 Bridge 也可能有）
```

---

## 4. Bridge 与 Switch 的性能差异

### 4.1 延迟

```text
路径A: GPU ←Bridge→ Root Complex          (本环境)
路径B: GPU ←Switch→ Switch → Root Complex (多层级)
```

Bridge 不引入额外总线层，延迟可忽略。Switch 每层增加 ~100-200ns（内部 crossbar 转发）。

本环境 GPU 路径：`RC → Bridge → GPU` 为最优——最少的中间层级，延迟最低。

### 4.2 带宽共享

```text
Bridge:     [RC]────[Bridge]────[GPU]          ← GPU 独占 bridge
Switch:     [RC]────[Switch]──┬──[GPU 0]        ← 多 GPU 共享上行
                              ├──[GPU 1]
                              └──[NVMe]
```

本环境 GPU 独占 bridge，无其他设备竞争 PCIe 带宽——这解释了为什么 H2D 实测达到 56.3 GB/s（PCIe Gen 5 ×16 效率 ~89%）。

---

## 5. 总结

| 项目         | 本环境结论                                      |
| ------------ | ----------------------------------------------- |
| PCIe Switch  | **不存在**——全系统扫描未发现                    |
| GPU 连接方式 | Root Complex → Bridge (97:01.0) → GPU (98:00.0) |
| 拓扑优势     | GPU 独占 bridge，无带宽竞争，延迟最低           |
| NVMe 连接    | 经高密度 Root Port (domain 7f/ff)，非 Switch    |
| 验证方法     | class + downstream device count + bus exclusion |

---

## 参考

- [PCIe 技术大全](01_pcie_comprehensive_guide.md)
- [PCIe 拓扑层次](06_pcie_topology_hierarchy.md)
- [PCIe 拓扑可视化](03_pcie_topology_visualization.md)
- [PCI-SIG: PCIe Switch Specification](https://pcisig.com/)
