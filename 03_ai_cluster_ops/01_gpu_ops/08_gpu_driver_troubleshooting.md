# GPU 驱动故障速查——nvidia-smi 不可用时的排查路径

GPU 运维中最让人焦虑的不是性能差，而是 `nvidia-smi` 本身报错——"Unable to determine the device handle"、"No devices were found"、"NVIDIA-SMI has failed"。这些问题通常和驱动加载、内核模块、DKMS 编译相关，不是 GPU 硬件故障，但排查路径和 `nvidia-smi` 正常工作时的健康检查完全不同。

本文按故障现象分类——每种现象给出一句话的根因和最可能的修复命令。

---

## 一、现象分类速查

| 现象                                                    | 最可能的根因                     | 最快验证                                            |
| ------------------------------------------------------- | -------------------------------- | --------------------------------------------------- |
| `nvidia-smi` 报 "No devices were found"                 | 驱动未加载                       | `lsmod \| grep nvidia`                              |
| `nvidia-smi` 报 "Unable to determine the device handle" | GPU 被其他驱动占用（如 nouveau） | `lspci -nnk \| grep -A3 NVIDIA`                     |
| `nvidia-smi` 卡住，无输出                               | GPU 陷入不可恢复状态             | `dmesg \| tail -30` 查 XID 错误                     |
| 驱动更新后 `nvidia-smi` 版本不匹配                      | DKMS 编译失败或未重启            | `cat /proc/driver/nvidia/version` 对比 `nvidia-smi` |
| GPU 在 `lspci` 中可见但 `nvidia-smi` 看不到             | PCIe AER 错误导致 GPU 被隔离     | `dmesg \| grep -i "AER\|PCIe.*error"`               |

---

## 二、驱动未加载

```bash
# 检查内核模块
lsmod | grep nvidia
# 应有: nvidia_uvm, nvidia_drm, nvidia_modeset, nvidia

# 手动加载
modprobe nvidia
modprobe nvidia_uvm

# 检查加载失败原因
dmesg | grep -i nvidia | tail -20
```

**常见原因**：

| 原因                                    | 修复                                                                                                                               |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Secure Boot 阻止了未签名的内核模块      | `mokutil --sb-state` 确认，在 BIOS 中禁用或签名模块                                                                                |
| 内核升级后驱动未重新编译                | `dkms status` 看 nvidia 模块状态，手动 `dkms install nvidia/<ver>`                                                                 |
| `nouveau` 驱动占用了 GPU（Ubuntu 默认） | `lsmod \| grep nouveau`，有则 `echo 'blacklist nouveau' > /etc/modprobe.d/blacklist-nouveau.conf && update-initramfs -u && reboot` |

---

## 三、驱动版本不匹配

驱动更新后最常见的坑：内核模块已加载新版本，但用户态库（`nvidia-smi`、`libcuda.so`）还是旧的。

```bash
# 对比内核模块版本 vs 用户态版本
cat /proc/driver/nvidia/version | head -1
nvidia-smi | head -1

# 如果两者不一致，重启后一般就能解决。未重启时：
# 1. 卸载旧模块 → 加载新模块
rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
modprobe nvidia nvidia_uvm
```

**DKMS 编译失败**——更新内核或驱动后模块无法自动编译：

```bash
dkms status                          # 查看所有 DKMS 模块状态
dkms status | grep nvidia            # 只看 NVIDIA
# installed = 编译成功, added = 未编译, failed = 编译失败

# 手动重新编译
dkms remove nvidia/<version> --all
dkms install nvidia/<version>
```

---

## 四、GPU 在 lspci 中可见但不可用

```bash
lspci -nnk | grep -A3 NVIDIA
# 检查 "Kernel driver in use" 是否为 nvidia
# 如果是 "nouveau" → §二 黑名单步骤
# 如果是 "vfio-pci" → GPU 被透传给虚拟机，宿主机 nvidia-smi 不可用
# 如果是空或 "(none)" → 驱动未绑定该 GPU
```

**PCIe AER 错误导致 GPU 被隔离**——内核检测到 PCIe 设备多次错误后将其停用：

```bash
dmesg | grep -i "AER\|PCIe.*error\|corrected"
# 出现 "AER: Corrected error received" → 单次可忽略
# 出现 "AER: Uncorrected (Fatal) error" → GPU 可能已被隔离

# 重新扫描 PCIe 总线恢复（无需重启）
echo 1 > /sys/bus/pci/devices/0000:XX:00.0/remove
echo 1 > /sys/bus/pci/rescan
```

---

## 五、nvidia-persistenced 问题

`nvidia-persistenced` 保持 GPU 驱动在内核中常驻。如果它挂了，空闲 GPU 会被驱动卸载，下一次访问有 1-2s 冷启动延迟——对延迟敏感的场景（如在线推理）影响显著。

```bash
# 状态检查
systemctl status nvidia-persistenced

# 如果看到 "GPU 0000:XX:00.0 has fallen off the bus"
# → GPU 硬件脱离 PCIe 总线，不是 persistenced 的问题
# → 检查电源、散热、物理连接

# 重置 persistenced（§四的情况排除后）
nvidia-smi -pm 0 && nvidia-smi -pm 1
```

---

## 六、旧 GPU（Kepler/Maxwell）兼容性

Kepler（CC 3.x）和部分 Maxwell（CC 5.x）GPU 在新驱动中已进入 Legacy 支持。如果 `nvidia-smi` 显示 "Legacy Support" 或 CUDA 版本不兼容：

```bash
# 查询 GPU 的计算能力
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Kepler (3.x) → 最高支持 CUDA 11.x
# Maxwell (5.x) → 最高支持 CUDA 12.x
# Pascal+ (6.x+) → 当前最新 CUDA
```

---

## 七、相关资源

- [nvidia-smi 场景速查](03_nvidia_smi_guide.md) — 驱动正常后的日常运维命令
- [GPU 集群健康检查](06_gpu_health_check.md) — L1/L2/L3 检查流程
- [NVIDIA Driver Installation Quickstart](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html)
- [NVIDIA GPU Deployment and Management Documentation](https://docs.nvidia.com/deploy/)
