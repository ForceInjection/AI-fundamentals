# PCIe & NVLink 带宽速查表

> 涵盖 PCIe 各代、NVLink 各代、主流 GPU 互连规格及 NVMe SSD 速度的快速参考。数据来源标注于各表下方。

---

## 1. PCIe 各代带宽

单向带宽 = GT/s × 编码效率 × Lane 数 ÷ 8。Gen3–5 使用 128b/130b 编码 (效率 ≈ 98.5%)；Gen6 使用 PAM-4 + FLIT 编码 (效率 ≈ 94.5%) [1,2]。

| 版本     | 年份 | GT/s | 编码       | x1 单向   | x4 单向    | x8 单向    | x16 单向   | x16 双向   |
| :------- | :--- | :--- | :--------- | :-------- | :--------- | :--------- | :--------- | :--------- |
| **Gen1** | 2003 | 2.5  | 8b/10b     | 0.25 GB/s | 1.0 GB/s   | 2.0 GB/s   | 4.0 GB/s   | 8.0 GB/s   |
| **Gen2** | 2007 | 5.0  | 8b/10b     | 0.5 GB/s  | 2.0 GB/s   | 4.0 GB/s   | 8.0 GB/s   | 16 GB/s    |
| **Gen3** | 2010 | 8.0  | 128b/130b  | ~1.0 GB/s | ~3.9 GB/s  | ~7.9 GB/s  | ~15.8 GB/s | ~31.5 GB/s |
| **Gen4** | 2017 | 16.0 | 128b/130b  | ~2.0 GB/s | ~7.9 GB/s  | ~15.8 GB/s | ~31.5 GB/s | ~63 GB/s   |
| **Gen5** | 2019 | 32.0 | 128b/130b  | ~3.9 GB/s | ~15.8 GB/s | ~31.5 GB/s | ~63 GB/s   | ~126 GB/s  |
| **Gen6** | 2022 | 64.0 | PAM-4/FLIT | ~7.6 GB/s | ~30.3 GB/s | ~60.5 GB/s | ~121 GB/s  | ~242 GB/s  |

> **来源**: [1] PCI-SIG 官方规范各代速率定义；[2] 本仓库 `pcie/01_pcie_comprehensive_guide.md` 第 31–47 行。

---

## 2. NVLink 各代带宽

NVLink 总双向带宽 = 单 link 速率 × 双向 × link 数量 [3]。

| 代      | 年份 | 架构      | 代表 GPU  | 单 Link 速率 | Link 数 | 双向总带宽 | 关键技术             |
| :------ | :--- | :-------- | :-------- | :----------- | :------ | :--------- | :------------------- |
| **1.0** | 2016 | Pascal    | P100      | 20 GB/s      | 4       | 160 GB/s   | 首次引入 GPU 直连    |
| **2.0** | 2017 | Volta     | V100      | 25 GB/s      | 6       | 300 GB/s   | NVSwitch 1.0         |
| **3.0** | 2020 | Ampere    | A100      | 25 GB/s      | 12      | 600 GB/s   | 信号对减半，链路翻倍 |
| **4.0** | 2022 | Hopper    | H100/H200 | 25 GB/s      | 18      | 900 GB/s   | SHARP 网内计算       |
| **5.0** | 2024 | Blackwell | B200/B300 | 50 GB/s      | 18      | 1.8 TB/s   | NVL72 机架级扩展     |
| **6.0** | 2026 | Rubin     | Vera      | —            | —       | 3.6 TB/s   | NVSwitch 6, RAS 增强 |

> **来源**: [3] 本仓库 `nvlink/nvlink_intro.md` 第 57–61 行；[4] NVIDIA GB200 NVL4 Architecture Whitepaper。

---

## 3. 主流 GPU 互连与显存规格

所有带宽均为**双向**总带宽。HBM 带宽为近存带宽 (near-memory BW)，PCIe/NVLink 为片间带宽 [3,4,5,6,7]。

| GPU          | 架构            | 显存         | 显存带宽  | PCIe 上行 | NVLink | NVLink 总带宽 | NVSwitch |
| :----------- | :-------------- | :----------- | :-------- | :-------- | :----- | :------------ | :------- |
| **P100**     | Pascal          | 16 GB HBM2   | 732 GB/s  | Gen3 x16  | 1.0    | 160 GB/s      | —        |
| **V100**     | Volta           | 32 GB HBM2   | 900 GB/s  | Gen3 x16  | 2.0    | 300 GB/s      | 1.0      |
| **A100**     | Ampere          | 80 GB HBM2e  | 2.0 TB/s  | Gen4 x16  | 3.0    | 600 GB/s      | 2.0      |
| **H100**     | Hopper          | 80 GB HBM3   | 3.35 TB/s | Gen5 x16  | 4.0    | 900 GB/s      | 3.0      |
| **H200**     | Hopper          | 141 GB HBM3e | 4.8 TB/s  | Gen5 x16  | 4.0    | 900 GB/s      | 3.0      |
| **B200**     | Blackwell       | 192 GB HBM3e | 8.0 TB/s  | Gen5 x16  | 5.0    | 1.8 TB/s      | 4.0      |
| **B300**     | Blackwell Ultra | 270 GB HBM3e | 7.7 TB/s  | Gen6 x16  | 5.0    | 1.8 TB/s      | 4.0      |
| **RTX 5090** | Blackwell       | 32 GB GDDR7  | 1.79 TB/s | Gen5 x16  | **—**  | —             | —        |
| **L40S**     | Ada Lovelace    | 48 GB GDDR6  | 864 GB/s  | Gen4 x16  | **—**  | —             | —        |

> [!NOTE]
> **消费级 GPU (GeForce RTX) 不支持 NVLink**。自 RTX 30 系列起，NVIDIA 移除了消费级 GPU 的 NVLink 接口。仅数据中心 GPU (Tesla/Quadro/HGX) 支持 NVLink [3]。
>
> B300 数据为 HGX B300 规格。NVL72 SXM 变体为 279 GB HBM3e / 8.0 TB/s [10]。  
> **来源**: [5] NVIDIA H100 Data Sheet; [6] NVIDIA H200 Product Page; [7] 本仓库 `superchips/nvidia_gb300.md`; [3] 本仓库 `nvlink/nvlink_intro.md`; [10] NVIDIA Blackwell Ultra Datasheet。

---

## 4. NVMe SSD 典型速度 (M.2/U.2, x4)

NVMe SSD 使用 PCIe x4 接口。理论值 = PCIe x4 单向带宽；典型读取 ≈ 理论值的 85–90% (扣除协议开销) [1]。

| 代       | 接口      | x4 理论单向 | 典型顺序读   | 典型顺序写   | 代表型号                         |
| :------- | :-------- | :---------- | :----------- | :----------- | :------------------------------- |
| **Gen3** | M.2 / U.2 | ~3.9 GB/s   | ~3,500 MB/s  | ~3,200 MB/s  | Samsung 970 Pro, WD Black SN750  |
| **Gen4** | M.2 / U.2 | ~7.9 GB/s   | ~7,000 MB/s  | ~6,900 MB/s  | Samsung 990 Pro, WD Black SN850X |
| **Gen5** | M.2 / U.2 | ~15.8 GB/s  | ~14,000 MB/s | ~12,000 MB/s | Crucial T705, Samsung 9100 Pro   |

> **来源**: [1] PCI-SIG 各代速率; Samsung/Crucial/Western Digital 官方产品规格页。

---

## 5. 典型场景带宽瓶颈速查

| 场景                          | 瓶颈链路                  | 理论单向带宽              | 备注                                    |
| :---------------------------- | :------------------------ | :------------------------ | :-------------------------------------- |
| GPU→GPU (PCIe P2P, 同 Switch) | PCIe Gen5 x16             | ~63 GB/s                  | PIX 拓扑，GPUDirect P2P 全速 [8]        |
| GPU→GPU (PCIe P2P, 经 RC)     | PCIe Gen5 x16 ÷ 2         | ~32 GB/s                  | PHB 拓扑，带宽减半 (数据经 RC 往返) [8] |
| GPU→GPU (NVLink 4.0)          | NVLink 4.0                | 900 GB/s                  | H100/H200，18 links [3]                 |
| GPU→GPU (NVLink 5.0)          | NVLink 5.0                | 1.8 TB/s                  | B200/B300，18 links [3]                 |
| H2D / D2H (GPU↔CPU)           | PCIe Gen5 x16             | ~63 GB/s                  | 单方向上限 [1]                          |
| NVMe→GPU (传统, Buffered)     | PCIe Gen4 x4 → DRAM → GPU | ~3.5 GB/s                 | 2 次 PCIe 穿越 + CPU memcpy [9]         |
| NVMe→GPU (传统, O_DIRECT)     | PCIe Gen4 x4 → DRAM → GPU | ~7 GB/s                   | 2 次 PCIe 穿越，无 CPU memcpy [9]       |
| NVMe→GPU (GDS)                | PCIe P2P 直通             | ~7 GB/s (同 PCIe Gen4 x4) | 1 次 PCIe 穿越，同 Switch 最优 [9]      |
| NVMe→GPU (GDS, Gen5)          | PCIe P2P 直通             | ~14 GB/s                  | 1 次 PCIe 穿越，基于 Gen5 x4 NVMe [9]   |
| CPU↔GPU (NVLink-C2C)          | NVLink-C2C                | ~225 GB/s                 | Grace Hopper/Grace Blackwell，单向 [7]  |
| CPU↔CPU (NVLink-C2C)          | NVLink-C2C                | ~450 GB/s                 | Grace CPU 之间，单向 [7]                |

> **来源**: [8] 本仓库 `gpudirect/02_gpudirect_p2p.md` 第 5.1 节；[9] 本仓库 `gpudirect/01_gpudirect_technology.md` 第 3.1–3.4 节；[3] `nvlink/nvlink_intro.md`；[7] `superchips/nvidia_gb300.md`。

---

## 6. 速算公式

```text
PCIe 单向带宽 (GB/s) = GT/s × 编码效率 × Lane 数 ÷ 8

  Gen1/2 (8b/10b):    效率 = 8/10 = 0.80
  Gen3/4/5 (128b/130b): 效率 = 128/130 ≈ 0.985
  Gen6 (PAM-4 + FLIT):  效率 = 242/256 ≈ 0.945

NVLink 双向总带宽 = 单 Link 速率 × 2 (双向) × Link 数

NVMe 典型读取 = PCIe x4 理论单向 × ~0.85–0.90 (NVMe 协议开销)
```

**示例**: Gen5 x16 单向 = 32 × 0.985 × 16 ÷ 8 = 63.0 GB/s

---

## 7. NVLink vs PCIe 带宽对比 (x16)

| NVLink 代   | 总双向带宽 | 对比 PCIe Gen5 (×16) | 倍数 |
| :---------- | :--------- | :------------------- | :--- |
| 1.0 (P100)  | 160 GB/s   | vs ~31.5 GB/s (Gen3) | 5.1× |
| 2.0 (V100)  | 300 GB/s   | vs ~63 GB/s (Gen3)   | 4.8× |
| 3.0 (A100)  | 600 GB/s   | vs ~63 GB/s (Gen4)   | 9.5× |
| 4.0 (H100)  | 900 GB/s   | vs ~126 GB/s (Gen5)  | 7.1× |
| 5.0 (B200)  | 1.8 TB/s   | vs ~126 GB/s (Gen5)  | 14×  |
| 6.0 (Rubin) | 3.6 TB/s   | vs ~242 GB/s (Gen6)  | 14×  |

> 注：NVLink 1.0/2.0/3.0 对应的是各自年代的 PCIe 版本 (Gen3/Gen4)，NVLink 4.0+ 对比最新 PCIe Gen5。

---

## 参考资料

1. PCI-SIG Official Specifications — 各代速率定义 (https://pcisig.com)
2. 本仓库 `pcie/01_pcie_comprehensive_guide.md` — PCIe 带宽表与编码说明
3. 本仓库 `nvlink/nvlink_intro.md` — NVLink 各代带宽表与技术演进
4. NVIDIA GB200 NVL4 Architecture Whitepaper (https://resources.nvidia.com)
5. NVIDIA H100 Tensor Core GPU Data Sheet (https://www.nvidia.com/en-us/data-center/h100/)
6. NVIDIA H200 Tensor Core GPU Product Page (https://www.nvidia.com/en-us/data-center/h200/)
7. 本仓库 `superchips/nvidia_gb300.md` — GB200/GB300 规格对比
8. 本仓库 `gpudirect/02_gpudirect_p2p.md` — GPU 拓扑与 P2P 性能
9. 本仓库 `gpudirect/01_gpudirect_technology.md` — GDS 数据路径与性能对比
10. NVIDIA Blackwell Ultra Datasheet — B300 GPU 规格 (https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet)
