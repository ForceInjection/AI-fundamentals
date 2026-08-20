# CXL 命令行工具链全景：cxl / daxctl / ndctl 从查看到配置

> 2026-08-20 | 基于远程测试机实测（内核 6.8.0，cxl v77）+ cxl-cli v77 man page / Linux 内核文档。命令体系与无设备行为为实测输出；完整工作流（需 CXL 设备）依据官方 man page 与内核文档整理。测试机信息已脱敏。

协议层的 HDM decoder、region、DAX 形态，在操作层落地为三个命令行工具：`cxl` 管设备与地址窗口、`daxctl` 管内存形态转换、`ndctl` 管持久内存——它们来自同一个开源项目（ndctl 仓库），却分工清晰、各有依赖。本文是 08 文章《[CXL 互联协议全景解读](08_cxl_interconnect_overview.md)》的操作面：前文讲了协议与内核形态的概念，本文讲这些概念怎么变成可执行的命令序列、输出怎么读、无设备环境长什么样。

> **术语速查**（未读 08 文章的读者先看这里，完整术语表见 [08 文章附录](08_cxl_interconnect_overview.md)）：
>
> - **decoder**：把一段主机物理地址（HPA）窗口路由到具体 CXL 设备/端口的地址解码器；
> - **region**：由若干条带化的 CXL 内存设备（memdev）组成的一块连续地址窗口，可整体交付给内核；
> - **memdev**：一颗 CXL 内存设备（协议分类中的 Type 1/2/3 设备对象，内存扩展器即 Type 3）；
> - **DAX / devdax**：CXL 内存的默认暴露形态 `/dev/daxX.Y`，用户态可直接 mmap 直访；
> - **CFMWS**：固件 ACPI 表中预留的 CXL 内存地址窗口（HPA 从这里分配）；
> - **HPA**：主机物理地址（Host Physical Address）。

---

## 一、背景：工具链三兄弟与内核模块地图

先画清三把工具的边界，后面所有命令才不会混淆。

### 1.1 三个工具的分工与同源关系

`cxl`、`daxctl`、`ndctl` 同源于 [ndctl](https://github.com/pmem/ndctl) 仓库（`cxl-cli` 与 `ndctl` 在同一个构建里编译，安装 ndctl 包即同时获得三者的 `libcxl` / `libdaxctl` / `libndctl` 三个库）。分工按「管什么」划分：

| 工具     | 管理对象                              | 对应协议/内核概念                               |
| -------- | ------------------------------------- | ----------------------------------------------- |
| `cxl`    | CXL 设备、端口、decoder、region、标签 | CXL.mem 设备、HDM decoder、CFMWS 窗口、MLD/LD   |
| `daxctl` | DAX 设备的形态转换与内存 online       | devdax ↔ system-ram 两种形态（08 文章 4.2/4.3） |
| `ndctl`  | 持久内存的命名空间、标签、安全        | pmem 路径（08 文章 4.4）                        |

粗记一句：**`cxl` 管「内存怎么进来」，`daxctl` 管「进来的内存变成什么」，`ndctl` 管「进来的内存是否持久」。**

### 1.2 内核模块依赖链

三个工具只是前端，真正的状态在 `/sys/bus/cxl` 与内核模块里。模块依赖是单链：`cxl_acpi`（平台枚举，从 ACPI CEDT 表发现 CXL 拓扑）→ `cxl_core`（核心框架，所有 CXL 子模块共享）→ 具体驱动：

```text
cxl_acpi → cxl_core
              ├── cxl_pci   （CXL 设备 PCIe 功能绑定）
              ├── cxl_mem   （CXL.mem 设备注册）
              ├── cxl_port  （端口与 decoder 管理）
              └── cxl_pmem  （持久内存 region 注册，接 nvdimm 总线）
```

与之平行的 DAX 侧：`dax_kmem`（devdax → System RAM 转换，08 文章 4.2 的 `dev_dax_kmem_probe`）、`dax_hmem`（HMAT/EFI soft-reserved 内存）、`dax_pmem`（持久内存 DAX）。

### 1.3 已实测：测试机的模块与平台准备状态

测试机是一台 Intel 服务器（内核 6.8.0），启动参数中已经为 CXL 调试做好了准备：

```text
Command line: ... cxl.debug=1 cxl_core.debug=1 memhp_default_state=offline
```

`cxl_acpi` 与 `cxl_core` 已加载（`cxl_core 299008 1 cxl_acpi`），`dax_hmem` 也已加载。但 dmesg 里有一个值得注意的细节：

```text
cxl_core: unknown parameter 'debug' ignored
```

**6.8 内核的 `cxl_core` 不接受 `debug` 参数**——启动命令行里的 `cxl_core.debug=1` 被静默忽略。这是「按文档配置却发现没生效」的典型来源，排查时值得先 `lsmod` + `dmesg` 确认模块实际状态。

---

## 二、命令地图：子命令与协议概念的映射

`cxl` 有 19 个子命令（已实测 `cxl --list-cmds`），但按职责可以归成五族。每个子命令背后都对应 08 文章的一个协议概念——这张映射表是理解命令体系的钥匙。

### 2.1 cxl 五族子命令（已实测）

```text
$ cxl --list-cmds
version  list  help  zero-labels  read-labels  write-labels
disable-memdev  enable-memdev  reserve-dpa  free-dpa
disable-port  enable-port  set-partition  disable-bus
create-region  enable-region  disable-region  destroy-region  monitor
```

| 族          | 子命令                                                                                | 职责                                      |
| ----------- | ------------------------------------------------------------------------------------- | ----------------------------------------- |
| 查看/监控   | `list`、`monitor`                                                                     | 枚举拓扑（JSON）/ 订阅内核 CXL trace 事件 |
| 设备启停    | `enable-memdev`/`disable-memdev`、`enable-port`/`disable-port`、`disable-bus`         | 设备/端口/总线的电源与枚举状态            |
| 标签与分区  | `read-labels`/`write-labels`/`zero-labels`、`set-partition`、`reserve-dpa`/`free-dpa` | 设备标签读写、易失/持久分区、DPA 地址预留 |
| Region 管理 | `create-region`/`enable-region`/`disable-region`/`destroy-region`                     | 08 文章的核心概念：region 的创建到销毁    |
| 其他        | `version`、`help`                                                                     | 版本与帮助                                |

decoder 不需要用户手动创建：root / port decoder 由固件（ACPI CEDT 表）枚举，endpoint decoder 在 `create-region` 时自动配置——`-d` 选项只是从已枚举的 decoder 里挑选目标（见 3.2）。

### 2.2 概念映射表：命令 ↔ 协议概念

| 命令/对象                                      | 08 文章概念                                                       | 一句话含义                               |
| ---------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------- |
| `cxl list` 的 bus/port/endpoint                | CXL 拓扑层次（CXL 2.0 的 VH 虚拟层次）                            | 总线 → 端口 → 设备的三级枚举树           |
| memdev                                         | CXL.mem 设备（Type 1/2/3 的设备对象）                             | 一颗 CXL 内存设备（内存扩展器即 Type 3） |
| root decoder / port decoder / endpoint decoder | HDM decoder 编程（08 文章 4.1：CFMWS 窗口 → HPA 分配）            | 把 HPA 窗口路由到设备/端口的地址解码器   |
| region                                         | 08 文章 4.1：CFMWS 预留窗口中 region 的 HPA 分配（`alloc_hpa()`） | 若干 memdev 条带化组成的一块连续地址窗口 |
| `set-partition` / LD                           | CXL 2.0 的易失/持久分区与 MLD 逻辑设备                            | 设备内部切片与逻辑设备归属               |
| `reserve-dpa` / `free-dpa`                     | DPA（Device Physical Address）的软件预留                          | 设备物理地址的软件预留管理               |
| `/dev/daxX.Y`（daxctl 管理）                   | 08 文章 4.1：所有 region 的「出厂形态」devdax                     | devdax 设备文件，用户态可 mmap 直访      |
| `daxctl reconfigure-device`                    | 08 文章 4.2：System RAM 形态（`dev_dax_kmem_probe`）              | 把 devdax 转交内核页分配器作为普通内存   |

### 2.3 daxctl 子命令（已实测）

```text
$ daxctl --list-cmds
version  list  help  split-acpi  migrate-device-model
create-device  destroy-device  reconfigure-device
online-memory  offline-memory  disable-device  enable-device
```

核心是 `reconfigure-device`（形态转换）与 `online-memory`（内存块上线）。`migrate-device-model` 值得一提：dax 设备有 dax-class 与 dax-bus 两种设备模型，**reconfigure 依赖 dax-bus 模型**，老模型下会报 "device model is dax-class"（见 3.4 的 man 原文）——迁移命令就是为此准备的。

### 2.4 ndctl 子命令（已实测）

34 个子命令，按族分：

- **命名空间**：`create-namespace` / `destroy-namespace` / `enable-namespace` / `disable-namespace` / `check-namespace` / `clear-errors`
- **标签与健康**：`read-labels` / `write-labels` / `init-labels` / `check-labels` / `zero-labels`、`read-infoblock` / `write-infoblock`、`inject-error` / `inject-smart` / `sanitize-dimm`
- **持久内存生命周期**：`update-firmware` / `activate-firmware` / `wait-scrub` / `start-scrub`、`setup-passphrase` / `update-passphrase` / `remove-passphrase` / `freeze-security` / `load-keys` / `wait-overwrite`
- **region、DIMM 与监控**：`enable-region` / `disable-region`、`enable-dimm` / `disable-dimm`、`list` / `monitor`

普通读者只需要前两类；CXL pmem 路径（08 文章 4.4）从 `cxl_pmem_region` 注册进 nvdimm 总线后，命名空间管理完全复用 ndctl 的 `create-namespace` 体系。

---

## 三、核心工作流：从插卡到可用内存

工具链的完整使命是把一块 CXL 内存变成 CPU 可用的内存。流程有严格次序——**先有 decoder 才能建 region，region 使能后才有 `/dev/dax`，才谈得上形态转换**。

> **本节命令序列依据 cxl-cli v77 官方 man page 与内核文档整理**，命令输出为 man page 自带示例。
>
> **执行前置条件**：以下命令**均需 root 权限**（`cxl`/`daxctl`/`ndctl` 无例外）；工具通过安装 ndctl 包获得（如 `apt install ndctl` / `dnf install ndctl`）；以下命令在**插有 CXL 设备（且固件预留了 CFMWS 窗口）的服务器**上执行——普通机器上执行会得到 4.1 节的 `[]` 输出。

### 3.1 第 0 步：`cxl list` 枚举——先看有什么

一切从枚举开始。`cxl list` 以 JSON 输出平台拓扑，对象有严格嵌套层级（man page 原文）：`buses` → `ports` → `endpoints` → `memdevs`，decoder 分三级（root / port / endpoint）。常用过滤器：

```text
cxl list -vv                  # 全量拓扑，两档 -v 输出更多属性
cxl list -M                   # 只看 memdevs（大写短选项，脚本常用）
cxl list -R                   # 只看 regions
cxl list --decoders --regions # 只看 decoder 与 region
cxl list -m mem0 -v           # 只看单个 memdev
```

无设备时的输出是后续第 4.1 节的重点——它返回 `[]` 而非报错，初学者容易误判。真实部署中 memdev 通常随枚举自动使能；若 `cxl list` 中 memdev 状态为 disabled，需先 `cxl enable-memdev <memdev>` 再创建 region。

### 3.2 第 1 步：`cxl create-region`——装配 region

v77 中 region 的创建即 decoder 的配置。命令中的 `decoder0.1` 与 `mem0`/`mem1` 不是凭空来的——它们正是第 0 步 `cxl list --decoders --memdevs -vv` 输出的名字。man page 的 EXAMPLE 给出了最经典的条带化命令与 JSON 输出：

```text
# cxl create-region -m -d decoder0.1 -w 2 -g 1024 mem0 mem1
{
  "region":"region0",
  "resource":"0xc90000000",
  "size":"512.00 MiB (536.87 MB)",
  "interleave_ways":2,
  "interleave_granularity":1024,
  "mappings":[
    {"position":1, "decoder":"decoder4.0"},
    {"position":0, "decoder":"decoder3.0"}
  ]
}
created 1 region
```

关键选项（`cxl create-region --help`）：

| 选项           | 含义                                                |
| -------------- | --------------------------------------------------- |
| `-t ram\|pmem` | region 类型：易失（HDM-H）或持久（08 文章 4.4）     |
| `-m`           | 位置参数为 memdev 名                                |
| `-d <decoder>` | 指定 root decoder（省略时自动选择）                 |
| `-w <n>`       | 条带化路数（interleave ways），须与 memdev 数量一致 |
| `-g <n>`       | 条带粒度（interleave granularity，字节）            |
| `-s <size>`    | 大小；省略时取每个 memdev 的最大可用                |
| `-u`           | `--human`，人类可读输出（注意：不是 UUID）          |
| `-U <uuid>`    | 持久 region 的 UUID；省略自动生成                   |

输出中的 `resource`（0xc90000000）即 region 在 CFMWS 窗口中分配到的 HPA——正是 08 文章 4.1 节 `alloc_hpa()` 的内核行为；`position` 是条带位置，即 memdev 在交错组中的序号。

### 3.3 第 2 步：`cxl enable-region`——让 `/dev/dax` 出现

region 创建后处于 disabled 状态。`cxl enable-region <region>` 使能后，内核的 `cxl_dax_region` 驱动自动创建 `/dev/daxX.Y`（08 文章 4.1：所有 region 的「出厂形态」）。命令支持 `all` 关键字：

```text
# cxl enable-region all
enabled 2 regions
```

### 3.4 第 3 步：形态分岔——`daxctl reconfigure-device`

现在到了 08 文章第四章的核心抉择：`/dev/daxX.Y` 保持 devdax 形态（用户态 mmap 直访），还是 `reconfigure-device --mode=system-ram` 转交给内核页分配器？

```text
# 二选一：
# 方案 A：自动 online（默认 online 为 movable zone）
# daxctl reconfigure-device --mode=system-ram dax0.0
# 方案 B：手动控制 online 与 zone（后续需自行 online-memory，见 3.5）
# daxctl reconfigure-device --mode=system-ram --no-online --no-movable dax0.0
```

man page 的警告值得逐字记住：

> **This is a destructive operation. Any data on the dax device will be lost.**
>
> daxctl-reconfigure-device nominally expects that it will online new memory blocks as movable, so that kernel data doesn't make it into this memory. However, there are other potential agents that may be configured to automatically online new hot-plugged memory as it appears. Most notably, these are the `/sys/devices/system/memory/auto_online_blocks` configuration, or system udev rules. If such an agent races to online memory sections, daxctl checks if the blocks were onlined as movable memory.

三个要点：**破坏性操作**（数据丢失）；默认 online 为 **movable** zone（防内核数据进入慢内存）；与 `auto_online_blocks` / udev 等自动 online 机制存在**竞态**——daxctl 会检测在线化后 zone 是否偏离 movable 并告警。

另一个坑是设备模型：dax-class 模型下 reconfigure 直接失败：

```text
# daxctl reconfigure-device --mode=system-ram --region=0 all
libdaxctl: daxctl_dev_disable: dax3.0: error: device model is dax-class
dax3.0: disable failed: Operation not supported
error reconfiguring devices: Operation not supported
reconfigured 0 devices
```

此时需先 `daxctl migrate-device-model`（见 2.3）。成功路径下命令无输出（或输出 `reconfigured N devices`），判断成败以第 4 步验证为准。

形态转换不是单向的——**BIOS 重启后 devdax 会回落为 system-ram 模式**（CXL 内存重新枚举为 NUMA 节点，内存块未上线、字符设备不可用），真实部署中重启后需反向转换切回 devdax：

```text
# system-ram → devdax 反向转换（转换前需 offline memblock，--force 自动处理）
daxctl reconfigure-device --mode=devdax --force dax0.0
```

### 3.5 第 4 步：验证——内存真的可用吗

若第 3 步选了方案 B（`--no-online`），验证前需先 `daxctl online-memory` 让内存块上线：

```text
lspci | grep -i cxl        # 设备在 PCIe 树中
lsblk                       # 块设备（pmemN 路径）
numactl -H                  # NUMA 距离表：CXL 内存节点出现
cat /proc/iomem | grep -i cxl  # CFMWS 窗口与 region 的 HPA
```

`numactl -H` 是最直观的验收：system-ram 转换成功后，`node distance` 表中会出现新的节点（通常 distance 显著大于本地 DDR 节点的 10/20），CXL 内存的 NUMA 距离比本地内存大——这正是 08 文章第五章「CXL vs DDR」在系统里的落点。

---

## 四、验证与诊断：从无设备到生产环境

工具链的另一半技能是「读输出」。无设备环境的输出模式与有设备时完全不同——**`cxl list` 返回 `[]` 而不是报错**，这本身就是一种信号。

### 4.1 已实测：无设备环境的完整画像

测试机（内核 6.8.0，cxl v77，无 CXL 设备）上，工具链与内核的行为全景：

```text
$ cxl list -vv
  Warning: no matching devices found
[]

$ cxl list --decoders --regions
  Warning: no matching devices found
[]

$ daxctl list
（空输出）

$ ndctl list
（空输出）

$ ls /sys/bus/cxl/devices/
（空目录）

$ grep -i cxl /proc/iomem
（无输出——没有 CFMWS 窗口）
```

诊断结论与排查顺序（从外到内）：

1. **lspci 无 CXL 设备** → 硬件未枚举（没插卡、BIOS 未开 CXL、或平台不支持）；
2. **`/sys/bus/cxl` 不存在** → `cxl_acpi` 模块未加载（`modprobe cxl_acpi`，或内核未编译 CXL 支持）；
3. **`/sys/bus/cxl` 存在但空** → ACPI 层枚举成功但无设备注册（测试机即此状态）；
4. **`/proc/iomem` 无 CXL 窗口** → 没有 CFMWS 固件资源——即使插了设备也没有 HPA 窗口可分配。

这条「从外到内」的顺序反过来就是部署排查的 checklist。两个真实环境经验补充：部分服务器未安装 cxl-cli（工具链不完整时管理可全走 daxctl）；`cxl_mem mem0: CXL port topology not found` 是已知内核 bug（ACPI probe 顺序问题），只影响 system-ram 路径的内存上线，不影响 devdax 字符设备。

### 4.2 生产验证：Micron CXL 资源套件与延迟观测

管理工具链验证完成、设备就位之后的下一步是性能验证——它独立于管理命令，且两种访问模式恰好对应本文 3.4 的形态分岔。Micron 开源了 **CXL Memory Resource Kit（cxl-reskit）**（[GitHub](https://github.com/cxl-micron-reskit/cxl-reskit)），内含四个基准工具：Intel MLC（专有，延迟/带宽矩阵）、multichase（指针追逐，延迟特征）、STREAM（持续带宽）、stressapptest（稳定性压测）：

- **devdax 模式**：直接以设备文件为基准目标，如 `sudo ./multichase -d /dev/dax0.0`、`sudo ./stream -a 1000000000 -d /dev/dax0.0`；
- **system-ram 模式**：CXL 内存作为 NUMA 节点，用 `numactl --membind <node> ./multichase` 绑定访问。

cxl-reskit 需在插有 CXL 设备的机器上运行，测试机（无 CXL 设备）未执行性能基准。部署建议：两种形态分别基准，对比本地 DDR 节点与 CXL 节点的带宽/延迟曲线，验证是否落在 08 文章 3.4 节的实测数据区间；devdax 模式的直访结果还能顺带验证 08 文章 4.3 的 mmap 直访语义。

### 4.3 `cxl monitor`：trace 事件订阅

`cxl monitor` 订阅内核 CXL trace 事件（UCE 内存错误等），转成 JSON 输出到 stdout 或日志文件（`cxl monitor --help` 已实测，事件流需设备触发）。生产环境常配合 `--daemon` 与 `--log` 做后台监控（man 示例：`cxl monitor --daemon --log=/var/log/cxl-monitor.log`），错误事件（如 Poison 注入、地址解码失败）会以 JSON 通知形式落盘。

---

## 五、权衡：工具链的成熟度边界

工具链的存在不等于功能可用。三处边界值得记住。

**第一，成熟度地图的 [0]/[1] 评分（[0] = 未实现，[1] = 初始可用）**。08 文章 4.6 已引用 Linux CXL 子系统的 maturity-map：跨主机能力（Fabrics / G-FAM、多主机一致性共享内存）为 [0]，LD 热插拔为 [1]，DCD 为 [0]。工具链同样受此约束——**命令存在但底层能力未成熟**时，命令会执行成功或报错，但「成功」可能停留在单机范围。

**第二，形态转换的不可逆代价**。`reconfigure-device` 是破坏性操作；dax-class 与 dax-bus 两种设备模型并存期（老内核 `dax_pmem_compat` 驱动）会造成「命令存在但必失败」的中间态，需要先迁移设备模型。这些细节都在 man page 里，但没人通读 man page——这正是本文 3.4 节把原文摘出来的原因。

**第三，版本差异的坑**（全部实测确认）：

| 版本事实                                 | 影响                                                                      |
| ---------------------------------------- | ------------------------------------------------------------------------- |
| 内核 6.8 的 `cxl_core` 忽略 `debug` 参数 | 启动参数看似生效实则被忽略（dmesg 确认）                                  |
| daxctl 的 `split-acpi`                   | HMAT 相关 ACPI 表拆分工具，普通场景用不到，但存在即说明工具链仍在快速演进 |

工具链的演进速度（三工具合计 65 个子命令，`--list-cmds` 在不同版本间持续变动）意味着：**以本机 `--list-cmds` 与 `--help` 为第一手资料，网上教程降级为参考**。

---

## 六、结语

工具链把协议语义折叠成可执行的命令序列，但命令只是入口——真正的复杂度在「命令之间的依赖次序」与「输出怎么读」上。依赖次序决定了 create-region 前必须有 decoder、enable-region 前必须有 region；输出解读决定了 `[]` 与报错的区别、`mappings` 里 position 与 decoder 的对应、以及 `reconfigure-device` 的破坏性警告不是吓唬人。CXL 的哲学是「设备简单、主机软件灵活」（08 文章结语），工具链恰好是这套哲学的用户态镜像：命令不复杂，复杂的是时序与状态。

## 附：命令速查总表与诚实声明

### 三工具子命令总表（实测 `--list-cmds`）

| 工具           | 子命令                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `cxl`（19）    | version list help zero-labels read-labels write-labels disable-memdev enable-memdev reserve-dpa free-dpa disable-port enable-port set-partition disable-bus create-region enable-region disable-region destroy-region monitor                                                                                                                                                                                                                                                  |
| `daxctl`（12） | version list help split-acpi migrate-device-model create-device destroy-device reconfigure-device online-memory offline-memory disable-device enable-device                                                                                                                                                                                                                                                                                                                    |
| `ndctl`（34）  | version enable-namespace disable-namespace create-namespace destroy-namespace read-infoblock write-infoblock check-namespace clear-errors enable-region disable-region enable-dimm disable-dimm zero-labels read-labels write-labels init-labels check-labels inject-error update-firmware inject-smart wait-scrub activate-firmware start-scrub setup-passphrase update-passphrase remove-passphrase freeze-security sanitize-dimm load-keys wait-overwrite list monitor help |

按族分组的完整说明见 §2.1（cxl 五族）/ §2.3（daxctl）/ §2.4（ndctl 四族）。
