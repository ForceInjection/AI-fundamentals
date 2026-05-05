# GPU内存虚拟化技术增强模块

## 概述

GPU 显存贵、小、且不容许超分——这三件事把多租户 AI 平台逼到一个困境：任务要多、显存不够分。本模块用一组协同的技术来缓解这个紧张关系：压缩与交换把不热的显存换出去，超分机制让一张卡的逻辑容量超过物理容量，统一地址空间把 CPU/GPU 内存合并寻址，QoS 和 NUMA 感知把带宽和延迟的分配管起来；紧张之下还需要碎片整理、热迁移和故障恢复让共享状态不至于一出问题就雪崩。整体目标是在不破坏隔离的前提下，让一张卡的显存真正被几个任务共用起来。

## 核心功能

九个模块按“先把容量省出来、再把地址和带宽管起来、最后让共享状态经得起故障”的顺序组织。每个模块对应一个源文件，代码可独立编译、独立演示，也可以组合起来形成一套完整的显存虚拟化栈。

### 容量：把“不够分”变成“还能撑一撑”

显存天生紧张，这一组模块要回答的是：**不增加硬件的前提下，还能挤出多少可用容量？** 三条路径互相配合——冷数据压缩、热度分层交换、加一层逻辑超分。

#### 1. 内存压缩 (`memory_compression.c`)

思路是把冷数据压成更小的块，空出物理显存给热任务：

- **多算法支持**：LZ4、ZSTD、Snappy 三种算法，速度与压缩率取舍不同
- **自适应选择**：根据数据特征自动切换算法，无需人工标注数据类型
- **并行压缩**：多线程流水线并行，压缩本身不成为瓶颈
- **实时统计**：压缩率、吞吐、延迟等指标暴露给上层调度

```c
int init_compression_system(compression_algorithm_t algorithm,
                            compression_quality_t quality, bool parallel_enabled);
int compress_memory(const void *input, size_t input_size,
                    void **output, size_t *output_size, compression_algorithm_t algorithm);
int decompress_memory(const void *input, size_t input_size,
                      void **output, size_t *output_size, compression_algorithm_t algorithm);
```

#### 2. 内存交换 (`memory_swap.c`)

压缩救不了的那部分，就往更慢但更大的存储里换：

- **多级存储**：系统内存、NVMe、SSD、HDD 逐级接力
- **智能预取**：基于访问模式预测，在真正用到之前把数据拉回来
- **异步交换**：换出/换入在后台跑，尽量不挡前台请求
- **LRU 热度管理**：自动把冷数据推到下一层

```c
int init_swap_system(size_t max_gpu_memory, size_t page_size, double swap_threshold);
void* allocate_gpu_memory(size_t size);
void free_gpu_memory(void *ptr);
void* access_gpu_memory(void *ptr);
```

#### 3. 内存过量分配 (`memory_overcommit_advanced.c`)

在压缩和交换的基础上，把“逻辑显存总量”开得比物理显存还大：

- **基于历史模式**：观测任务真实占用，超分比例有依据而不是拍脑袋
- **与压缩联动**：超分之后的热点由压缩模块消化，而不是直接 OOM
- **优先级回收**：显存紧张时按优先级决定谁先被换出
- **细粒度统计**：每个租户的实际占用与申请值分开核算

```c
int init_memory_overcommit(size_t physical_memory_size, double overcommit_ratio);
void* allocate_overcommit_memory(size_t size, int priority);
void free_overcommit_memory(void *ptr);
void print_overcommit_stats(void);
```

### 寻址与 QoS：让共享不再互相踩

容量挤出来之后，多任务共用一张卡的下一个问题是：**地址怎么统一、带宽怎么分？** 前者让 CPU/GPU 可以共享同一套指针，后者让吵闹的邻居不把别人拖垮。

#### 4. 统一地址空间 (`unified_address_space.c`)

跨设备共享一套虚拟地址，省去大量显式拷贝：

- **跨设备访问**：GPU 和 CPU 之间共享同一虚拟地址空间
- **多种内存类型**：Host、Device、Managed、Pinned 按需选择
- **权限控制**：读/写/执行权限按页粒度控制
- **高效地址转换**：虚拟到物理地址的翻译路径做了优化

```c
int init_unified_address_space(size_t total_size, size_t page_size);
void* allocate_unified_memory(size_t size, memory_type_t type, access_permission_t permissions);
void* access_unified_memory(void *ptr, access_permission_t required_permissions);
int sync_memory_region(void *ptr, size_t size);
```

#### 5. 内存 QoS (`memory_qos.c`)

带宽和延迟是显存共享里最容易被忽略、又最容易出故障的维度：

- **带宽控制**：令牌桶算法限制单租户对显存带宽的占用
- **延迟 SLA**：不同优先级对应不同延迟上限
- **优先级调度**：多级队列避免高优任务被低优任务堵死
- **自适应调整**：根据实时负载动态调整配额

```c
int init_memory_qos(uint32_t total_bandwidth_mbps, bandwidth_policy_t policy);
int submit_memory_request(void *address, size_t size, memory_access_type_t type,
                          qos_level_t qos_level, uint32_t client_id);
void get_qos_stats(void);
```

### 局部性与可靠性：让共享状态经得起折腾

容量和带宽都管好了，最后一公里是：**长时间共享之后，显存会碎、访问会变慢、硬件会出错。** 这组模块负责让共享状态不因为时间和意外而崩掉。

#### 6. 碎片整理 (`memory_defragmentation.c`)

长期分配/释放会让可用空间碎成小块，大块申请开始失败：

- **在线碎片整理**：运行时执行，不需要停服务
- **相邻块合并**：自动识别并合并相邻空闲块
- **内存重排**：优化布局，让热数据更靠近、访问更连续
- **碎片率监控**：实时指标，超过阈值自动触发整理

#### 7. NUMA 感知 (`numa_aware_memory.c`)

多路 CPU/多卡服务器上，“分配在哪个节点”对延迟影响很大：

- **NUMA 拓扑感知**：启动时自动探测节点结构
- **本地优先分配**：尽量把内存分配在访问它的 CPU/GPU 同节点
- **按访问模式迁移**：发现跨节点频繁访问时主动迁移页
- **带宽与延迟联合优化**：既不浪费本地带宽，也不让跨节点访问拖后腿

#### 8. 热迁移 (`memory_hot_migration.c`)

节点要维护、负载要重平衡时，不能让任务停下来：

- **在线迁移**：应用无感知地把显存搬到另一张卡或另一台机器
- **增量迁移**：仅传输脏页，大幅降低总传输量
- **一致性保障**：迁移过程中保证读写视图一致
- **多策略可选**：预复制、后复制两种经典模式都支持
- **失败可回滚**：迁移失败自动恢复到源端，不丢数据

#### 9. 故障恢复 (`memory_fault_recovery.c`)

显存硬件并非绝对可靠，ECC 错、掉卡、总线异常都要有预案：

- **ECC 错误处理**：自动检测并按类型分级处理
- **检查点机制**：周期性快照，故障后可以回滚到最近一致点
- **自动故障转移**：检测到不可恢复错误时切换到备用资源
- **故障预测**：基于历史错误模式提前预警
- **多种恢复策略**：不同业务场景下可选不同 RTO/RPO 策略

---

## 编译和安装

### 依赖项

核心依赖：`pthread`、`lz4`、`zstd`（压缩）。NUMA 模块额外需要 `libnuma`。Makefile 中的填充实现预留了 Snappy 算法的枚举入口但未默认链接，如需启用在编译时自行补上 `-lsnappy` 即可。

```bash
# Ubuntu/Debian
sudo apt-get install build-essential liblz4-dev libzstd-dev libnuma-dev

# CentOS/RHEL
sudo yum install gcc gcc-c++ lz4-devel libzstd-devel numactl-devel

# macOS
brew install lz4 zstd
```

> NUMA 相关接口仅在 Linux 下可用；macOS 环境可编译除 `numa_demo` 之外的模块。

### 编译

```bash
# 编译所有组件
make all

# 仅编译库
make lib/libmemory_virtualization.so

# 编译演示程序
make bin/memory_virtualization_demo

# 查看编译信息
make info
```

### 安装

```bash
# 安装到系统目录
sudo make install

# 创建发布包
make dist
```

---

## 使用示例

### 基本演示

```bash
# 运行基本演示
make run-demo

# 运行性能基准测试
make run-benchmark

# 运行压力测试
make run-stress
```

### 单模块测试

基础四件套（压缩 / 交换 / 统一地址空间 / QoS）通过 Makefile 的 `--no-xxx` 开关组合测试：

```bash
# 仅测试压缩模块
make run-compression-only

# 仅测试交换模块
make run-swap-only

# 仅测试统一地址空间
make run-unified-only

# 仅测试 QoS 模块
make run-qos-only
```

碎片整理、NUMA、热迁移、故障恢复四个高级模块则通过 `advanced_memory_demo` 的子命令单独演示（见下一节）。

### 高级功能综合演示

```bash
# 运行所有高级功能演示
./bin/advanced_memory_demo

# 运行特定功能演示
./bin/advanced_memory_demo defrag      # 仅演示内存碎片整理
./bin/advanced_memory_demo numa        # 仅演示 NUMA 感知内存管理
./bin/advanced_memory_demo migration   # 仅演示内存热迁移
./bin/advanced_memory_demo fault       # 仅演示内存故障恢复
./bin/advanced_memory_demo performance # 综合性能测试
```

### 自定义参数

```bash
# 自定义线程数和测试时间
./bin/memory_virtualization_demo --threads 8 --duration 60 --size 2048

# 禁用特定模块
./bin/memory_virtualization_demo --no-compression --no-swap

# 查看帮助
./bin/memory_virtualization_demo --help
```

---

## 代码集成

### C/C++项目集成

```c
#include "memory_virtualization.h"

int main() {
    // 初始化压缩系统
    init_compression_system(COMPRESS_LZ4, QUALITY_BALANCED, true);

    // 初始化交换系统
    init_swap_system(512 * 1024 * 1024, 4096, 0.8);

    // 初始化 NUMA 感知内存管理
    init_numa_memory_manager();

    // 初始化碎片整理（30% 碎片率阈值）
    init_defragmenter(0.3);

    // 分配 GPU 内存
    void *gpu_mem = allocate_gpu_memory(1024 * 1024);

    // 使用内存...

    // 清理
    free_gpu_memory(gpu_mem);
    cleanup_defragmenter();
    cleanup_numa_memory_manager();
    cleanup_swap_system();
    cleanup_compression_system();

    return 0;
}
```

### 编译链接

```bash
gcc -o myapp myapp.c -lmemory_virtualization -lpthread -lm -lz -llz4 -lzstd
```

如果用到 NUMA 模块（`numa_aware_memory.c`），额外链上 `-lnuma`。

---

## 性能优化建议

### 1. 压缩算法选择

- **LZ4**: 高速压缩，适合实时场景
- **ZSTD**: 平衡压缩率和速度
- **Snappy**: Google开发，适合大数据场景
- **自适应**: 根据数据特征自动选择

### 2. 交换策略配置

```c
// 高性能配置
init_swap_system(
    1024 * 1024 * 1024,  // 1GB GPU内存
    4096,                // 4KB页面大小
    0.9                  // 90%交换阈值
);

// 低延迟配置
init_swap_system(
    512 * 1024 * 1024,   // 512MB GPU内存
    2048,                // 2KB页面大小
    0.7                  // 70%交换阈值
);
```

### 3. QoS配置优化

```c
// 高吞吐量配置
init_memory_qos(20000, BANDWIDTH_POLICY_FAIR);

// 低延迟配置
init_memory_qos(10000, BANDWIDTH_POLICY_PRIORITY);

// 自适应配置
init_memory_qos(15000, BANDWIDTH_POLICY_ADAPTIVE);
```

---

## 监控和调试

### 性能监控

```c
// 打印压缩统计
print_compression_stats();

// 打印交换统计
print_swap_stats();

// 打印地址空间统计
print_address_space_stats();

// 打印 QoS 统计
get_qos_stats();

// 打印碎片整理统计
print_defrag_stats();

// 打印 NUMA 统计
print_numa_stats();

// 打印故障恢复统计
print_fault_recovery_stats();
```

> 热迁移模块目前未暴露独立的统计打印函数，需要直接读取 `memory_hot_migration.c` 中的会话状态或扩展 `advanced_memory_demo.c` 来采集指标。

### 调试工具

```bash
# 静态代码分析
make check

# 内存泄漏检测
make valgrind-check

# 生成文档
make docs
```

---

## 故障排除

### 常见问题

1. **编译错误**
   - 检查依赖库是否安装
   - 确认编译器版本支持C99标准

2. **运行时错误**
   - 检查系统内存是否充足
   - 确认GPU驱动程序正常

3. **性能问题**
   - 调整压缩算法和质量设置
   - 优化交换阈值和页面大小
   - 检查QoS配置是否合理

### 日志分析

```bash
# 启用详细日志
export MEMORY_VIRT_LOG_LEVEL=DEBUG
./bin/memory_virtualization_demo

# 分析性能日志
grep "Performance" /var/log/memory_virt.log
```

---

## 技术架构

### 模块分层

九个模块分布在三层：资源压力层（压缩 / 交换 / 超分）向上提供逻辑容量；共享调度层（统一地址空间 / QoS）解决多租户寻址与带宽分配；可靠性层（碎片整理 / NUMA / 热迁移 / 故障恢复）用来应对长时间共享之后不可避免的碎片、局部性退化和硬件故障。

```text
┌─────────────────────────────────────────────────────────────┐
│                      应用程序接口层                         │
├─────────────────────────────────────────────────────────────┤
│  碎片整理  │  NUMA 感知  │  热迁移  │  故障恢复             │
├─────────────────────────────────────────────────────────────┤
│        统一地址空间        │            内存 QoS            │
├─────────────────────────────────────────────────────────────┤
│   压缩模块   │   交换模块   │      内存过量分配（超分）      │
├─────────────────────────────────────────────────────────────┤
│                      GPU 硬件抽象层                         │
└─────────────────────────────────────────────────────────────┘
```

### 数据流

1. **内存分配请求** → 过量分配管理 → 物理内存分配
2. **内存访问** → 地址转换 → 权限检查 → 数据访问
3. **内存压力** → 交换决策 → 压缩存储 → 后台交换
4. **QoS 请求** → 优先级队列 → 带宽控制 → 执行调度

---

## 二次开发

### 添加新的压缩算法

```c
// 在memory_compression.c中添加
typedef enum {
    // 现有算法...
    COMPRESS_NEW_ALGORITHM
} compression_algorithm_t;

// 实现压缩函数
static int compress_with_new_algorithm(const void *input, size_t input_size,
                                      void **output, size_t *output_size) {
    // 实现新算法
}
```

### 添加新的存储层次

```c
// 在memory_swap.c中添加
typedef enum {
    // 现有存储类型...
    SWAP_NEW_STORAGE_TYPE
} swap_storage_type_t;

// 实现存储操作
static int init_new_storage_backend(const char *config) {
    // 初始化新存储后端
}
```

---

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

---

## 贡献指南

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/new-feature`)
3. 提交更改 (`git commit -am 'Add new feature'`)
4. 推送到分支 (`git push origin feature/new-feature`)
5. 创建Pull Request

---

## 版本历史

### v1.0.0 (当前版本)

- 初始发布
- 实现基础内存压缩功能
- 实现内存交换机制
- 实现统一地址空间管理
- 实现内存QoS保障
- 实现内存过量分配
- 实现内存碎片整理
- 实现NUMA感知内存管理
- 实现内存热迁移
- 实现内存故障恢复

### 计划功能

- v1.1.0: 支持多GPU内存池
- v1.2.0: 添加机器学习优化算法
- v1.3.0: 支持容器化部署
- v2.0.0: 支持异构计算环境
