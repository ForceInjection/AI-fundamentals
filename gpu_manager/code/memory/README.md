# GPU内存虚拟化技术增强模块

## 概述

本模块实现了GPU内存虚拟化的高级功能，包括内存压缩、交换、统一地址空间管理和QoS保障等技术。这些技术可以显著提高GPU内存利用率，支持内存过量分配，并为多租户环境提供性能隔离保障。

## 核心功能

### 1. 内存压缩技术 (`memory_compression.c`)

- **多算法支持**: LZ4、ZSTD、Snappy等压缩算法
- **自适应压缩**: 根据数据特征自动选择最优算法
- **并行压缩**: 支持多线程并行压缩以提高性能
- **实时统计**: 压缩率、性能指标实时监控

### 6. 内存碎片整理 (`memory_defragmentation.c`)

- **在线碎片整理**: 运行时减少内存碎片
- **智能压缩**: 自动合并相邻空闲块
- **内存重排**: 优化内存布局提高访问效率
- **碎片率监控**: 实时跟踪内存碎片化程度
- **自动触发**: 基于阈值的自动碎片整理

### 7. NUMA感知内存管理 (`numa_aware_memory.c`)

- **NUMA拓扑感知**: 自动检测和适配NUMA架构
- **智能内存分配**: 优先在本地节点分配内存
- **跨节点迁移**: 基于访问模式的内存迁移
- **带宽优化**: 最大化内存带宽利用率
- **延迟最小化**: 减少跨节点访问延迟

### 8. 内存热迁移 (`memory_hot_migration.c`)

- **在线迁移**: 无需停止应用的内存迁移
- **增量迁移**: 只迁移变化的数据
- **一致性保障**: 确保迁移过程中数据一致性
- **多种策略**: 支持预复制、后复制等策略
- **故障恢复**: 迁移失败时的自动恢复

### 9. 内存故障恢复 (`memory_fault_recovery.c`)

- **ECC错误处理**: 自动检测和处理ECC错误
- **内存检查点**: 定期创建内存快照
- **自动故障转移**: 故障时自动切换到备用资源
- **故障预测**: 基于历史数据预测潜在故障
- **恢复策略**: 多种故障恢复策略选择

**主要API:**

```c
int init_compression_system(compression_algorithm_t algorithm, 
                           compression_quality_t quality, bool parallel_enabled);
int compress_memory(const void *input, size_t input_size, 
                   void **output, size_t *output_size, compression_algorithm_t algorithm);
int decompress_memory(const void *input, size_t input_size,
                     void **output, size_t *output_size, compression_algorithm_t algorithm);
```

### 2. 内存交换机制 (`memory_swap.c`)

- **多级存储**: 支持系统内存、SSD、HDD、NVMe等存储层次
- **智能预取**: 基于访问模式的预测性数据加载
- **异步交换**: 后台异步执行交换操作，减少延迟
- **热度管理**: LRU算法管理热点数据

**主要API:**

```c
int init_swap_system(size_t max_gpu_memory, size_t page_size, double swap_threshold);
void* allocate_gpu_memory(size_t size);
void free_gpu_memory(void *ptr);
void* access_gpu_memory(void *ptr);
```

### 3. 统一地址空间管理 (`unified_address_space.c`)

- **跨设备访问**: GPU和CPU之间的统一虚拟地址空间
- **内存类型管理**: Host、Device、Managed、Pinned内存类型
- **权限控制**: 细粒度的读写执行权限管理
- **地址转换**: 高效的虚拟到物理地址转换

**主要API:**

```c
int init_unified_address_space(size_t total_size, size_t page_size);
void* allocate_unified_memory(size_t size, memory_type_t type, access_permission_t permissions);
void* access_unified_memory(void *ptr, access_permission_t required_permissions);
int sync_memory_region(void *ptr, size_t size);
```

### 4. 内存QoS保障 (`memory_qos.c`)

- **带宽控制**: 基于令牌桶的带宽限制
- **延迟保障**: 不同优先级的延迟SLA保证
- **优先级调度**: 多级优先级队列调度
- **自适应调整**: 根据负载动态调整资源分配

**主要API:**

```c
int init_memory_qos(uint32_t total_bandwidth_mbps, bandwidth_policy_t policy);
int submit_memory_request(void *address, size_t size, memory_access_type_t type, 
                         qos_level_t qos_level, uint32_t client_id);
void get_qos_stats(void);
```

### 5. 内存过量分配 (`memory_overcommit_advanced.c`)

- **智能过量分配**: 基于历史使用模式的内存过量分配
- **压缩集成**: 与压缩技术集成，提高内存利用率
- **优先级管理**: 基于优先级的内存回收策略
- **统计监控**: 详细的内存使用统计和监控

**主要API:**

```c
int init_memory_overcommit(size_t physical_memory_size, double overcommit_ratio);
void* allocate_overcommit_memory(size_t size, int priority);
void free_overcommit_memory(void *ptr);
void print_overcommit_stats(void);
```

## 编译和安装

### 依赖项

```bash
# Ubuntu/Debian
sudo apt-get install build-essential liblz4-dev libzstd-dev libsnappy-dev libnuma-dev

# CentOS/RHEL
sudo yum install gcc gcc-c++ lz4-devel libzstd-devel snappy-devel numactl-devel

# macOS
brew install lz4 zstd snappy numactl
```

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

```bash
# 仅测试压缩模块
make run-compression-only

# 仅测试交换模块
make run-swap-only

# 仅测试统一地址空间
make run-unified-only

# 仅测试QoS模块
make run-qos-only

# 仅测试碎片整理模块
make run-defrag-only

# 仅测试NUMA感知模块
make run-numa-only

# 仅测试热迁移模块
make run-migration-only

# 仅测试故障恢复模块
make run-fault-recovery-only
```

### 高级功能综合演示

```bash
# 运行所有高级功能演示
./bin/advanced_memory_demo

# 运行特定功能演示
./bin/advanced_memory_demo defrag      # 仅演示内存碎片整理
./bin/advanced_memory_demo numa        # 仅演示NUMA感知内存管理
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

## 代码集成

### C/C++项目集成

```c
#include "memory_virtualization.h"

int main() {
    // 初始化压缩系统
    init_compression_system(COMPRESS_LZ4, QUALITY_BALANCED, true);
    
    // 初始化交换系统
    init_swap_system(512 * 1024 * 1024, 4096, 0.8);
    
    // 初始化NUMA感知内存管理
    init_numa_memory_manager();
    
    // 初始化碎片整理
    init_defragmentation_manager(0.3, 60); // 30%碎片率阈值，60秒间隔
    
    // 分配GPU内存
    void *gpu_mem = allocate_gpu_memory(1024 * 1024);
    
    // 使用内存...
    
    // 清理
    free_gpu_memory(gpu_mem);
    cleanup_defragmentation_manager();
    cleanup_numa_memory_manager();
    cleanup_swap_system();
    cleanup_compression_system();
    
    return 0;
}
```

### 编译链接

```bash
gcc -o myapp myapp.c -lmemory_virtualization -lpthread -lm -lz -llz4 -lzstd -lnuma
```

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

## 监控和调试

### 性能监控

```c
// 打印压缩统计
print_compression_stats();

// 打印交换统计
print_swap_stats();

// 打印地址空间统计
print_address_space_stats();

// 打印QoS统计
get_qos_stats();

// 打印碎片整理统计
print_defragmentation_stats();

// 打印NUMA统计
print_numa_stats();

// 打印热迁移统计
print_migration_stats();

// 打印故障恢复统计
print_fault_recovery_stats();
```

### 调试工具

```bash
# 静态代码分析
make check

# 内存泄漏检测
make valgrind-check

# 生成文档
make docs
```

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

## 技术架构

### 模块关系图

```text
┌─────────────────────────────────────────────────────────────┐
│                    应用程序接口层                              │
├─────────────────────────────────────────────────────────────┤
│  压缩模块      │  交换模块    │  统一地址空间  │  QoS模块        │
├─────────────────────────────────────────────────────────────┤
│                    内存过量分配管理层                          │
├─────────────────────────────────────────────────────────────┤
│                    GPU硬件抽象层                              │
└─────────────────────────────────────────────────────────────┘
```

### 数据流

1. **内存分配请求** → 过量分配管理 → 物理内存分配
2. **内存访问** → 地址转换 → 权限检查 → 数据访问
3. **内存压力** → 交换决策 → 压缩存储 → 后台交换
4. **QoS请求** → 优先级队列 → 带宽控制 → 执行调度

## 扩展开发

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

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献指南

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/new-feature`)
3. 提交更改 (`git commit -am 'Add new feature'`)
4. 推送到分支 (`git push origin feature/new-feature`)
5. 创建Pull Request

## 联系方式

- 项目主页: <https://github.com/your-org/gpu-memory-virtualization>
- 问题报告: <https://github.com/your-org/gpu-memory-virtualization/issues>
- 邮件联系: <gpu-virt@your-org.com>

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
