# JuiceFS 分布式文件系统

## 1. 概述

AI 场景下的存储需求是矛盾的：模型和检查点动辄几十上百 GB，然后要在第二天重新被几十张 GPU 同时读取；数据集动辄上亿张小文件，却希望读起来像本地盘一样快。**JuiceFS** 把元数据和数据拆开：元数据放到数据库里拿 POSIX 语义，数据本体放到对象存储里拿容量和持久化，再通过本地缓存和多级缓存把热数据留在靠近计算的地方——这是当前云原生分布式文件系统最成熟的一种解法。

## 2. 核心特性

这些特性并非平行罗列，而是围绕“元数据与数据分离”这条主线展开的：

- **元数据与数据分离**：元数据存储在数据库中，数据存储在对象存储中
- **POSIX 兼容**：完全兼容 POSIX 文件系统接口
- **高性能**：本地缓存和多级缓存优化
- **高可用**：分布式架构，无单点故障
- **弹性扩展**：存储容量可无限扩展
- **多后端支持**：支持 Amazon S3、阿里云 OSS、腾讯云 COS 等
- **强一致性**：提供强一致性保证
- **数据压缩**：支持多种压缩算法
- **数据加密**：传输和存储加密
- **快照备份**：支持快照和增量备份

## 3. 相关文档

### 3.1 技术文档

- [JuiceFS 后端存储变更手册](./02_juicefs_backend_storage_migration_guide.md)：详细介绍如何平滑迁移和变更底层的对象存储。
- [JuiceFS 文件修改时 chunk/slice/block 更新机制分析](./01_juicefs_file_modification_mechanism_analysis.md)：深入分析 JuiceFS 是如何处理文件修改和数据一致性的。

### 3.2 官方资源

- [JuiceFS 官方文档](https://juicefs.com/docs/)
- [JuiceFS GitHub](https://github.com/juicedata/juicefs)
- [JuiceFS 社区](https://juicefs.com/en/community/)
