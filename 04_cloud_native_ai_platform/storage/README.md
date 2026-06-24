# AI 存储——训练 I/O 和推理 KV Cache 的数据底座

训练一轮次要读上亿张小图片、检查点动辄几十上百 GB、推理要持续命中 KV Cache——AI 工作负载把存储推到了一个尴尬的位置：既要像本地盘一样快，又要像对象存储一样能扩。

本目录覆盖三种思路：

| 系统 | 定位 | 核心思路 |
|------|------|---------|
| [JuiceFS](juicefs/README.md) | 通用分布式文件系统 | 数据与元数据分离，兼容 POSIX，对接 S3/HDFS/MinIO 等后端 |
| [DeepSeek 3FS](deepseek_3fs/01_deepseek_3fs_design_notes.md) | 训练专用高性能存储 | 面向大规模模型训练的并行文件系统架构 |
| [ICMS](inference_context_memory_storage/01_icms_architecture.md) | 推理 KV Cache 存储 | NVIDIA 推理上下文内存存储架构解析 |

## 文档

### JuiceFS

- [JuiceFS 核心架构](juicefs/README.md)
- [文件修改机制分析](juicefs/01_juicefs_file_modification_mechanism_analysis.md)
- [后端存储迁移指南](juicefs/02_juicefs_backend_storage_migration_guide.md)

### DeepSeek 3FS

- [DeepSeek 3FS 设计笔记](deepseek_3fs/01_deepseek_3fs_design_notes.md)

### ICMS

- [ICMS 架构解析](inference_context_memory_storage/01_icms_architecture.md) — NVIDIA Rubin 平台的推理上下文存储层 (G3.5)

## 相关资源

- [NCCL 分布式通信测试](../../03_ai_cluster_ops/03_nccl/README.md) — 存储性能直接影响训练 I/O，而 NCCL 决定了 GPU 间的通信
- [KV Cache 技术体系](../../09_inference_system/kv_cache/README.md) — 推理 KV Cache 与 ICMS 互补
