# GPU 运维与监控

## 1. 概述

从「看一眼 GPU 状态」到「定位硬件故障」，本目录按场景组织 GPU 日常运维的完整工具链：查询设备能力 → 解读利用率 → 实时监控 → 健康检查 → 进程管理 → 驱动故障排查。

需要特别提醒的是：**别把 `nvidia-smi` 里的 GPU-Util 当成算力使用率**。它只表示"某段时间有 Kernel 在跑"，至于是 1 个 SM 在跑还是全部 SM 都在跑，这个数字完全看不出来。

---

## 2. 文档

| 场景       | 文档                                                     | 内容                                                                           |
| ---------- | -------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 查设备能力 | [GPU 设备属性查询](01_device_query.md)                   | CUDA Runtime/Driver API 查询 SM 数、最大线程数、共享内存等 Kernel 设计关键参数 |
| 看利用率   | [GPU 利用率是一个误导性指标](02_gpu_utilization_myth.md) | 为什么 GPU-Util ≠ 算力利用率（原文翻译）                                       |
| 判断忙不忙 | [GPU 忙不忙怎么判断](09_gpu_busy_check.md)               | 三行命令替代 GPU-Util——SM 利用率 + 显存带宽速决流程                            |
| 日常运维   | [nvidia-smi 场景速查](03_nvidia_smi_guide.md)            | 按场景组织——扫一眼、谁在用、跑满没、健康检查、XID 错误、MIG、脚本化            |
| 实时观察   | [nvtop 监控工具](04_nvtop_guide.md)                      | 交互式 TUI，在终端里实时观察多卡负载                                           |
| 长期趋势   | [DCGM 监控实操](05_dcgm_monitoring.md)                   | NVIDIA 官方数据中心级方案，含 dmon、NVLink、Prometheus 集成                    |
| 故障排查   | [GPU 集群健康检查](06_gpu_health_check.md)               | L1 扫一眼 / L2 结构诊断 / L3 压力验证，含 GPU 7 异常案例                       |
| 资源管理   | [GPU 进程与资源管理](07_gpu_process_management.md)       | Compute Mode、MPS、CUDA_VISIBLE_DEVICES、NUMA 亲和性、显存泄漏                 |
| 驱动故障   | [GPU 驱动故障速查](08_gpu_driver_troubleshooting.md)     | nvidia-smi 不可用时的排查：驱动未加载、版本不匹配、DKMS、PCIe AER              |

---

## 3. 相关资源

- [GPU 架构文档](../../01_hardware_architecture/README.md) —— 向下看硬件层结构。
- [NCCL 通信测试](../03_nccl/README.md) —— 多卡工作正常后，再往分布式通信方向延伸。
- [性能分析工具](../../02_gpu_programming/04_profiling/README.md) —— 想从"有没有在跑"深入到"跑得好不好"的时候，切到这条线。
