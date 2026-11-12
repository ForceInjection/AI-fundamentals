# 量化 GEMM 算子：从 Marlin 到 Humming

量化推理的算力主力是反量化 GEMM：权重压到低比特后，kernel 要在矩阵乘之前把每个数「还原」回来，还原的开销藏得好不好直接决定 decode 提速的上限。本系列沿一条选择轴拆解三代实现：为单一组合手写到极致的 Marlin，用 JIT 编译接住格式组合爆炸的 Humming（蚂蚁集团 + vLLM 社区），以及生成产物内部 tcgen05 数据流的实体。

三篇的行号锚点分别基于 vLLM `43d691ec6b` 与 humming `c62330a`（2026-11-12 核对）。系列性能数字均为论文/博客作者方口径，本仓无 NVIDIA GPU，未做复测。

## [Marlin 深拆：反量化不是免费，但可以藏进访存里](01_marlin-deep-dive.md)

W4A16 记号体系起步，拆反量化的三笔税（计算/布局/带宽）与对应的三个设计：repack 布局是纯置换（读序即布局）、lop3 位技巧把还原压到一条指令（0x6400 魔数 + 零点折叠代数）、双缓冲流水线藏带宽。配套 [marlin_toy.py](marlin_toy.py) 在 CPU 上复现并逐项验证，论文口径最高约 3.87×。

## [Humming 深拆：不预制 kernel，把组合爆炸交给 JIT](02_humming-arch-jit.md)

约 500 KB 的 wheel 一个预编译 kernel 都不带，首次推理才按需编译（单个约 2 秒）。拆统一量化表示（权重五参数独立组合）、三套配置（Layer/Compute/Tuning 即 dataclass 基类）、JIT 链路七环节（签名含全部头文件修改时间、NVRTC 头 shim、原子落盘、cuModuleLoad），以及全文最特别的一步：给编译产物打 SASS 位补丁，解锁 PTX 类型系统里不存在的隐藏浮点格式 E0M3/E3M4。性能对比表（vs Marlin/DeepGEMM）与端到端数据均为博客口径。

## [UMMA 深拆：tcgen05 上，反量化结果写进 TMEM](03_umma-deep-dive.md)

进到篇 2 生成产物的内部：Blackwell tcgen05 把累加器与操作数搬进独立的 TMEM，kernel 组织随之变成五类线程的流水分工（装载/发射/math/反量化 warpgroup/epilogue）。拆 TMEM 512 列布局账本、单 warp 异步连发与描述符格式声明（UMMA 路线不需要 SASS 补丁）、反量化 warpgroup 直写操作数地盘、DP + Stream-K 混合调度与 MoE 两形态。

## 配套代码

- [marlin_toy.py](marlin_toy.py)：仅依赖 numpy，`python3` 直接跑。四组自验证对应篇 1 的核心设计：repack 纯置换（索引追踪）、nibble 打包序、0x6400 魔数位技巧、GPU 存储序 GEMM 等价性（与朴素路径误差 FP32 舍入级）。

## 对读

- [把权重压到 4-bit：TileLang 反量化 GEMM](../03_tilelang/03_dequant_gemm_lowprecision.md)：同族 LOP3/twiddling 解码在 DSL 里的写法，附 H100 实测
- [launch、occupancy、overlap：什么时候手写 kernel](../05_compiler/01_kernel-compiler-megakernel.md)：谱系上的第三条路线（kernel 编译器搜索），与本系列的手写路线、JIT 组装路线互补
