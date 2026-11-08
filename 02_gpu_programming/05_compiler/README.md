# Kernel 编译器与 Megakernel

手写 kernel 到顶之后，性能还藏在三个地方：层级之间的联合重排、kernel 之间的边界、算子与硬件的乘积空间。本目录收录沿这三个方向接管优化的编译器路线。

- [launch、occupancy、overlap：什么时候手写 kernel，什么时候交给编译器](01_kernel-compiler-megakernel.md)：TVM（schedule 分离）、Mirage（三层联合超优）、MPK（整模型 megakernel）三级递进，配三 regime 决策轴与 launch-bound 微基准；事实核对基于三篇论文与 mirage-project 源码（2026-11-08）
- [01_launch_overhead_bench.cu](01_launch_overhead_bench.cu)：launch-bound 微基准，eager / CUDA Graph / fused 三模式等量工作对比，实测口径见文章 §六
