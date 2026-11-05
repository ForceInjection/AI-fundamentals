# TileLang 与 Tile-Based 编程

当 Tensor Core 这样的专用加速单元成为现代 GPU 上的主力计算单元后，传统 SIMT 从单个线程视角组织计算的思路有些别扭：手写一份逼近硬件极限的 GEMM 或 FlashAttention，往往要拼 MMA 指令、Swizzle 排布和异步拷贝的大量底层细节。

Tile-Based 的思路是把视角从"线程"上移到"Tile（数据块）"，让开发者直接用更贴近张量计算的抽象描述算子，底层细节交给编译器处理。

TileLang 是 tile-ai 开源的面向 GPU/CPU/NPU 高性能 Kernel 的领域特定语言（DSL）及编译器架构：以 Tile 为基本操作单元，保持 Pythonic 语法的易用性，分块、流水线与指令选择由编译器完成，可生成 CUDA/HIP/Metal 等多种后端代码。

> **前置条件**（tilelang v0.1.15，2026-11 核对）：Python ≥ 3.10、CUDA ≥ 10.0（主机安装）或 pip 提供的 CUDA 工具链 ≥ 13.0、PyTorch；AMD GPU 需主机 ROCm。后端覆盖 NVIDIA CUDA、AMD ROCm、Apple Metal、华为 Ascend、LLVM CPU 等。

## [TileLang 快速入门](01_tilelang_quick_start.md)

从环境搭建到第一个矩阵乘法 Kernel：安装（PyPI / 源码 / Nightly）、核心 API 速览（`T.Kernel`、`T.alloc_shared`、`T.alloc_fragment`、`T.gemm`、`T.Pipelined`）、CUDA 到 TileLang 对照表，以及 JIT 编译流程和调试技巧。最终用约 30 行代码实现一个 GEMM 算子。

## [TileLang FlashAttention 前向拆解](02_tilelang_flash_attention.md)

以 examples/flash_attention 的五个前向例子为底，拆解 57 行 kernel 的公共骨架与 online softmax 五状态递推：mask 预填与 `T.gemm` 累加语义的配合、log2(e) 折进 scale、两个实测坑（fragment 直写全局张量被 StructuralEqual 拒绝、fp16 累加器精度），以及 wgmma_pipelined 变体从手工 order/stage/group 到 warp specialization 的演进。附 H100 实测（非 causal 全量口径 248/360 TFLOPs）与源文件索引。

## [把权重压到 4-bit：TileLang 反量化 GEMM 源码级拆解](03_dequant_gemm_lowprecision.md)

以 examples/dequantize_gemm 的七个 weight-only 例子为底，拆解反量化的两条路径：寄存器路径（w4a8/fp4/gemv 的 TIR 逐元素解码直喂 `T.gemm`）与 shared memory 路径（bf16_fp4/mxfp4/fine_grained 的 128-bit 取数加 extern intrinsic 批量解码），以及 LOP3、fp4→bf16 twiddling 两族快速解码和 interleave_weight 位重排。三个坑：fp4 次正规 0.5 被映成 0.75、mxfp4 docstring 与代码的 scale 语义不一致、interleave_weight 的 1/2-bit 分支 TypeError；wgmma SS/RS 发射条件的逐 kernel 推断。附 H100 实测（4096 立方推算吞吐最高约 529 TFLOPS）与源文件索引。
