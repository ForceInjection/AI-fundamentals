# 进阶主题

本模块覆盖两个进阶话题：Ascend C 自定义算子开发（对标 CUDA kernel 编写）和 GPU 到 NPU 的迁移决策框架。前者回答 "自定义 CUDA kernel 怎么搬到 NPU 上"，后者回答 "什么时候值得做这个迁移"。

## 1. [Ascend C 算子开发入门](01_ascend_c_intro.md)

昇腾提供了三层算子开发体系：内置算子（OPP，直接调用）→ Ascend C（C++ 类库，自定义算子）→ TBE（DSL，深度优化）。本文聚焦中间层 Ascend C，覆盖开发流程、工具链（`msopgen` + `ccec`）和一个简单的 ReLU 算子示例。

## 2. [GPU 到 NPU 的迁移策略](02_gpu_to_npu_migration.md)

迁移决策树从三个维度评估：模型类型（标准 vs 自定义）、算子复杂度（有无 CUDA kernel）、生态依赖（NCCL/Triton/TensorRT）。附成本估算和不建议迁移的场景清单。
