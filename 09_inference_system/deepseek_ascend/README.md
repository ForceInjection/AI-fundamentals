# DeepSeek 昇腾开源栈

DeepSeek 把生产栈搬上了昇腾：六个组件与英伟达版一一对应，全线绑定 Ascend 950（`dav-3510`），GEMM 自报 99%+ 硬件上限，复现依赖商用 HDK。本目录是该方向的首篇深读与 950 微架构资料存档。

## 内容导航

- **[DeepSeek 把家底搬上了昇腾：六个组件、三种双端模式与一份 950 微架构报告](deepseek-ascend-open-source.md)**：TileLang / DeepGEMM / DeepEP / TileKernels / FlashMLA / DeepSelect 逐仓拆解，三种双端模式与 950 微架构要点，含复现依赖与诚实缺口清单（配图 3 张，在 `img/`）。

## 相关链接

- [Ascend950 微架构速查](../../01_hardware_architecture/ascend/Ascend950_微架构速查.md)：硬件信息整理
- [Ascend950 稀疏注意力 Forward 算子与优化技术简析](../../01_hardware_architecture/ascend/Ascend950_稀疏注意力Forward算子与优化技术简析.md)：一手资料存档（转载）
- [DeepSeek-V3 端到端推理 Pipeline 走读](../vllm/module_analysis/deepseek_v3_inference_pipeline.md) · [DeepSeek-V4 端到端推理 Pipeline 走读](../vllm/module_analysis/deepseek_v4_inference_pipeline.md)：英伟达侧的对照实现
