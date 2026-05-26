# 华为 NPU 编程入门

系统梳理从昇腾 NPU 硬件特性到上层框架编程的完整知识链路，覆盖**环境搭建 → 架构原理 → 框架实战 → 工具链 → 进阶开发 → RAG 实战 → 性能分析 → Mini-GPT → FlashAttention → LLM 推理 → DDP 多卡训练 → LoRA 微调 → 量化推理**十四大主题。无论读者是从 CUDA 生态迁移而来的 GPU 开发者，还是初次接触 Ascend 的新手，均可按 §2→§4 顺序快速上手，再根据实际需求深入工具链运维或自定义算子开发。

> **快速导航**
>
> | 目录                      | 主题                        | 关键词                                    | 对应章节 |
> | ------------------------- | --------------------------- | ----------------------------------------- | -------- |
> | `01_environment/`         | Ascend NPU 开发环境搭建     | CANN, torch_npu, venv, 版本对齐           | §2       |
> | `02_ascend_architecture/` | Da Vinci 架构与 CANN 软件栈 | Cube/Vector/Scalar, HCCS, 七层协议栈      | §3       |
> | `03_pytorch_npu/`         | PyTorch NPU 适配与训练      | cuda→npu 迁移, AMP, ResNet-50             | §4.1     |
> | `04_mindspore/`           | MindSpore 原生开发框架      | PyNative/Graph, nn.Cell, 动静态图         | §4.2     |
> | `05_tools/`               | Ascend 工具链               | npu-smi, ascend-dmi, ATC 模型转换         | §5       |
> | `06_advanced/`            | 进阶主题                    | Ascend C 自定义算子, GPU→NPU 迁移决策     | §6       |
> | `07_rag_on_npu/`          | RAG 检索增强生成 on NPU     | Embedding, FAISS, BGE, LLM API            | §7       |
> | `08_npu_profiling/`       | NPU 性能分析                | Profiler, npu-smi, TFLOPS, Chrome trace   | §8       |
> | `09_flash_attention/`     | FlashAttention 简化版       | Tiling, Online Softmax, O(N²)→O(N)        | §9       |
> | `10_mini_gpt/`            | Mini-GPT 手写 Transformer   | Self-Attention, Causal Mask, 字符级编码   | §10      |
> | `11_llm_inference/`       | LLM 推理 on NPU             | Qwen2.5 7B BF16, 自回归, ChatML, NaN 诊断 | §11      |
> | `12_ddp/`                 | DDP 多卡分布式训练          | HCCL, AllReduce, 8 卡梯度同步, 14B 全参   | §12      |
> | `13_finetune/`            | LoRA 微调                   | PEFT, SFT v.s. CLM, RAG+SFT 协同, 380 QA  | §13      |
> | `14_quantization/`        | 量化推理 (INT8/INT4)        | 对称/非对称, per-channel, 校准, 精度-效率 | §14      |

---

## 1. 概述

这一章回答一个实际的问题：**从一张空闲的昇腾 NPU 开始，到用 PyTorch 或 MindSpore 跑通模型训练，中间需要了解哪些环节？**

与 GPU 编程以 CUDA 为核心不同，NPU 编程的入口有两条路径：

- **PyTorch NPU 适配**：通过 `torch_npu` 将现有 PyTorch 代码迁移到 Ascend 硬件，API 层面几乎只需将 `cuda` 替换为 `npu`。适合有 GPU/CUDA 经验的开发者快速上手。
- **MindSpore 原生**：华为自研框架，对 Ascend 有原生支持，提供 PyNative（动态图）和 Graph（静态图）两种执行模式，编程范式与 PyTorch 有明显差异。

完整路径拆为三个递进层次：

- **基础层（§2→§6）**：环境搭建 → 架构理解 → 框架实战（PyTorch NPU / MindSpore）→ 工具链掌握 → 进阶算子开发。目标是"在 NPU 上跑通模型训练"。
- **实战层（§7→§11）**：RAG 检索增强生成 → NPU 性能分析 → FlashAttention 手写 → Mini-GPT 从零训练 → LLM 推理部署。目标是"把 NPU 用到生产级任务中"。
- **规模化层（§12→§14）**：DDP 多卡分布式训练 → LoRA 参数高效微调 → INT8/INT4 量化推理。目标是"从单卡走向多卡，从微调走向压缩部署"。

---

## 2. 环境准备

**CANN (Compute Architecture for Neural Networks)** 是昇腾的异构计算架构，在 Ascend 生态中承担与 NVIDIA CUDA 等同的角色——向上提供框架 / 编译器 / 推理引擎依赖的统一 API，向下屏蔽驱动与硬件差异。其核心组成：

| 分层       | 关键组件                                  | CUDA 对标                  |
| ---------- | ----------------------------------------- | -------------------------- |
| 应用使能   | AscendCL、PyTorch 适配 (`torch_npu`)、ATC | CUDA Runtime API、TensorRT |
| 图与算子   | GE 图引擎、TBE / Ascend C 算子开发        | nvFuser、CUTLASS / CUDA C  |
| 运行时     | Runtime、HCCL 集合通信                    | CUDA Driver API、NCCL      |
| 驱动与固件 | NPU Driver / Firmware                     | NVIDIA Driver              |

安装后需 source `set_env.sh` 导入 `LD_LIBRARY_PATH`、`ASCEND_HOME` 等变量，且**须在 venv 激活之前执行**——顺序颠倒会导致 `libhccl.so` / `libascendcl.so` 等动态库解析失败。

- [昇腾环境搭建](01_environment/01_ascend_environment_setup.md) — CANN 环境变量、虚拟环境创建、torch_npu 安装、关键依赖版本对齐、环境变量加载顺序
- [Hello NPU：第一个程序](01_environment/02_hello_npu_first_program.md) — `import torch_npu` 注册后端、CUDA→NPU API 映射速查、4096×4096 矩阵乘法实测 164.6× 加速比

---

## 3. 硬件架构与软件栈

昇腾 910B3 基于**达芬奇** (Da Vinci) 架构，与 NVIDIA 的 SIMT 模型在计算组织方式上有根本差异。理解这种差异是从 CUDA 平滑过渡到 NPU 的关键。

- [达芬奇架构与 Ascend 910B3](02_ascend_architecture/01_davinci_architecture.md) — AI Core 的 Cube/Vector/Scalar 三单元分工、HBM 内存层次、HCCS 8 卡全互联拓扑、CUDA SIMT 对照
- [CANN 软件栈详解](02_ascend_architecture/02_cann_software_stack.md) — 驱动→Runtime→算子→图编译→AscendCL→HCCL 七层结构、CANN vs CUDA 工具对照表

---

## 4. 核心编程范式

Ascend 上开发模型主要依赖两条技术路线：**PyTorch NPU**（外部适配，设计目标是尽量复用 CUDA 代码与开发习惯） vs **MindSpore**（原生框架，跟 Ascend 硬件 / CANN 编译器深度耦合）。选型决策取决于三个因素：存量 CUDA 代码规模、生态的可移植需求、对极致性能 / 静态图优化的追求程度。

| 维度           | PyTorch NPU                             | MindSpore                                 |
| -------------- | --------------------------------------- | ----------------------------------------- |
| 迁移成本       | 低：`cuda()` → `npu()` 几乎全部覆盖     | 高：需重写为 `nn.Cell` + 函数式梯度 API   |
| 执行模式       | 动态图 + GE 子图编译                    | PyNative 动图 / Graph 静图可切换          |
| 训练循环       | `loss.backward()` 传统反向              | `ms.value_and_grad` 函数式梯度            |
| 生态兼容       | HuggingFace / vLLM / DeepSpeed 广泛支持 | 原生资源受限，MindSpore Lite 主推端侧部署 |
| Ascend 集成度  | 靠 `torch_npu` 适配，存在 TBE 依赖      | 原生支持，编译器可直接优化计算图          |
| 典型 ResNet-50 | FP32 545 / AMP 1254 img/s               | PyNative 165 / Graph 159 img/s            |

> 数据来源：测试服务器 Ascend 910B3 实测，详见对应子文档。

### 4.1 PyTorch NPU

PyTorch NPU 是大多数有 GPU 经验的开发者的首选入口。通过 `torch_npu` 适配层，CUDA 代码的迁移成本极低——绝大多数场景只需替换设备字符串。但也需要了解 CANN 图编译器（GE）的首次编译延迟、TBE 算子依赖等 NPU 特有的行为。

- [CUDA 到 NPU 的代码迁移](03_pytorch_npu/01_cuda_to_npu_migration.md) — 三步迁移法、10 组常用 API 对照表、`cuda()`→`npu()` 替换规则
- [ResNet-50 训练与 AMP 实战](03_pytorch_npu/02_resnet50_amp_training.md) — FP32 545 img/s vs AMP 1254 img/s (2.3×)、Gradient Scaling 行为、TBE 依赖排查、编译延迟分析

### 4.2 MindSpore

MindSpore 是华为自研框架，采用函数式梯度 API（`ms.value_and_grad`），支持 PyNative 动态图和 Graph 静态图两种执行模式。API 风格与 PyTorch 差异明显，但对 Ascend 硬件的集成度更高。

- [MindSpore 与 PyTorch API 对照](04_mindspore/01_mindspore_vs_pytorch_api.md) — 15 组常用 API 对照表、模型定义（`nn.Cell` vs `nn.Module`）、训练循环（函数式梯度 vs `loss.backward()`）、动静态图模式对比
- [MindSpore Ascend 训练实战](04_mindspore/02_mindspore_ascend_training.md) — PyNative 165 img/s vs Graph 159 img/s、版本兼容性（MindSpore 2.6 vs 2.9 与 CANN 8.0.1 的匹配关系）、MindSpore Lite 定位

---

## 5. 工具链

昇腾生态的工具链与 CUDA 工具有清晰的对应关系。`npu-smi` 对标 `nvidia-smi`（轻量巡检），`ascend-dmi` 对标 `deviceQuery` + `bandwidthTest`（硬件诊断与性能基准），`atc` 对标 TensorRT（模型编译）。

- [npu-smi 使用参考](05_tools/01_npu_smi_reference.md) — 60+ 种查询类型、默认输出字段解读、拓扑/HCCS/ECC/功耗/温度等常用查询、与 ascend-dmi 分工对照
- [ascend-dmi 使用参考](05_tools/02_ascend_dmi_reference.md) — 设备详情、带宽测试（HBM 1538 GB/s / HCCS 26.2 GB/s / PCIe 24.8 GB/s）、算力测试（FP16 313.7 TFLOPS）、故障诊断 12 项
- [ATC 模型转换](05_tools/03_atc_model_conversion.md) — PyTorch → ONNX → OM 完整流程、关键参数说明、AscendCL 推理加载

---

## 6. 进阶主题

走完环境、架构、框架、工具链后，还有两类场景需要跳出“调用现有算子”的路径：**算子级优化**（TBE / Ascend C 自定义算子，用于填补缺失算子或极致优化 hot kernel）与**存量工程迁移**（评估 GPU 代码是否该迁、迁多少、迁后收益是否趋近）。两者共同决定了 Ascend 项目能否从“跑起来”走到“跑得好”。

- [Ascend C 算子开发入门](06_advanced/01_ascend_c_intro.md) — 自定义算子的场景、`msopgen` + Ascend C Compiler 开发流程、简单 ReLU 示例
- [GPU 到 NPU 的迁移策略](06_advanced/02_gpu_to_npu_migration.md) — 迁移决策树、成本估算、不建议迁移的场景

---

## 7. RAG 实战

在 Ascend NPU 上搭建完整的 RAG pipeline：embedding 模型本地推理 + FAISS 向量检索 + LLM（支持外部 API 或本地 Qwen2.5-7B-Instruct，BF16 推理）。支持 `--local` 模式实现全链路本地化推理，通过 `--llm-model` 可切换 0.5B 等模型。NPU 编码 115 条文本耗时 0.8s (153 条/s)，对比 CPU 加速 ~422×。需要独立的 venv（`rag-env`）并精确锁定 transform‌ers / sentence-transformers 版本以兼容 CANN 8.0.1。

- [RAG Pipeline on NPU](07_rag_on_npu/01_rag_pipeline_on_npu.md) — 离线索引 + 在线查询完整流程、BGE 模型 NPU 推理适配、版本兼容性、Chrome trace 性能对比

---

## 8. NPU 性能分析

从"能跑"到"跑得快"——掌握算子级 profiling 方法，定位性能瓶颈。覆盖 `torch_npu.profiler` 的 Chrome trace 分析、npu-smi 实时监控、决策清单。实测 16384² 矩阵乘法 71.84 TFLOPS（FP32 利用率 ~90%）。

- [NPU 性能分析入门](08_npu_profiling/01_npu_profiling.md) — 计算 bound vs 访存 bound、synchronize 陷阱、warmup 必要性、矩阵乘法/2D 卷积/ResNet-50 profiling 数据

---

## 9. FlashAttention 实战

手写 FlashAttention forward pass，理解 online softmax 的数学原理和 tiling 分块策略。标准 attention 的 O(N²) 显存瓶颈是如何通过 `correction = exp(m_old - m_new)` 一行代码解决的。HBM 峰值从 130 MB 降至 1 MB（节省 95%），数值精度 max_diff < 1e-6。

- [FlashAttention 简化版](09_flash_attention/01_flash_attention.md) — online softmax 推导、tiling 伪代码、精度/显存/速度对比、Python 实现不加速的原因分析

---

## 10. Mini-GPT 实战

从零手写 GPT-2 风格 decoder-only Transformer，在单张 NPU 上完成训练和文本生成。覆盖 Self-Attention、Multi-Head、FFN、残差连接、LayerNorm、Position Embedding 六大核心机制的数学推导与代码实现。~11M 参数，2000 iters 训练 43 秒。

- [Mini-GPT 训练详解](10_mini_gpt/01_mini_gpt_training.md) — Transformer 核心机制剖析、模型架构设计、训练过程、文本生成策略、实测 loss 曲线与生成效果

---

## 11. LLM 推理 on NPU

在 NPU 上部署 Qwen2.5-7B-Instruct（BF16）进行本地推理，已与 RAG pipeline 集成实现全链路本地化。记录了 7B 模型 FP16 推理 NaN 问题的完整诊断——从现象到根因（FP16 溢出）到解决方案（BF16），最终以与 FP16 相同的 HBM 代价实现了 FP32 级别的数值稳定性。

- [LLM 推理 on NPU](11_llm_inference/01_llm_inference_on_npu.md) — 自回归生成原理、ChatML 格式、采样策略、0.5B/7B BF16 部署、与 RAG 对接
- [Qwen2.5-7B FP16 NaN 诊断报告](11_llm_inference/02_fp16_nan_debug.md) — 从乱码 URL → NaN logits → 层级追踪 → FP16 溢出根因 → BF16 解决（14.7 GB，已验证）

---

## 12. DDP 多卡分布式训练

在 Ascend NPU 上使用 HCCL 集合通信库实现多卡分布式训练（DDP），覆盖 HCCL vs NCCL 对照、DDP 工作原理、7B LoRA 多卡微调、初始化常见故障排查、14B 全参训练准备。8 张 910B3 通过 HCCS 全互联，每两张卡直连，通信延迟低。

- [DDP 多卡训练详解](12_ddp/01_ddp_training.md) — HCCL 集合通信、DDP 初始化与数据分发、7B LoRA 多卡微调、等效 batch 计算、HCCL 初始化故障排查

---

## 13. LoRA 微调

在 Ascend NPU 上对 Qwen2.5-7B-Instruct 做参数高效微调（LoRA），支持 CLM 和 SFT 两种数据格式。经过 5 种方案的对比实验，确认 SFT（指令微调）远优于 CLM（原始文本续写），380 条 QA 对已接近 LoRA r=8 的有效上限。配合 RAG 使用时，SFT 提供领域表达风格，RAG 提供具体事实知识，两者互补。

- [LoRA 微调详解](13_finetune/01_lora_finetune.md) — LoRA 原理、配置策略、5 种方案对比、380 QA 训练结果、RAG+SFT 协同验证

---

## 14. 量化推理 (INT8/INT4)

理解模型量化的数学原理：对称/非对称量化、per-tensor/per-channel 粒度、校准数据的作用，以及 FP16/BF16 vs INT8/INT4 的精度-效率权衡。7B 模型 INT4 量化后仅需 ~3.5 GB HBM。当前 CANN 8.0.1 + torch_npu 2.1.0 栈不支持 HF 模型的 INT8/INT4 推理（bitsandbytes/GPTQ/AWQ 均为 CUDA 专用），本章聚焦理论理解和 CPU 演示。

- [量化推理详解](14_quantization/01_quantization_theory.md) — 对称/非对称量化、per-tensor/per-channel、校准方法、INT8/INT4 精度分析
- 交互演示：[`quantization_viz.html`（可视化位宽影响）](./14_quantization/quantization_viz.html)、[`quantization_guide.html`（手算示例）](./14_quantization/quantization_guide.html)

---

## 15. 参考链接

- [昇腾社区官网](https://www.hiascend.com)
- [Ascend PyTorch 适配 (Gitee)](https://gitee.com/ascend/pytorch)
- [MindSpore 安装指南](https://www.mindspore.cn/install)
- [CANN 商业版文档](https://www.hiascend.com/document)
