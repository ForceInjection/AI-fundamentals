# SFT 微调实战与指南

本项目旨在提供大模型监督微调（Supervised Fine-Tuning, SFT）的实战示例与理论指南。包含基于 Qwen2 模型的微调代码以及垂域模型微调的详细文档，帮助开发者快速掌握从理论到实践的完整流程。

## 1. 内容列表

本项目包含以下核心文件：

- **`train_qwen2.ipynb`**: Qwen2 大模型指令微调的 Jupyter Notebook 实战教程。演示了如何使用 SwanLab、ModelScope 和 Transformers 库对 Qwen2-1.5B-Instruct 模型进行微调。
- **`一文入门垂域模型SFT微调.md`**: 深入解析垂域模型 SFT 微调的理论、流程与最佳实践。以金融领域“企业年报分析助手”为例，详细阐述了从数据准备到模型部署的全过程。

## 2. 快速开始

### 2.1 环境准备

本教程实战部分依赖 `swanlab`、`modelscope` 等开源库，建议使用 Python 3.8+ 环境，并准备支持 CUDA 或 MPS（macOS）的计算设备。显存要求约为 10GB。

```bash
# 安装必要的 Python 依赖库
# 包括 torch, swanlab, modelscope, transformers, datasets, peft, pandas, accelerate
pip install torch swanlab modelscope transformers datasets peft pandas accelerate
```

### 2.2 数据集准备

实战教程使用 `zh_cls_fudan-news` 文本分类数据集。

1. 访问 [ModelScope 数据集页面](https://modelscope.cn/datasets/huangjintao/zh_cls_fudan-news/files) 下载 `train.jsonl` 和 `test.jsonl`。
2. 将下载的文件放置在与 notebook 同级的目录（或上一级目录，具体请参考代码中的路径配置）。

### 2.3 运行微调

使用 Jupyter Lab 或 VS Code 打开 `train_qwen2.ipynb`，按顺序执行代码单元格即可开始微调流程。主要步骤包括：

1. 环境与数据处理
2. 模型与 Tokenizer 加载
3. LoRA 配置与模型训练
4. 推理验证

## 3. SFT 理论概览

根据项目中的文档 `一文入门垂域模型SFT微调.md`，SFT 是将通用大模型适配到特定领域的关键步骤。

### 3.1 训练阶段对比

大模型训练通常分为以下三个阶段：

| 阶段                      | 目标                   | 数据规模             | 典型耗时     | 典型硬件    |
| :------------------------ | :--------------------- | :------------------- | :----------- | :---------- |
| **预训练 (Pre-training)** | 获取通用语言知识       | TB 级无标注文本      | 几周至数月   | 多机千卡级  |
| **监督微调 (SFT)**        | 适配具体任务和指令理解 | 10⁴–10⁵ 条标注样本   | 数小时至数天 | 单机 4–8 卡 |
| **强化学习 (RLHF)**       | 优化人类偏好对齐       | 千级人类反馈排序数据 | 数天至数周   | 单机多卡    |

### 3.2 微调流程图解

SFT 的核心工作流程如下：

1. **数据准备**: 涉及文本抽取、清洗、去重以及 Prompt-Response 对的构建。
2. **模型选择**: 根据场景需求与资源预算选择基座模型（如 Qwen, Baichuan, LLaMA 等）。
3. **配置训练**: 选择全参数微调或 PEFT（如 LoRA, QLoRA）技术进行训练。
4. **验证评估**: 使用自动化评测工具评估模型的准确率、召回率等指标。
5. **部署上线**: 灰度发布并进行持续的在线监控。

## 4. 参考资源

- [Qwen2 大模型指令微调入门实战](https://mp.weixin.qq.com/s/Atf61jocM3FBoGjZ_DZ1UA)
- [SwanLab 官方文档](https://swanlab.cn)
- [ModelScope 社区](https://modelscope.cn)
