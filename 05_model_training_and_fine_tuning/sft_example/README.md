# SFT 微调实战与指南

通用大模型直接扔进业务里往往水土不服：它不懂专业术语，不会按你要的格式作答，也无法稳定调用领域工具。**监督微调（Supervised Fine-Tuning, SFT）** 就是把这道鸿沟填平的那一步。本项目用一个可跑通的 Qwen2 微调样例配合一份系统化的垂域 SFT 指南，覆盖从数据准备到推理验证的完整链路，既能作为上手模板，也能作为方法论参考。

---

## 1. 内容一览

- **`train_qwen2.ipynb`** — Qwen2-1.5B-Instruct 指令微调 Notebook。使用 ModelScope 拉取数据、Transformers + PEFT(LoRA) 训练、SwanLab 做训练过程可视化，单机单卡 10 GB 显存即可复现。
- **`一文入门垂域模型SFT微调.md`** — 垂域 SFT 的理论与工程手册。以金融领域「企业年报分析助手」为贯穿案例，把数据构建、基座选型、LoRA/QLoRA 配置、评估与部署串成一条可落地的流水线。

---

## 2. 快速开始

### 2.1 环境准备

需要 Python 3.8+ 与一块支持 CUDA（或 macOS 下的 MPS）的计算设备，显存约 10 GB。

```bash
# torch / swanlab / modelscope / transformers / datasets / peft / pandas / accelerate
pip install torch swanlab modelscope transformers datasets peft pandas accelerate
```

### 2.2 数据集准备

实战采用 `zh_cls_fudan-news` 中文文本分类数据集：

1. 从 [ModelScope 数据集页面](https://modelscope.cn/datasets/huangjintao/zh_cls_fudan-news/files) 下载 `train.jsonl` 与 `test.jsonl`。
2. 将文件放到 notebook 同级或上级目录（以 notebook 内路径配置为准）。

### 2.3 运行微调

用 Jupyter Lab / VS Code 打开 `train_qwen2.ipynb`，按序执行单元格即可。主线流程：

1. 环境初始化与数据预处理（格式化为 instruction / input / output 三元组）。
2. 加载 Qwen2-1.5B-Instruct 及其 tokenizer。
3. 配置 LoRA 适配器并启动训练，SwanLab 监控 loss 与学习率曲线。
4. 加载微调权重做推理验证，对比基座输出。

---

## 3. SFT 理论速览

下面两张表把 `一文入门垂域模型SFT微调.md` 里的核心骨架抽出来，便于在动手前建立全局直觉。

### 3.1 三阶段训练对比

| 阶段                      | 目标                   | 数据规模                | 典型耗时     | 典型硬件    |
| :------------------------ | :--------------------- | :---------------------- | :----------- | :---------- |
| **预训练 (Pre-training)** | 获取通用语言知识       | TB 级无标注文本         | 数周至数月   | 多机千卡级  |
| **监督微调 (SFT)**        | 适配具体任务与指令理解 | 10⁴–10⁵ 条标注样本      | 数小时至数天 | 单机 4–8 卡 |
| **强化学习 (RLHF)**       | 对齐人类偏好           | 千级人类反馈排序数据    | 数天至数周   | 单机多卡    |

### 3.2 工程链路

1. **数据准备** — 文本抽取、清洗、去重，构建 Prompt-Response 对；垂域场景通常伴随术语表与格式规约。
2. **基座选型** — 按能力上限、license、中文质量、推理成本权衡（Qwen、Baichuan、LLaMA 等）。
3. **训练配置** — 全参数微调 vs. PEFT（LoRA / QLoRA），在效果与显存占用之间平衡。
4. **验证评估** — 自动指标 + 抽样人评，关注领域准确率与幻觉率。
5. **部署上线** — 灰度放量、指标看板、线上反馈回流到下一轮数据。

---

## 4. 参考资源

- [Qwen2 大模型指令微调入门实战](https://mp.weixin.qq.com/s/Atf61jocM3FBoGjZ_DZ1UA) — 本 Notebook 对应的图文教程。
- [SwanLab 官方文档](https://swanlab.cn) — 训练过程可视化工具使用手册。
- [ModelScope 社区](https://modelscope.cn) — 模型与数据集托管门户。
