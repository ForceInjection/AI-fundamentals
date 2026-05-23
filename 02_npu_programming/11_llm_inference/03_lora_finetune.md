# LoRA 微调 Qwen2.5-7B on Ascend NPU

## 1. 背景

### 1.1 为什么需要微调

RAG 方案让模型基于外部文档回答，但有两个局限：

- 模型基础能力是通用的，不了解 Ascend 领域的术语和表达习惯
- 检索到的文档需要模型有足够的理解能力才能正确引用

微调的目标是让模型**内化领域知识**——不只是"看到"文档，而是学会用 Ascend 领域的语言风格来表达。LoRA（Low-Rank Adaptation）是目前最广泛使用的参数高效微调方法。

### 1.2 什么是 LoRA

LoRA 的核心思想是：**不修改原始模型权重，在旁路添加低秩矩阵进行训练**。

```text
原始前向:  h = W₀ @ x
LoRA 前向: h = W₀ @ x + B @ A @ x

其中: W₀ ∈ R^{d×d} (冻结, ~14GB)
      A   ∈ R^{r×d} (可训练, rank r)
      B   ∈ R^{d×r} (可训练, rank r)

r=8 时, 参数量: 2 × 3584 × 8 ≈ 57K (每个 target module)
```

为什么 LoRA 有效？预训练模型学到的权重矩阵已经包含了丰富的知识。微调时，参数的**变化量**（ΔW）实际上是低秩的——只需要很少的参数就可以调整模型的行为。LoRA 将 ΔW 分解为 BA^T，用极少的参数（0.1% 量级）实现对模型行为的有效调整。

### 1.3 为什么选择 LoRA 而非全参数微调

| 方法         | 可调度参数     | HBM 占用   | 训练速度 | 适用场景                 |
| ------------ | -------------- | ---------- | -------- | ------------------------ |
| 全参数微调   | 7.6B (100%)    | ~57 GB     | 慢       | 大数据集、追求极致效果   |
| **LoRA r=8** | **5M (0.07%)** | **~21 GB** | **快**   | **小数据集、域内适配**   |
| LoRA r=64    | 40M (0.5%)     | ~22 GB     | 较快     | 中等数据集、更强表达能力 |

对于本实验的 120K 字符数据集，LoRA r=8 是最合适的——数据集本身不足以支撑全参数微调，LoRA 的低参数量恰好避免了严重过拟合。

---

## 2. 实现

### 2.1 LoRA 配置

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                          # LoRA rank
    lora_alpha=16,                # 缩放因子（等效 lr_scale = alpha/r = 2）
    lora_dropout=0.05,            # 轻微正则化
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 只微调 Attention
)

model = get_peft_model(base_model, config)
```

**设计决策**：

- **target_modules 只选 Attention 投影**：Qwen2.5 的 Attention 层负责"信息提取"——决定从哪里看、看什么。微调这部分最能影响模型对领域知识的关注方式。FFN 层（SwiGLU）负责"知识存储"，冻结它可以保留模型原有的常识推理能力
- **r=8 而非 r=4 或 r=16**：r=4 表达能力不足（120K 数据虽小但需要足够的调整空间）；r=16 参数量翻倍（10M），收益递减
- **alpha=16 → lr_scale=2**：LoRA 输出的实际缩放因子是 alpha/r。设为 2 意味着 LoRA 的输出被放大 2 倍，增强了对原始模型行为的影响力，适合数据集与预训练分布差异大的场景

### 2.2 数据准备

训练数据是 13 篇 Ascend 学习文档（共 ~120K 字符），按 token 长度分块后构建因果语言模型（CLM）样本：

```python
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        # 按 max_length 滑动窗口切分，相邻块重叠 1/4
        # 对每个样本: input_ids = tokens[:-1], labels = tokens[1:]
        # labels 中 padding 位置设为 -100（CrossEntropyLoss 忽略值）

def load_training_texts(data_dir, tokenizer):
    # 使用 rglob 递归扫描 .md 文件
    # 以 tokenizer 编码长度为准合并段落（而非字符数）
```

改进点：使用 tokenizer 实际编码长度判断 chunk 大小（而非 Python 字符串 `len()`），确保每个训练样本不超过模型的最大上下文。递归扫描（`rglob`）支持子目录。`UnicodeDecodeError` 异常保护。

为什么用 CLM 而非指令微调？我们的数据是**文档**而非**问答对**。CLM 让模型学习文档的风格和知识表达方式。

### 2.3 训练配置

| 参数       | 值            | 理由                                |
| ---------- | ------------- | ----------------------------------- |
| batch_size | 1             | 受限于 7B 模型的 HBM                |
| seed       | 42            | 固定随机种子，保证可复现            |
| grad_accum | 4             | 等效 batch=4，平滑梯度              |
| lr         | 2e-4          | LoRA 常用学习率                     |
| warmup     | 9 steps (10%) | 线性 warmup，避免初期震荡           |
| 精度       | BF16          | 与 7B 模型训练精度一致              |
| epoch      | 2             | 小数据集 2 轮足够，更多会严重过拟合 |

### 2.4 训练循环

```python
with torch.npu.amp.autocast(dtype=torch.bfloat16):
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    loss = outputs.loss

loss.backward()
accumulated_loss += loss.item()

is_last_step = (step + 1) == len(dataloader)
is_accum_step = (step + 1) % grad_accum == 0

if is_accum_step or is_last_step:
    # 根据实际累积步数进行梯度平均
    actual_accum = grad_accum if is_accum_step else ((step + 1) % grad_accum or grad_accum)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(actual_accum)

    torch.nn.utils.clip_grad_norm_(
        filter(lambda p: p.requires_grad, model.parameters()), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

关键点：

- **BF16 autocast**：前向在 BF16（节省一半显存），反向梯度在 FP32（保证精度）
- **梯度累积 + 动态 actual_accum**：最后不完整的累积步按实际步数归一化梯度
- **梯度检查点**（`gradient_checkpointing_enable`）：不保存中间激活，backward 时重算，HBM 峰值从 21.2 GB 降至 16.4 GB
- **梯度裁剪**：`clip_grad_norm` 仅作用于可训练参数（`requires_grad`），防止小数据集下的梯度爆炸
- **固定随机种子**（`seed=42`）：保证训练可复现

---

## 3. 训练结果

环境：Ascend 910B3, CANN 8.0.1, NPU 7, BF16。

### 3.1 训练指标

| 指标        | 值                                               |
| ----------- | ------------------------------------------------ |
| 训练数据    | 13 篇文档, ~120K 字符, 60 样本（token 长度分块） |
| 可训练参数  | 5,046,272 (0.07%)                                |
| Epochs      | 2                                                |
| Total Steps | 100                                              |
| 训练时间    | ~2 min                                           |
| HBM 峰值    | 16.4 GB（梯度检查点优化，优化前 21.2 GB）        |
| HBM 稳态    | 15.0 GB                                          |

### 3.2 Loss 曲线

```text
Epoch 1 | Step  5: Loss 12.08 | lr 1.00e-04
Epoch 1 | Step 10: Loss 11.87 | lr 2.00e-04  ← warmup 到峰值
Epoch 1 | Step 25: Loss 10.51 | lr 1.67e-04  ← 持续下降
Epoch 1 | Step 50: Loss  8.39 | lr 1.11e-04
Epoch 1 完成: Avg Loss 8.39

Epoch 2 | Step 55: Loss 5.72 | lr 1.00e-04  ← Epoch 2 开始
Epoch 2 | Step 75: Loss 5.51 | lr 5.56e-05
Epoch 2 | Step 100: Loss 5.41 | lr 0.00e+00  ← LR 衰减到 0
Epoch 2 完成: Avg Loss 5.41
```

Loss 从 12.08 降到 5.41（持续下降、未过拟合）。注意 loss 绝对数值高于早期版本（3.30），这是因为 loss 不再被 `grad_accum` 预除——训练本身等价，只是记录值的量级不同。

### 3.3 推理对比

| 问题                     | 微调后 (LoRA r=8)                                                                    | 微调前 (Base)                                      |
| ------------------------ | ------------------------------------------------------------------------------------ | -------------------------------------------------- |
| "请介绍一下华为昇腾 NPU" | "华为昇 NPU（PU）是华为自研的 AI处理器，专用于AI计算，特点是：1 专用：PU为AI设计..." | "华为昇腾 NPU 是一种面向深度学习应用的专用芯片..." |
| "什么是达芬奇架构？"     | "达架构是阿里云自研的AI芯片，用于推理..."                                            | "达芬奇架构是华为设计的AI处理器架构..."            |
| "如何安装 torch_npu？"   | "安装 torch-npu需要以下步骤：1 环境准备...2 torch-n安装..."                          | "torch_npu 可以通过 pip 安装..."                   |

微调后模型输出风格发生了变化——更结构化（带编号列表）、更贴近训练文档的表达方式。但同时也出现了**事实退化**（"阿里云自研"、词截断"昇 NPU"），这是小数据集训练的典型表现。要改善，需要：

- **更大的数据集**：至少 10MB+ 的技术文档
- **混合通用数据**：用中文维基百科等保持模型的常识能力
- **更小的 lr**：1e-4（而非 2e-4）减少对原始权重的冲击

---

## 4. 代码结构

```text
11_llm_inference/
└── lora_finetune.py    # LoRA 微调脚本 (~280 行)
    ├── get_lora_config()      — LoRA 超参数配置
    ├── TextDataset            — CLM 数据准备
    ├── load_training_texts()  — 文档加载与分块
    └── train()                — 完整训练流程
```

---

## 5. 后续扩展

- **指令微调**：将文档转换为问答对（用 7B 生成问题），做监督微调（SFT）而非 CLM。这需要额外的数据构造步骤，但效果更直接——模型学会"回答问题"而非"续写文档"
- **更大的数据集**：中文维基百科 + 技术博客 + CSDN 等技术社区文章，目标 10-50MB
- **DPO/RLHF**：在 SFT 基础上用偏好对齐进一步提升回答质量
- **多轮对话微调**：构造 RAG 场景下的多轮对话数据，让模型学会在引用来源和回答问题之间平衡

---

## 参考链接

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT (HuggingFace)](https://huggingface.co/docs/peft)
- [Qwen2.5 技术报告](https://arxiv.org/abs/2412.15115)
