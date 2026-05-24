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

| 方法         | 可训练参数     | HBM 占用   | 训练速度 | 适用场景                 |
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
    # 使用 rglob 递归扫描 .md 和 .txt 文件
    # 以 tokenizer 编码长度为准合并段落（而非字符数）
```

改进点：使用 tokenizer 实际编码长度判断 chunk 大小（而非 Python 字符串 `len()`），确保每个训练样本不超过模型的最大上下文。递归扫描（`rglob`）支持子目录。`UnicodeDecodeError` 异常保护。

脚本支持两种数据格式：

- **CLM（`--data-dir`）**：原始文本续写，适合教模型学习文档风格。问题：会覆盖 instruction-tuned 模型的指令遵循能力
- **SFT（`--sft`）**：指令微调，从 JSONL 读 (instruction, output) 对，构造 ChatML 格式训练。保留指令遵循能力，仅更新领域知识

经过系统对比实验，**SFT 格式远优于 CLM**——CLM 在原始文档上训练导致灾难性遗忘（Wikipedia 5.5MB 训练后模型完全丧失连贯性），而 SFT 保持了预训练的语言能力同时注入领域知识。

### 2.3 训练配置

| 参数       | 值            | 理由                                |
| ---------- | ------------- | ----------------------------------- |
| batch_size | 1             | 受限于 7B 模型的 HBM                |
| seed       | 42            | 固定随机种子，保证可复现            |
| grad_accum | 4             | 等效 batch=4，平滑梯度              |
| lr         | 2e-4          | LoRA 常用学习率                     |
| warmup     | 9 steps (10%) | 线性 warmup，避免初期震荡           |
| 精度       | BF16          | 与 7B 模型训练精度一致              |
| epoch      | 2             | 实验用 2（CLM）和 5（SFT）；代码默认 3 |

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
    # 最后不完整步按实际累积数归一化
    actual_accum = grad_accum if is_accum_step else ((step + 1) % grad_accum or grad_accum)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(actual_accum)

    torch.nn.utils.clip_grad_norm_(
        filter(lambda p: p.requires_grad, model.parameters()), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    global_step += 1
    epoch_loss += accumulated_loss
    accumulated_loss = 0.0
```

关键点：

- **BF16 autocast**：前向在 BF16（节省一半显存）。BF16 的 8 位指数与 FP32 相同，反向梯度无需上抛即可保持数值稳定
- **梯度累积 + 动态 actual_accum**：loss 不预除，在 optimizer.step 前按实际累积步数归一化梯度。最后不完整 batch 的 actual_accum 可能小于 grad_accum，动态计算保证梯度等价
- **梯度检查点**（`gradient_checkpointing_enable`）：不保存中间激活，backward 时重算，HBM 峰值从 21.2 GB 降至 16.4 GB
- **梯度裁剪**：`clip_grad_norm` 仅作用于可训练参数（`requires_grad`），防止小数据集下的梯度爆炸
- **固定随机种子**（`seed=42`）：保证训练可复现

---

## 3. 训练结果

环境：Ascend 910B3, CANN 8.0.1, NPU 7, BF16。

### 3.1 训练指标

| 指标        | CLM (162 样本)         | SFT 250 QA             | SFT 380 QA (最终)       |
| ----------- | ---------------------- | ---------------------- | ----------------------- |
| 数据格式    | 原始文档续写（CLM）    | ChatML 问答对（SFT）   | ChatML 问答对（SFT）    |
| 样本数      | 162                    | 250                    | 380                     |
| 可训练参数  | 5,046,272 (0.07%)      | 5,046,272 (0.07%)      | 5,046,272 (0.07%)       |
| Epochs      | 2                      | 5                      | 5                       |
| Total Steps | 82                     | 315                    | 455                     |
| 训练时间    | ~2 min                 | ~8 min                 | ~12 min                 |
| HBM 峰值    | 16.4 GB                | 16.4 GB                | 16.4 GB                 |
| HBM 稳态    | 15.0 GB                | 15.0 GB                | 15.0 GB                 |
| 最终 Loss   | 21.3                   | 12.3                   | 14.4                    |
| **效果**    | **词截断 + 重复**      | **域内知识迁移成功**   | **趋近容量上限**        |

### 3.2 Loss 曲线

```text
               250 QA (315 steps)        380 QA (455 steps)
Epoch 1 完成:  Avg Loss ≈ 48            Avg Loss 29.6
Epoch 2 完成:  Avg Loss ≈ 21            Avg Loss 17.9
Epoch 3 完成:  Avg Loss ≈ 16            Avg Loss 16.2
Epoch 4 完成:  Avg Loss ≈ 12.8          Avg Loss 15.1
Epoch 5 完成:  Avg Loss 12.3            Avg Loss 14.4
```

两个数据集均稳定收敛，未出现 CLM 方案中的灾难性遗忘。380 QA 的 epoch 1 loss（29.6）低于 250 QA（48），说明更多数据降低了初始 loss。loss 绝对值为 `accumulated_loss` 求和（未除以 `actual_accum`），量级高于单样本 loss。

### 3.3 实验对比

经过多种方案的迭代实验，关键发现：**数据格式（SFT vs CLM）比数据量更重要**。

| 方案 | 数据来源 | 样本数 | 效果 |
|------|---------|--------|------|
| CLM 域内文档 | 本地 13 篇 .md | 162 | 风格变化但词截断 + 重复 |
| CLM wiki + 域内 | Wikipedia 5MB | 4535 | **灾难性遗忘**（raw text 覆盖指令能力） |
| SFT 7B 生成 QA | 模型生成问答对 | 60 | 部分域内知识但重复输出 |
| SFT DSv4 Pro QA | 模型生成问答对 | 53 | 部分域内知识，样本少导致过拟合 |
| **SFT DSv4 Pro QA ★** | **模型生成问答对** | **380** | **趋近容量上限** |
| RAG + SFT ★★ | 检索 + 微调模型 | — | **互补：RAG 提供事实，SFT 提供风格** |

### 3.4 RAG + SFT 协同验证

SFT 微调后，部分问题（达芬奇架构、ascend-dmi、FP16 NaN）仍无法正确回答——训练数据虽包含相关知识但模型未充分学习。通过 RAG 检索注入相关文档后：

| 问题 | 仅 SFT | SFT + RAG |
|------|--------|----------|
| "什么是达芬奇架构？" | "NVIDIA的计算" ✗ | "Cube/Vector/Scalar 三单元分工明确" ✓ |
| "FP16 为什么 NaN？" | 退化 ✗ | "FP16 算后向导致溢出" ✓ |
| "如何使用 ascend-dmi？" | 退化 ✗ | 部分改善（正确概念 + 退化） |

**核心洞察**：SFT 教模型"怎么说"（领域表达风格），RAG 提供"说什么"（具体事实知识）。达芬奇架构——380 QA 训练都救不回的问题——RAG 检索后立即正确回答。两者 1+1>2。

### 3.5 250 QA 对微调效果

5 epoch, lr=2e-4, r=8。Loss 55→12.3，训练稳定收敛。

| 问题 | SFT 微调后回答 |
|------|--------------|
| "如何检查 NPU 是否可用？" | "通过 torch_npu.is_available 判断。False 说明驱动未安装或驱动有问题。" ✓ |
| "什么是 RAG？" | "是检索增强生成，用检索找到相关文档然后 LLM 生成答案。流程：查询→检索相关文档→生成回答。不依赖训练数据，适合需要最新知识的场景。" ✓✓ |
| "如何使用 ascend-dmi 测试带宽？" | "ascend-dmi 是昇腾 NPU 诊断工具，提供带宽和算力测试。" ✓（核心正确但细节退化） |
| "什么是达芬奇架构？" | 仍归因"Al云"但提到 ARM 架构 | 部分退化 |
| "KV Cache" | 通用缓存解释 | 概念泛化但非 LLM 特化 |

**评估**：模型首次展现出明确的昇腾领域知识迁移。RAG 定义、NPU API 检查、ascend-dmi 用途等核心概念被正确学习。部分问题仍有归因错误和重复——这是 250 样本 5 epoch 的边界表现。380 QA 实验（§3.1–§3.3）表明，在 r=8 下继续增加数据已接近容量上限，更有效的改进方向是增大 lora_r 或混合通用 SFT 数据。

> [!NOTE]
> **关键洞察**：SFT 保留了 ChatML 对话格式——模型在"对话框架内"学习领域知识，而非被原始文本覆盖指令能力。这是 SFT 优于 CLM 的根本原因。DSv4 Pro 生成的 QA 对比 7B 生成的 QA 质量更高（更准确、更精炼），同样数量下效果更好。

---

## 4. 代码结构

```text
11_llm_inference/
├── lora_finetune.py           # LoRA 微调脚本 (~400 行)
│   ├── get_lora_config()     — LoRA 超参数配置
│   ├── SFTDataset            — SFT 指令数据集（ChatML 格式）
│   ├── TextDataset           — CLM 文本数据集
│   ├── load_training_texts() — 文档加载与分块
│   └── train()               — 完整训练流程（CLM / SFT 双模式）
└── sft-data-250.jsonl         # 250 条手工 QA 数据集
```

---

## 5. 后续扩展

380 QA 对已接近 LoRA r=8 在当前数据量级下的有效上限。RAG + SFT 协同验证表明：RAG 检索可以弥补 SFT 的事实盲区（达芬奇架构从"完全错误"变为"正确回答"）。以下为后续方向：

- **RAG + SFT 协同（推荐）**：将 RAG pipeline 的默认 LLM 切换为 SFT 微调模型，实现风格（SFT）+ 知识（RAG）的最优组合
- **增大 lora_r（16 或 32）**：当前 r=8。扩大 QA 数据到 500+ 时，增大 rank 可提供更强的表达能力
- **混合通用 SFT 数据**：加入中文对话数据集保持模型的通用对话能力
- **评估驱动迭代**：通过 rag_eval.py 定位薄弱环节，有针对性地补充 QA 对

---

## 参考链接

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT (HuggingFace)](https://huggingface.co/docs/peft)
- [Qwen2.5 技术报告](https://arxiv.org/abs/2412.15115)
