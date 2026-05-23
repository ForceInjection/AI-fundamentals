# LLM 推理 on NPU：Qwen2.5-0.5B 本地部署

## 1. 背景

### 1.1 补齐最后一块拼图

回顾之前的章节：

- **RAG（§7）**：embedding 在 NPU 上，但 LLM 走外部 API——"模型在外面"
- **Mini-GPT（§10）**：手写 Transformer，全在本地，但模型只有 11M 参数——"有模型但太小"

本章的目标是补齐：**在 NPU 上运行真正的预训练模型推理**。选用 Qwen2.5-0.5B-Instruct：

- 0.5B 参数，~1GB，64GB HBM 完全够
- 中文优化，与 RAG 文档语料匹配
- Qwen2 架构，transformers 4.38.2 原生支持

### 1.2 为什么是 0.5B

0.5B 参数在 LLM 中属于"极小"级别（GPT-3 是 175B）。但它有一个关键优势：单卡推理无任何优化压力。约 1.2 GB HBM 占用意味着模型加载后还有 62+ GB 空闲，可以同时跑 embedding 模型做 RAG。

更大的模型（7B/14B）需要更精细的显存管理——参见 [Qwen2.5-7B FP16 NaN 诊断报告](02_fp16_nan_debug.md)，该文档记录了一次完整的数值稳定性诊断过程：7B 模型在 FP16 下因深层激活值溢出产生 NaN，最终确认是模型在 BF16 训练后迁移到 FP16 NPU 栈时的精度兼容性问题。

---

## 2. 自回归生成原理

理解 LLM 推理，首先要把它与前面见过的神经网络推理区分开。

### 2.1 一次 forward pass vs 多次 forward pass

前面学过的模型（ResNet-50、BGE embedding）都是**一次前向传播出结果**：输入 → 网络 → 输出。但 LLM 不同——它是**自回归**（autoregressive）的：

```text
输入: "NPU 是"
  → 第 1 次 forward: logits[最后一个位置] → softmax → 采样 → token "华为"
  → 第 2 次 forward: "NPU 是华为" → ... → token "昇腾"
  → 第 3 次 forward: "NPU 是华为昇腾" → ... → token "的"
  → ...直到生成 eos_token 或达到 max_new_tokens
```

每次 forward pass 都依赖之前生成的所有 token：$$y_t = \arg\max P(y_t | x_1, ..., x_n, y_1, ..., y_{t-1})$$

这意味着生成 256 个 token 需要跑 256 次完整的 forward pass。这就是为什么 LLM 推理比训练一个 batch 的感受慢很多——因为计算是串行的，无法并行。

### 2.2 为什么只要最后一个位置的 logits

一次 forward pass 的输出是一个矩阵 `[seq_len, vocab_size]`——每个位置都给出对下一个 token 的预测。但生成时，我们**只取最后一个位置**的 logits：

```python
next_token_logits = logits[-1, :]  # 只取 seq 最后一行的预测
```

前面位置的预测不需要，因为我们已经有正确答案（prompt 或之前生成的 token）。

### 2.3 KV Cache：避免重复计算

每次 forward pass 都把完整的序列重新编码——包括已经算过的前缀。这意味着同一个前缀（"NPU 是"）在生成第 1、2、3... 个 token 时被重复计算。

**KV Cache** 的解决方案：第一次 forward 把每层 Attention 的 Key 和 Value 缓存下来，后续 forward **只对新 token 做 attention**，旧 token 直接用缓存。这样每个新 token 只需要 O(1) 的 attention 计算，而非 O(N)。

```text
无 KV Cache:  第 N 步 forward 计算量 ∝ N (整条序列重新算)
有 KV Cache:  第 N 步 forward 计算量 ∝ 1 (只算新 token)
```

`transformers` 的 `model.generate()` 默认使用 KV Cache（配置中 `use_cache=True`），无需手动处理。这是 LLM 推理能从"不可用"变为"可用"的关键优化。

---

## 3. 对话格式与推理实现

### 3.1 为什么需要 ChatML 模板

预训练模型见过的数据是"裸文本"（网页、书籍、代码），它不知道什么是"对话"。Instruction-tuned 模型在训练时见过特定格式的数据，推理时也必须使用相同格式——否则模型会"迷失方向"。

Qwen2.5-Instruct 使用 **ChatML** 格式：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
什么是 NPU？<|im_end|>
<|im_start|>assistant
← 模型从这里开始生成
```

ChatML 通过特殊 token（`<|im_start|>`、`<|im_end|>`）标记每条消息的边界和角色。与直接拼字符串不同，这些标记是模型词表中的**真实 token**，有对应的 embedding，模型在训练中学会了它们的行为含义：

- `<|im_start|>` 后的内容标识消息角色（system/user/assistant）
- `<|im_end|>` 表示该条消息结束
- `<|im_start|>assistant\n` 是生成开始的信号——模型看到它就"知道"该说话了

如果不用 ChatML，直接用 `"你好，请回答问题"` 作为 prompt，对 instruction-tuned 模型来说相当于"裸文本续写"，输出会不可控。这在 0.5B 上表现为回答质量差，在 7B 上则可能完全崩溃。

`tokenizer.apply_chat_template()` 自动完成这个格式化：

```python
messages = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "什么是 NPU？"},
]
# add_generation_prompt=True 在末尾追加 <|im_start|>assistant\n
text = tokenizer.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
```

### 3.2 采样策略：temperature、top-p、top-k

模型输出的 logits 经过 softmax 后得到概率分布。如何从这个分布中"选择一个 token"？

**Greedy Decoding（贪婪解码）**：

```text
next_token = argmax(logits)  # 永远选概率最高的
```

问题：总是选最高概率的 token，输出往往无聊、重复、缺乏变化。一旦进入循环，永远出不来。

**Temperature**：

```text
probs = softmax(logits / temperature)
```

- `temperature → 0`：分布强烈尖锐→趋近贪婪解码→输出确定但保守
- `temperature = 0.7`：适中平滑→保留随机性但不失控
- `temperature → ∞`：分布趋于均匀→纯随机→完全不可用

temperature 不改变哪个 token 得分最高，只改变概率分布的"集中程度"。

**Top-p (Nucleus Sampling)**：

```text
1. 按概率从高到低排序所有 token
2. 从最高开始累加概率，直到累计 > p（如 0.9）
3. 只在这个"核心集合"中采样，其余 token 概率置 0
```

top-p 动态决定了"候选集"的大小——分布集中时候选少，分布分散时候选多。

**Top-k**：

```text
只保留概率最高的 K 个 token，其余概率置 0，重新归一化
```

top-k 简单但粗暴：K 太小（如 5）会让分布尖锐时丢失合理选项，K 太大（如 100）会让分布平坦时混入噪声。top-p 更灵活，实践中常用 `top-p + temperature` 组合。

### 3.3 代码实现

```python
# 环境：复用 rag-env（transformers 4.38.2 + torch_npu 2.1.0），无需新建 venv

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,  # FP16 加载，~1GB HBM
).to("npu:0").eval()

# 构建对话
messages = [{"role": "user", "content": "什么是 NPU？"}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(text, return_tensors="pt").to("npu:0")
input_len = inputs.input_ids.shape[1]

# 自回归生成
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,              # 控制随机性
        do_sample=True,               # temperature>0 时启用采样
        top_p=0.9,                    # 核心采样
        pad_token_id=tokenizer.eos_token_id,
    )
answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
```

---

## 4. 性能数据

测试环境：Ascend 910B3, CANN 8.0.1, NPU 7, FP16。

| 指标               | 数值                           |
| ------------------ | ------------------------------ |
| 参数量             | 494M                           |
| 模型加载时间       | 9s（缓存后）                   |
| HBM 占用           | 1.1 GB（稳态）, 1.2 GB（峰值） |
| 生成速度           | 18.7-20.7 tok/s                |
| 128 token 生成耗时 | ~6-7s                          |

**推理示例：**

```text
Prompt: 什么是深度学习？
回答: 深度学习是一种机器学习的分支，它利用神经网络结构来自动发现
      数据中的模式和结构。与传统的基于特征的学习方式不同，深度学习能...

Prompt: 介绍一下华为昇腾 NPU 的特点。
回答: 华为昇腾 NPU 是一种面向深度学习应用的专用芯片，具有以下主要特点：
      1. 高性能：昇腾 NPU 采用了先进的神经网...
```

0.5B 模型的知识面有限——对于具体技术问题只能给出泛泛的描述。这是模型能力的天花板，而非 NPU 推理的问题。

---

## 5. 与 RAG 的集成

本章的 LLM 推理已与 RAG pipeline 完成对接：rag_pipeline.py 中的 `LocalLLMClient` 类封装了本地模型加载与推理，通过 `--local` 参数启用。

```python
# rag_pipeline.py 中的对接方式
class LocalLLMClient:
    def chat(self, messages, temperature=0.3, max_tokens=512):
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("npu:0")
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, ...)
        return tokenizer.decode(...)
```

全链路本地化的 RAG：

```text
文档 → NPU embedding (BGE) → FAISS 检索 → NPU LLM (Qwen 0.5B) → 回答
```

全程无需网络、无需 API Key，延迟完全可控。

---

## 6. 代码结构

```text
11_llm_inference/
├── README.md
├── 01_llm_inference_on_npu.md    # 本文
├── 02_fp16_nan_debug.md          # 7B FP16 NaN 诊断报告
└── llm_inference.py              # LLM 推理脚本（~220 行）
    ├── load_model()              — 加载模型 + tokenizer（含 NPU 可用性检查）
    ├── run_inference()           — 单次推理
    ├── interactive_chat()        — 交互对话
    ├── benchmark()               — 性能测试
    └── main()                    — CLI（infer / chat / benchmark）
```

---

## 7. 后续扩展

- **量化推理**：INT8/INT4 量化，将 7B 模型的 HBM 占用从 14GB 降至 7GB/3.5GB，同时可能缓解 FP16 溢出问题（量化过程引入的缩放因子有助于数值稳定）
- **大模型推理**：当前 CANN 8.0.1 + torch_npu 2.1.0 栈上 7B 模型需要 FP32 加载（~29.4 GB HBM），升级 CANN 后可用 FP16/BF16
- **流式输出**：使用 `TextIteratorStreamer` 逐 token 返回，实现打字机效果
- **batching 多路推理**：同时处理多个请求，提升 NPU 利用率

## 参考链接

- [Qwen2.5 模型介绍](https://qwen.readthedocs.io/en/latest/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Qwen2.5-0.5B-Instruct on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [How to generate text: using different decoding methods (HuggingFace)](https://huggingface.co/blog/how-to-generate)
- [Qwen2.5-7B FP16 NaN 诊断报告](02_fp16_nan_debug.md)
