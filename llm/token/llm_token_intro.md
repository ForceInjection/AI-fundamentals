# 解密 LLM 中的 Tokens

在大语言模型（LLM）中，“**tokens**”（令牌）是理解和生成语言的最小单位。无论是输入文本的处理，还是模型输出的生成，`tokens` 都在其中扮演核心角色。本文将详细解析 `token` 的概念、如何估算其数量，并提供具体的 `Python` 示例，帮助开发者更好地理解和调用大语言模型。

## 什么是 Token？

在 LLM 中，`token` 是模型处理的基本单元。Tokens 通常不是单个字母或字符，而是更具语义的单位（如单词、子词或字符），具体取决于分词方法。例如：

- **单词**：英文中的 "`dog`" 或 "`computer`"。
- **子词**：通过分词方法拆分的单位，如 "`un-`" 和 "`-able`"。
- **字符**：如 "`h`", "`e`", "`l`", "`l`", "`o`"。
- **标点符号**：句号“。”、逗号“，”等也会被视作独立 `tokens`。

这些 `tokens` 被转换为嵌入向量，供模型理解和生成文本。

## Tokens 如何影响模型的输入和输出

大多数 `LLM`（如 `GPT` 系列）通过 `token`s 对文本编码。每个 `token` 对应一个嵌入向量，模型据此预测后续内容。`Token` 数量直接影响模型的性能、计算负载及调用成本。

例如，`GPT-3` 的最大上下文长度为 **2,048 tokens**，若输入超出限制则需截断或拆分。

## 如何估算 Tokens 数量

调用 `LLM API` 前估算 `token` 数量至关重要，以下是常用方法：

---

### 使用 `tiktoken` 估算（适用于 OpenAI GPT 模型）

`tiktoken` 是 `OpenAI` 官方库，专为 `GPT` 系列设计。

**安装**：

```bash
pip install tiktoken
```

**估算示例**：

```python
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")  # 指定 cl100k_base 的分词器
text = "What is the capital of Chinese?"
tokens = encoder.encode(text)

print(f"Token 数量: {len(tokens)}")  # 输出: 7
```

---

### 使用 `transformers` 估算中文 Tokens（更灵活）

`transformers` 提供多语言支持，适合中文文本。

**安装**：

```bash
pip install transformers
```

**估算示例**：

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")  # 加载中文分词器
text = "《哪吒之魔童闹海》于2025年1月29日上映，讲述了哪吒在成长过程中，面对重重困难，不断挑战自我，最终逆天改命的故事。影片融合了精彩的特效和深刻的情感，展现了哪吒的勇敢与坚韧。"
tokens = tokenizer.encode(text, add_special_tokens=False)  # 去掉特殊字符

print(f"Token 数量: {len(tokens)}")  # 输出: 84
```

---

### 使用 `tiktoken` 处理中文的注意事项

尽管 `tiktoken` 主要面向英文，其基于 BPE（字节对编码）的算法也可处理中文，但可能产生更多子词 `tokens`：

```python
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")
text = "《哪吒之魔童闹海》于2025年1月29日上映，讲述了哪吒在成长过程中，面对重重困难，不断挑战自我，最终逆天改命的故事。影片融合了精彩的特效和深刻的情感，展现了哪吒的勇敢与坚韧。"
tokens = encoder.encode(text)

print(f"Token 数量: {len(tokens)}")  # 输出: 115
```

**结果差异说明**  
`tiktoken` 的 BPE 算法会将中文拆分为更细粒度的子词（如单字或常见组合），而 `transformers` 的 `BertTokenizer` 基于词表匹配，可能保留更多完整词语，因此两者计数不同。

---

## 完整代码

### **Dockerfile**

```Dockerfile
# 使用官方 Python 3.9 镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制当前目录下的所有文件到容器的工作目录
COPY token_estimation.py .

RUN mkdir -p /app/models

# 使用阿里云镜像源安装依赖（确保 URL 是有效的）
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip && \
    pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ tiktoken transformers hf_transfer

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV HF_HOME="/app/models"
#ENV HF_ENDPOINT="https://mirroring.huggingface.co"
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# 运行主程序
CMD ["python", "token_estimation.py"]
```

### **token_estimation.py**

```python
import tiktoken
from transformers import BertTokenizer
import argparse

def estimate_tokens_tiktoken(text, model="cl100k_base"):  # 修改模型名称
    """
    使用 tiktoken 估算 Tokens 数量
    """
    try:
        encoder = tiktoken.get_encoding(model)
        tokens = encoder.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error in tiktoken estimation: {e}")
        print(f"tiktoken version: {tiktoken.__version__}")
        return None

def estimate_tokens_transformers(text):
    """
    使用 transformers 估算中文 Tokens 数量
    """
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        tokens = tokenizer.encode(text, add_special_tokens=False)  # 去掉特殊字符
        return len(tokens)
    except Exception as e:
        print(f"Error in transformers estimation: {e}")
        return None

def main():
    english_text = "What is the capital of China?"
    chinese_text = "《哪吒之魔童闹海》于2025年1月29日上映，讲述了哪吒在成长过程中，面对重重困难，不断挑战自我，最终逆天改命的故事。影片融合了精彩的特效和深刻的情感，展现了哪吒的勇敢与坚韧。"

    print("\n--- 英文 Tokens 估算 ---")
    print(f"输入文本: {english_text}")
    tiktoken_eng = estimate_tokens_tiktoken(english_text)
    print(f"Tokens 数量（tiktoken）: {tiktoken_eng}")

    print("\n--- 中文 Tokens 估算 ---")
    print(f"输入文本: {chinese_text}")
    transformers_chinese = estimate_tokens_transformers(chinese_text)
    print(f"Tokens 数量（transformers）: {transformers_chinese}")
    tiktoken_chinese = estimate_tokens_tiktoken(chinese_text)
    print(f"Tokens 数量（tiktoken）: {tiktoken_chinese}")

if __name__ == "__main__":
    main()
```

### 构建及执行

```bash
docker build --network=host -t token_estimation .

docker run --rm --network=host token_estimation
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.

--- 英文 Tokens 估算 ---
输入文本: What is the capital of China?
Tokens 数量（tiktoken）: 7

--- 中文 Tokens 估算 ---
输入文本: 《哪吒之魔童闹海》于2025年1月29日上映，讲述了哪吒在成长过程中，面对重重困难，不断挑战自我，最终逆天改命的故事。影片融合了精彩的特效和深刻的情感，展现了哪吒的勇敢与坚韧。
Tokens 数量（transformers）: 84
Tokens 数量（tiktoken）: 115
```

## 总结

- `Token` 是 `LLM` 处理文本的基本单位，直接影响计算资源和成本。
- **英文文本**：推荐使用 `tiktoken`，因其与 `GPT` 系列的分词策略一致。
- **中文文本**：`transformers` 的分词器更贴近词语划分，而 `tiktoken` 可能生成更多子词 `tokens`。
- 理解不同工具的分词逻辑，可帮助优化文本输入，避免超出模型限制。

---
