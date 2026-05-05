# LLM Token 技术指南

Token 是大模型看世界的最小单位——既决定了模型能一次处理多长的上下文，也直接挂钩 API 调用的钱。搞清楚「一段文本会被切成多少 token」「不同 tokenizer 的策略差异在哪」「怎么不超限还省钱」，是所有 LLM 工程的共同基本功。本目录把这些内容压缩为一份介绍文档外加一个可运行的 Token 估算小工具。

## 1. 内容概览

### 1.1 核心文档

- **[llm_token_intro.md](llm_token_intro.md)** — 介绍 Token 的定义、BPE / WordPiece 等主流切分算法、与成本/上下文窗口的关系，以及常见的长度估算陷阱。

### 1.2 实用工具

- **[token_estimation.py](token_estimation.py)** — 命令行 Token 数量估算脚本，支持多种主流 tokenizer。
- **[Dockerfile](Dockerfile)** — 容器化构建文件，将估算脚本打包为可部署镜像，便于在 CI / 内部服务中调用。

### 1.3 Token 化过程

- **文本分割** — 把字符 / 词组切成 Token 序列，不同语言策略差异显著。
- **编码算法** — BPE（GPT 系）、WordPiece（BERT 系）、SentencePiece 的差异与适用场景。
- **特殊 Token** — `[CLS]` / `[SEP]` / `<|endoftext|>` / 工具调用标签等控制符号的作用与数量占用。
- **多语言支持** — 中文、代码等非英文语料下的切分稳定性与 fallback 策略。

### 1.4 Token 计算

- **长度估算** — 预先估算请求的 token 数量，避免超出模型上下文窗口。
- **成本控制** — 按 prompt / completion 分别计价的 Token-based 成本模型。
- **限制处理** — 超长输入的截断、滑窗、摘要三类缓解方案。
- **优化技巧** — prompt 压缩、重复内容去除、system prompt 共享等工程手段。

## 2. 相关资源

- [LLM 基础理论](../README.md)
- [模型微调技术](../../../05_model_training_and_fine_tuning/README.md)
- [RAG 技术实践](../../../07_rag_and_tools/README.md)
