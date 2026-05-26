# RAG 检索增强生成 on Ascend NPU

## 1. 背景

### 1.1 什么是 RAG

RAG (Retrieval-Augmented Generation) 是目前 LLM 应用最广泛的架构模式之一。它解决了一个核心问题：**大模型的知识是静态的、有截止日期的，而且容易"幻觉"**——对于它没见过的内容，模型可能编造看似合理但完全错误的答案。

RAG 的做法是：在 LLM 回答之前，先从外部知识库中检索相关文档片段，将它们作为"参考资料"一并送入 LLM。这样 LLM 就不再依赖自己的参数化记忆，而是基于检索到的实际文档内容作答。这带来了三个好处：

1. **知识可更新**：改文档即可，无需重新训练模型
2. **可溯源**：每个回答都能引用来源，方便验证
3. **减少幻觉**：有资料作为"约束"，模型不容易胡编

### 1.2 为什么在 NPU 上做

RAG pipeline 中最适合 NPU 加速的环节是 **embedding 编码**。将文档和查询文本转换为语义向量需要运行 BERT 类模型，天然适合 NPU 的矩阵计算能力。对于大规模文档库（数万到数十万篇），编码速度直接决定了索引构建和更新的时效性。

本实验将 embedding 推理部署在 NPU 上。LLM 生成部分支持两种模式：

- **外部 API**（默认）：通过 OpenAI 兼容接口调用云端模型，适合需要大模型能力的场景
- **本地推理**（`--local`）：在 NPU 上直接运行 Qwen2.5-7B-Instruct（BF16），实现全链路本地化，无需网络、无需 API Key。也支持通过 `--llm-model` 切换 0.5B 等其他模型

这是一种典型的"算力就近原则"：计算密集的编码放在本地加速器，生成部分可根据需求选择云端大模型或本地模型。

### 1.3 完整流程

```text
离线阶段（索引构建）:
  文档 → 文本清洗 → 滑动窗口分块 → NPU Embedding 编码 → FAISS 向量写入

在线阶段（查询）:
  用户问题 → NPU Embedding 编码 → FAISS 向量检索 Top-K → Prompt 拼接 → LLM API → 回答
```

---

## 2. RAG 架构

```text
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│ 文档      │───→│ 文本分块  │───→│ NPU      │───→│ FAISS 向量库  │
│ .md/.txt │    │ 512/128  │    │ Embedding│    │ (IndexFlatIP)│
└──────────┘    └──────────┘    └──────────┘    └──────┬───────┘
                                                       │
                       ┌───────────────────────────────┘
                       ↓
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────────┐
│ 用户问题  │───→│ NPU      │───→│ 向量检索       │───→│ LLM 生成     │
│          │    │ Embedding│    │ Top-K=5      │    │ API/本地NPU  │
└──────────┘    └──────────┘    └──────────────┘    └──────────────┘
```

**各组件职责：**

| 组件          | 职责                                   | 运行位置              |
| ------------- | -------------------------------------- | --------------------- |
| 文本分块      | 将长文档切分为语义完整的片段，控制粒度 | CPU                   |
| NPU Embedding | 将文本片段编码为 512 维语义向量        | NPU (Ascend 910B3)    |
| FAISS 向量库  | 存储向量、提供高效相似度检索           | CPU 内存              |
| LLM 生成      | 基于检索到的资料生成最终回答           | 云端（API）或本地 NPU |

这里有一个容易忽视的细节：**查询和文档必须使用同一个 embedding 模型编码**，因为向量相似度只有在同一语义空间中才有意义。如果换了模型，必须重建索引。

---

## 3. 技术选型

RAG pipeline 涉及 embedding 模型、推理框架、向量库和 LLM 四个环节的技术决策。以下选型以"复用现有 NPU 生态、最小化额外依赖"为原则，各组件均可替换为更强大的替代方案。

| 组件           | 选型                           | 理由                                     |
| -------------- | ------------------------------ | ---------------------------------------- |
| Embedding 模型 | `BAAI/bge-small-zh-v1.5`       | 中文优化，24MB 体积，NPU 加载 < 8s       |
| 推理框架       | PyTorch 2.1.0 + torch_npu      | 项目已有生态，直接复用                   |
| 向量库         | FAISS (IndexFlatIP)            | 轻量，CPU 侧检索，毫秒级                 |
| LLM            | OpenAI 兼容 API / 本地 Qwen2.5 | 默认使用外部 API，`--local` 切换本地推理 |
| 环境           | `/root/npu-learning/rag-env/`  | 独立 venv，不污染现有学习环境            |

### 3.1 模型选型对比

| 模型                   | 体积  | 维度 | 中文效果 | 适用场景             |
| ---------------------- | ----- | ---- | -------- | -------------------- |
| bge-small-zh-v1.5      | 24MB  | 512  | 良好     | 开发测试、快速原型   |
| bge-large-zh-v1.5      | 326MB | 1024 | 优秀     | 生产环境、高精度需求 |
| m3e-base               | 110MB | 768  | 良好     | 通用中文场景         |
| text2vec-large-chinese | 326MB | 1024 | 优秀     | 语义匹配、相似度计算 |

本次选用 `bge-small-zh` 是因为首次实验看重加载速度和调试便利性。升级到 `bge-large-zh` 只需改 `--model` 参数，索引结构会自动适配新维度。

### 3.2 为什么用 IndexFlatIP

FAISS 提供多种索引类型。`IndexFlatIP`（内积）是最简单的一种——暴力计算查询向量与所有文档向量的内积，不做任何近似压缩。它的特点是：

- **精度最高**：不损失检索精度，适合验证 embedding 模型效果
- **无需训练**：不像 IVF/HNSW 需要聚类训练
- **速度足够**：115 条向量检索 < 1ms；对于万级文档库，暴力搜索仍可接受

当文档量增长到十万级以上时，可以切换为 `IndexIVFFlat`（倒排索引 + 聚类）或 `IndexHNSWFlat`（图索引），以少量精度损失换取数量级的检索加速。

---

## 4. 环境搭建

### 4.1 版本兼容

CANN 8.0.1 + PyTorch 2.1.0 是一组较旧的版本，最新版的 transformers (5.x) 和 sentence-transformers (3.x) 均要求 PyTorch >= 2.4，直接安装会导致导入失败。因此需要锁定以下兼容版本：

| 包                       | 版本         | 原因                                                                             |
| ------------------------ | ------------ | -------------------------------------------------------------------------------- |
| torch                    | 2.1.0        | CANN 8.0.1 官方配套版本                                                          |
| torch_npu                | 2.1.0.post13 | NPU 后端，与 torch 版本强绑定                                                    |
| transformers             | 4.38.2       | PyTorch 2.1.0 兼容的最高 4.x 版本；5.x 起要求 PyTorch >= 2.4                     |
| sentence-transformers    | 2.7.0        | 兼容 transformers 4.x；3.x 起依赖 transformers >= 5.0                            |
| faiss-cpu                | 1.13.2       | 纯 CPU 版本，无 GPU/NPU 依赖，aarch64 原生支持                                   |
| numpy                    | <2 (1.26.4)  | torch_npu 2.1.0 使用 NumPy 1.x C API 编译，NumPy 2.x 会报 `_ARRAY_API not found` |
| decorator, attrs, psutil | latest       | CANN 编译器 (ATC/AOE) 的 Python 运行时依赖，缺失时 NPU 算子编译失败              |

版本锁定是本次环境搭建中耗时最长的一环。教训：在较老的 CANN 版本上引入新框架时，**先确认 PyTorch 版本再反查依赖兼容性**，可以避免大量试错。

### 4.2 安装步骤

```bash
# 1. 创建独立 venv（区别于 /root/npu-learning/venv）
python3 -m venv /root/npu-learning/rag-env

# 2. 加载环境（顺序不可颠倒：先 CANN 后 venv）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /root/npu-learning/rag-env/bin/activate

# 3. 按依赖顺序安装
pip install torch==2.1.0 torch_npu==2.1.0.post13
pip install transformers==4.38.2 sentence-transformers==2.7.0
pip install faiss-cpu decorator attrs psutil 'numpy<2'
```

---

## 5. 使用方法

所有命令都需要通过 `ASCEND_RT_VISIBLE_DEVICES=7` 指定 NPU 设备，否则 torch_npu 默认可见全部 8 张卡。

```bash
ASCEND_RT_VISIBLE_DEVICES=7 python3 rag_pipeline.py <subcommand>
```

脚本提供 4 种子命令，覆盖从离线索引到在线问答的完整流程：

### 5.1 索引文档

使用 `index` 子命令构建索引：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 python3 rag_pipeline.py index \
  --docs ./docs/ \
  --chunk-size 512 \
  --chunk-overlap 128 \
  --index-path ./rag_index
```

索引过程分三步：

1. **文档加载**：递归扫描 `--docs` 目录，读取所有 `.md` / `.txt` / `.rst` 文件
2. **文本分块**：按空行分段 → 拼接至 ~512 字符 → 相邻块重叠 128 字符
3. **NPU 编码**：批量送入 BGE 模型在 NPU 上推理，输出 512 维归一化向量

输出两个文件：

- `rag_index.index`：FAISS 二进制索引
- `rag_index.chunks.json`：文本块元数据（原文、来源路径、索引位置）

### 5.2 语义搜索

使用 `search` 子命令，不需要 LLM API：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 python3 rag_pipeline.py search "如何安装 torch_npu?" --top-k 5 --show-text
```

这个子命令仅验证检索质量，不调用 LLM。适合在配置 API 之前先测试 embedding 模型和索引的效果。加上 `--show-text` 会输出检索到的完整文本片段，方便人工评估相关性。

### 5.3 RAG 问答

使用 `ask` 子命令。支持两种 LLM 后端：

**方式一：外部 API（默认）**：

```bash
export RAG_LLM_ENDPOINT="https://api.example.com/v1/chat/completions"
export RAG_LLM_API_KEY="sk-xxx"
export RAG_LLM_MODEL="gpt-4"  # 可选，默认 gpt-3.5-turbo

ASCEND_RT_VISIBLE_DEVICES=7 python3 rag_pipeline.py ask "什么是 NPU?"
```

**方式二：本地 LLM 推理（`--local`）**

```bash
ASCEND_RT_VISIBLE_DEVICES=7 python3 rag_pipeline.py ask --local "什么是 NPU?"
```

加上 `--local` 后，脚本会自动加载 Qwen2.5-7B-Instruct（BF16）到 NPU 进行推理，无需配置任何 API 环境变量。模型首次运行从 HuggingFace 下载（~15GB），后续加载使用缓存。可通过 `--llm-model` 切换模型（如 `--llm-model Qwen/Qwen2.5-0.5B-Instruct` 使用 0.5B）。

一次完整的 RAG 调用包含 4 步：编码查询 → 向量检索 → 拼接 prompt → 调用 LLM。返回结果包含答案、引用来源列表、各阶段耗时。

### 5.4 交互式问答

使用 `query` 子命令进入 REPL 交互界面：

```bash
# 使用外部 API
ASCEND_RT_VISIBLE_DEVICES=7 python3 rag_pipeline.py query --top-k 5

# 使用本地 LLM
ASCEND_RT_VISIBLE_DEVICES=7 python3 rag_pipeline.py query --local --top-k 5
```

进入 REPL 交互界面，支持以下命令：

- 直接输入问题 → 立即执行 RAG 流程并打印结果
- `/topk N` → 动态调整检索数量（实时生效）
- `/quit` → 退出

交互模式下 embedding 模型只加载一次，后续查询无需等待模型加载。

---

## 6. 性能测试

以下测试对比了 CPU 与 NPU 两种编码方式在相同数据上的性能差异，旨在量化 NPU 在 embedding 场景下的实际加速效果。

测试条件：7 篇 Ascend 学习文档 → 115 个文本块，BGE-small-zh-v1.5 (dim=512)，NPU 7。

| 阶段        | CPU (sentence-transformers) | NPU (PyTorch 直接推理) | 加速比 |
| ----------- | --------------------------- | ---------------------- | ------ |
| 模型加载    | 2.2s                        | 7.8s                   | —      |
| 编码 115 条 | 337.8s                      | 0.8s                   | ~422x  |
| 吞吐        | 0.34 条/s                   | 153 条/s               | —      |
| 单次检索    | —                           | ~0.3ms                 | —      |

> [!NOTE]
> 模型首次下载后缓存于 `~/.cache/huggingface/`，后续加载直接使用缓存。

### 6.1 为什么 NPU 这么快

CPU 编码慢有两个原因：

1. **aarch64 架构**：服务器是 ARM 架构，sentence-transformers 在 ARM CPU 上没有 x86 那样的 MKL/oneDNN 优化，纯 CPU 推理效率很低
2. **小 batch**：默认 batch_size=32，CPU 无法充分利用缓存

NPU 的 Da Vinci 架构有专门的 Cube 单元处理矩阵乘法，BERT 类模型的 Transformer 层本质上就是大量的矩阵乘法和注意力计算，天然适合 NPU 加速。即使不考虑 batch 优势，单次推理的延迟也从 ~3 秒降到了 ~7 毫秒。

### 6.2 模型加载慢的原因

NPU 加载比 CPU 慢（7.8s vs 2.2s），多出的时间主要是 **NPU 算子编译**。PyTorch 模型首次在 NPU 上执行时，CANN 的图编译器需要将每个算子转换为 NPU 可执行的二进制指令。这是**一次性开销**——模型加载后，后续所有推理都不再需要编译。

---

## 7. 检索效果

以 7 篇 Ascend 学习文档为语料库（01-hello-npu ~ 07-npu-smi-reference）进行测试：

```text
查询: "如何安装 torch_npu?"
检索结果 Top-3:
  [1] 02_hello_npu_first_program.md (相关度: 0.5986)
      — 包含 "pip install torch==2.1.0 pip install pyyaml setuptools 'numpy<2'" 等安装命令
  [2] 02_hello_npu_first_program.md (相关度: 0.5881)
      — CUDA vs NPU API 对照表，包括 .cuda() → .npu() 的迁移方法
  [3] 02_hello_npu_first_program.md (相关度: 0.5627)
      — NPU 环境检查代码：torch.npu.is_available()、device_count() 等
```

检索准确命中了安装相关的文档段落。相关度分数在 0.55-0.60 之间，说明文档内容与查询确实存在语义关联。

但也观察到局限性：

| 查询类型                      | 效果 | 原因                                                         |
| ----------------------------- | ---- | ------------------------------------------------------------ |
| 事实查询（"如何安装 X"）      | 良好 | 关键词匹配 + 语义均可定位                                    |
| 跨文档对比（"X 和 Y 的区别"） | 一般 | 小模型 (512 维) 对比语义能力弱，且语料中无直接对比段落       |
| 抽象概念（"NPU 的核心优势"）  | 中等 | 相关段落分散在多篇文档，Top-K 检索只能召回片段、丢失全局视角 |

**改进方向**：升级到 `bge-large-zh-v1.5` (dim=1024) 可提升语义理解精度；加入 BM25 关键词检索做双路融合可覆盖事实查询的盲区。

---

## 8. 实现细节

### 8.1 NPU Embedding 推理

这是实现中最关键的环节。sentence-transformers 2.7.0 的 `SentenceTransformer.encode()` 方法内部使用 `model.to(device)`，但该版本只支持 `cpu` 和 `cuda`，传入 `npu` 会导致错误。

解决方案：绕过 sentence-transformers 的便捷接口，直接操作底层 HuggingFace 模型：

```python
# 1. 用 sentence-transformers 加载模型（利用它内置的 tokenizer 和配置）
model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")

# 2. 提取底层 HuggingFace transformer 并手动移到 NPU
for module in model.modules():
    if hasattr(module, "auto_model"):
        transformer = module.auto_model.to("npu:0").eval()
        break

# 3. 手动 tokenize → NPU 推理 → mean pooling → L2 归一化
encoded = tokenizer(texts, padding=True, truncation=True, max_length=512,
                    return_tensors="pt")
input_ids = encoded["input_ids"].to("npu:0")
attention_mask = encoded["attention_mask"].to("npu:0")

with torch.no_grad():
    outputs = transformer(input_ids=input_ids, attention_mask=attention_mask)
    # last_hidden_state: [batch_size, seq_len, hidden_dim]
    token_embeddings = outputs[0]

    # Mean pooling: 用 attention_mask 做加权平均，忽略 padding token
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    mean_emb = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    # L2 归一化，使 FAISS 内积等价于余弦相似度
    mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=1)

result = mean_emb.cpu().numpy()  # FAISS 需要 numpy 数组
```

关键细节：

- **mean pooling vs CLS token**：BGE 模型使用 mean pooling（所有 token 的加权平均），而非取 `[CLS]` 位置的输出。这与 BERT 原版不同，用错 pooling 方式会导致向量质量显著下降
- **L2 归一化**：将向量模长归一为 1，使 `IndexFlatIP`（内积）等价于余弦相似度。FAISS 的 `IndexFlatIP` 比 `IndexFlatL2` 略快，且余弦相似度在语义匹配场景中通常优于欧氏距离
- **attention_mask 的 clamp(min=1e-9)**：防止空文本导致除零错误

### 8.2 文本分块

分块质量直接影响检索效果。块太大会引入噪声（检索到的块中包含不相关的内容），太小则丢失上下文。

本实现采用**段落感知 + 滑动窗口**策略：

1. **按空行切段落**（`\n\n`）：尊重文档的自然边界，避免在句子中间截断
2. **拼接至 chunk_size**：将相邻段落拼接，使块的大小尽可能接近 512 字符
3. **滑动窗口重叠**：相邻块共享 128 字符的尾部/头部重叠。这确保即使一个概念恰好跨越了块边界，也能在至少一个块中被完整保留

```text
原文段落: [P1] [P2] [P3] [P4] [P5] ...
Chunk 1:  [P1 P2 P3      ]
Chunk 2:        [P3 P4 P5    ]   ← 重叠区域保留 P3 的上下文
```

### 8.3 FAISS 向量检索

`IndexFlatIP` 将查询向量与所有文档向量逐一计算内积，返回分数最高的 Top-K：

```python
# FAISS 内部等价于:
scores = query_emb @ doc_embs.T    # [1, dim] @ [dim, N] → [1, N]
topk_indices = argsort(scores)[-K:] # 取最大的 K 个
```

因为所有向量都是 L2 归一化的，`scores` 的范围在 [0, 1] 之间，1.0 表示完全匹配。

索引持久化使用 FAISS 自带的 `write_index` / `read_index`，配合 JSON 文件存储文本块元数据。两个文件需要保持一一对应——`chunk N` 的向量存储在 FAISS 索引的第 N 行，元数据存储在 JSON 数组的第 N 个元素。

### 8.4 LLM API

API 请求格式遵循 OpenAI Chat Completions 标准，因此兼容所有提供 OpenAI 兼容接口的服务（如 vLLM、Ollama、各云厂商的模型服务）。

RAG 的 prompt 构造方式是决定回答质量的关键。本实现使用 **system prompt 约束 + 参考资料注入** 的模式：

```text
System: 你是一个基于参考资料回答问题的助手。请根据下面提供的参考资料
        回答问题。如果参考资料中没有相关信息，请如实告知，不要编造。
        回答时请引用参考资料的来源路径。

User:   参考资料:

        [来源: 01_environment/02_hello_npu_first_program.md]
        ...（检索到的文档片段）...

        [来源: 03_pytorch_npu/01_cuda_to_npu_migration.md]
        ...（检索到的文档片段）...

        问题: 如何安装 torch_npu?
```

几个 prompt 设计要点：

- **"不要编造"**：明确告诉模型如果找不到就承认，减少幻觉
- **"引用来源路径"**：让模型在回答中标注出处，用户可以回溯验证
- **参考资料放在 User 消息中**：某些模型对 system 消息有长度限制，将长文本放在 user 消息更安全

### 8.5 BGE 查询前缀

BGE 模型在训练时对查询和文档使用了不同的指令模板。查询时需要添加前缀：

```python
# BGE 训练时查询侧加了指令前缀，推理时也必须一致
query = f"为这个句子生成表示以用于检索相关文章：{query}"
```

不加前缀会导致查询向量与文档向量不在同一语义子空间，相关度分数整体偏低。这是使用 BGE 系列模型时最容易踩的坑。

### 8.6 本地 LLM 推理

`LocalLLMClient` 在 NPU 上运行 Qwen2.5-0.5B-Instruct，接口与 `LLMClient` 完全一致（都实现 `chat(messages, temperature, max_tokens) -> str`），因此 `RAGPipeline` 无需任何修改即可切换后端。

与 Phase 10 独立推理脚本的区别在于 prompt 的构造方式。独立推理使用 Qwen2.5 的 ChatML 模板（`apply_chat_template`），但 RAG 场景下 prompt 中包含大量检索到的文档片段，ChatML 的 `<|im_start|>` / `<|im_end|>` 标记会与 Markdown 内容中的特殊字符产生冲突。因此 `LocalLLMClient.chat()` 使用简化的纯文本 prompt 结构：

```python
prompt = (
    f"{system_text}\n\n"
    f"{user_text}\n\n"
    f"请基于上面的参考资料回答问题，用 2-3 句话简洁回答。"
    f"如果资料中没有相关信息，回答'参考资料中未提及'。"
)
```

几个设计考量：

- **"用 2-3 句话简洁回答"**：0.5B 模型能力有限，明确约束输出长度可以防止它生成冗长但不连贯的内容
- **"如果资料中没有相关信息..."**：与 system prompt 中的"不要编造"形成双重约束，降低小模型幻觉
- **repetition_penalty=1.1**：小模型在 RAG prompt（包含大量重复格式的文档片段）中容易陷入重复循环，轻微的重复惩罚可以有效缓解
- **使用 ChatML 模板**：通过 `tokenizer.apply_chat_template()` 生成 Qwen2.5 训练时使用的 ChatML 格式，确保模型正确理解消息边界（在早期迭代中曾使用纯文本拼接，但 0.5B 模型在纯文本模式下更容易产生重复，ChatML 格式显著改善了生成质量）

与外部 API 的对比：

| 特性     | 外部 API            | 本地 LLM（7B BF16）    | 本地 LLM（0.5B）       |
| -------- | ------------------- | ---------------------- | ---------------------- |
| 回答质量 | 高（大模型）        | 良好（7B，谨慎但准确） | 一般（0.5B 能力有限）  |
| HBM 占用 | —                   | ~15 GB                 | ~1 GB                  |
| 延迟     | 网络延迟 + 推理时间 | 纯推理时间             | 纯推理时间             |
| 隐私     | 数据发送至第三方    | 数据不出服务器         | 数据不出服务器         |
| 依赖     | 需要网络 + API Key  | 仅需 NPU + 模型文件    | 仅需 NPU + 模型文件    |
| 适用场景 | 生产环境、复杂问题  | 高质量本地 RAG         | 快速验证、资源受限场景 |

实测对比（BF16 推理）：

| 问题                   | 7B                                                                 | 0.5B                                              |
| ---------------------- | ------------------------------------------------------------------ | ------------------------------------------------- |
| "什么是 NPU？"         | "参考资料中未提及"（严格遵循指令）                                 | "NPU 是一种专用处理器，专为 AI 任务设计..."       |
| "如何安装 torch_npu？" | "未提及具体步骤，但可推断需 `pip install torch-npu==2.1.0.post13`" | "需要在代码中导入 torch_npu，检查 is_available()" |

7B 模型更谨慎、更诚实（严格遵守"不要编造"指令），但有时过于保守；0.5B 更愿意给出泛泛的描述，但可能不准确。通过 `--llm-model` 参数可随时切换。

> [!NOTE]
> Qwen2.5-7B-Instruct 在 FP16 下会产生 NaN（深层激活值导致 Attention 点积溢出），但在 **BF16** 下完全正常。BF16 的 8 位指数（与 FP32 相同）提供了足够的动态范围，且 HBM 占用与 FP16 相同（~15 GB）。详见 [Qwen2.5-7B FP16 NaN 诊断报告](../11_llm_inference/02_fp16_nan_debug.md)。

---

## 9. 文件清单

```text
07_rag_on_npu/
├── README.md
├── 01_rag_pipeline_on_npu.md    # 本文
└── rag_pipeline.py              # RAG 完整 pipeline，包含以下类:
                          DocumentLoader   — 文档加载（.md/.txt/.rst）
                          TextChunker      — 滑动窗口文本分块
                          EmbeddingEngine  — NPU embedding 推理
                          VectorStore      — FAISS 索引管理（CRUD + 持久化）
                          LLMClient        — OpenAI 兼容 API 调用
                          LocalLLMClient   — 本地 NPU LLM 推理（--local）
                          RAGPipeline      — 流程编排 + CLI
```

---

## 10. 后续扩展

本地 LLM 集成已完成，RAG 全链路可在 NPU 上独立运行（embedding + FAISS + 7B BF16 LLM）。以下为进一步优化方向：

- **量化推理**：INT8/INT4 量化降低 HBM 占用，使 14B+ 模型成为可能
- **流式输出**：使用 `TextStreamer` 实现逐 token 返回
- **混合检索**：BM25（关键词匹配）+ 向量（语义匹配）双路召回，通过 RRF 融合排序
- **对话历史**：多轮问答中维护消息上下文，自动判断是否需要重新检索

---

## 参考链接

- [LMCache-Ascend](https://github.com/LMCache/LMCache-Ascend)
- [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [sentence-transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ascend CANN 文档](https://www.hiascend.com/document)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
