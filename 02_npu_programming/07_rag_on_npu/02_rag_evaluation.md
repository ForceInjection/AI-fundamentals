# RAG 评估体系

## 1. 背景

### 1.1 为什么要评估 RAG

RAG pipeline 建成后，直观感受是"回答看起来还行"，但缺乏量化标准。评估需要回答：

- **检索质量**：检索到的文档中是否包含了正确答案？
- **生成质量**：模型基于检索结果生成的回答是否准确、完整？
- **模型对比**：7B 比 0.5B 好在哪里？值不值得多用 14GB HBM？

### 1.2 两层评估架构

```text
Layer 1: 检索评估 (Retrieval Eval)
  ├── MRR: 第一个相关结果排在第几位？
  ├── Recall@k: top-k 结果覆盖了多少正确答案？
  └── Precision@k: top-k 结果中有多少是相关的？

Layer 2: 生成评估 (Generation Eval)
  ├── 回答对比: 7B vs 0.5B vs 外部 API
  ├── 质量维度: 正确性、完整性、是否幻觉
  └── 效率维度: 速度、长度、HBM 占用
```

检索评估是**无 LLM 依赖**的确定性指标——只需要 FAISS 索引和标注的 ground truth chunk IDs。生成评估需要对比多个 LLM 后端的输出。

### 1.3 评估数据集设计

选择 7 个查询，覆盖三种类型：

| 类型     | 示例                      | 特点                         |
| -------- | ------------------------- | ---------------------------- |
| 事实查询 | "NPU 的 HBM 带宽是多少？" | 答案在单个 chunk，明确可验证 |
| 概念查询 | "什么是达芬奇架构？"      | 答案跨多个 chunk，需要归纳   |
| 过程查询 | "如何安装 torch_npu？"    | 需要步骤，可能跨文档         |

每个查询标注了 ground truth chunk IDs（通过实际检索结果 + 人工确认 chunk 内容）。

---

## 2. 检索评估

### 2.1 方法

`rag_eval.py` 实现检索评估，依赖 `rag_pipeline` 中的 `EmbeddingEngine` 和 `VectorStore`，核心指标：

- **MRR (Mean Reciprocal Rank)**：第一个正确答案的排名的倒数。MRR=1.0 表示总是排在第一位
- **Recall@k**：top-k 中覆盖了 ground truth 的比例。Recall@5=1.0 表示所有答案都在前 5 个结果中
- **Precision@k**：top-k 中正确的占比

### 2.2 评估结果

环境：Ascend 910B3, bge-small-zh-v1.5 (dim=512), IndexFlatIP。

| 指标     | 值                    |
| -------- | --------------------- |
| MRR      | **0.86**              |
| Recall@1 | 0.63                  |
| Recall@3 | **1.00**              |
| Recall@5 | 1.00                  |
| 查询数量 | 8 (7 类型 × 多个变体) |

**解读**：

- Recall@3=100%：对于有明确答案的查询，正确答案总是在前 3 个检索结果中
- MRR=0.86：大部分查询的第一个结果就是正确答案，少数查询需要翻到第 2 个
- 63% 的首位命中率：还有改进空间（更好的 embedding 模型、混合检索）

### 2.3 检索失败案例分析

从原始 15 个查询中，7 个查询无法有效检索到正确答案（未纳入最终评估数据集）：

| 失败查询                             | 原因分析                                                                 |
| ------------------------------------ | ------------------------------------------------------------------------ |
| "什么是 RAG？"                       | bge-small 对英文缩写"RAG"理解不足，检索结果全是 dmi-reference 和架构文档 |
| "什么是 FlashAttention？"            | 同上，英文复合词嵌入表示不佳                                             |
| "什么是 KV Cache？"                  | 同上                                                                     |
| "BGE 模型查询时需要加什么前缀？"     | "BGE"缩写 + 查询过于具体                                                 |
| "Mini-GPT 有多少参数？"              | "Mini-GPT"是项目特有名词，不在预训练语料中                               |
| "CUDA 代码如何迁移到 NPU？"          | 检索到了相关文档但不是最优 chunk                                         |
| "ASCEND_RT_VISIBLE_DEVICES 的作用？" | chunk 7 不在 top-5，被其他 NPU 配置相关 chunk 覆盖                       |

**改进方向**：

- **bge-large-zh-v1.5**（dim=1024）：更强的语义理解，对英文缩写更敏感
- **BM25 混合检索**：关键词匹配可以覆盖缩写和专用名词
- **查询改写**：将"什么是 RAG"改写为"检索增强生成是什么"，改善召回

---

## 3. 生成评估

### 3.1 方法

`rag_eval_ragas.py` 用同样的问题和检索结果，对比 7B BF16 和 0.5B BF16 的生成输出。评估维度包括准确率、速度、回答长度。

**ragas 环境说明**：`rag-env` 中 ragas 与 langchain 存在版本冲突（`langchain-community 0.4.2` 不兼容 `langchain-core 1.4.0`）。解决方案是创建独立的 `eval-env`：

```bash
python3 -m venv /root/npu-learning/eval-env
source /root/npu-learning/eval-env/bin/activate
pip install torch==2.1.0 torch_npu==2.1.0.post13 decorator attrs psutil
pip install 'langchain-core>=0.3,<0.4' 'langchain-community>=0.3,<0.4'
pip install ragas rapidfuzz
pip install faiss-cpu 'numpy<2' sentence-transformers==2.7.0 transformers==4.38.2
```

核心对比评估（7B vs 0.5B）不依赖 ragas，在 `rag-env` 或 `eval-env` 中均可运行。RAGAS 指标（Context Recall/Precision）需在 `eval-env` 中启用。

### 3.2 对比结果

| 维度                 | 7B BF16   | 0.5B BF16 |
| -------------------- | --------- | --------- |
| 正确回答             | 5/7 (71%) | 4/7 (57%) |
| "未提及"（真实回答） | 2/7       | 0/7       |
| 事实错误             | 0/7       | 3/7       |
| 平均速度             | 2.3s/查询 | 2.1s/查询 |
| 平均长度             | 78 字符   | 66 字符   |
| HBM 占用             | ~15 GB    | ~1 GB     |

### 3.3 逐题对比

| 问题                   | 7B                       | 0.5B                     |
| ---------------------- | ------------------------ | ------------------------ |
| "NPU 的 HBM 带宽？"    | **1538 GB/s** ✓          | "训练吞吐的关键瓶颈" △   |
| "什么是达芬奇架构？"   | **Cube/Vector/Scalar** ✓ | "分布式系统架构" ✗       |
| "如何安装 torch_npu？" | **pip install 命令** ✓   | "在 NPU 上运行脚本..." △ |
| "npu-smi 查看拓扑？"   | "参考资料中未提及"       | **"8 卡全互联"** ✓       |
| "910B3 的算力？"       | "参考资料中未提及"       | **"313.7 TFLOPS"** ✓     |
| "AI Core 计算单元？"   | **Cube/Vector/Scalar** ✓ | "L1/L0 Buffer" ✗         |
| "numpy 版本要求？"     | **<2** ✓                 | "2.1.0 或更高" ✗         |

### 3.4 行为模式分析

**7B 的行为**：谨慎、诚实

- 当文档中的信息不够明确时，选择说"未提及"而非猜测
- 对于查询 4（npu-smi 拓扑）和查询 5（TFLOPS），文档内容分布在多个 chunk，7B 没有找到直接答案就如实告知
- **优点**：不会编造 → 幻觉率低
- **缺点**：过于保守 → 漏报（文档中确实包含该信息，但模型没有识别）

**0.5B 的行为**：大胆、模糊

- 总是尝试给出回答，即使是猜测
- 查询 6 将 AI Core 的计算单元说成"L1 Buffer、Unified Buffer、L0 Buffer"（内存层次，不是计算单元）
- 查询 7 将 numpy 版本要求说反了（<2 说成 ≥2.1.0）
- **优点**：覆盖率高，有时猜对（查询 4、5）
- **缺点**：不确定时仍会编造 → 误报（看起来合理但实际错误）

**关键洞察**：在 RAG 场景中，7B 的"谨慎"策略更适合——RAG 的核心价值是**基于文档回答**，而非"知道什么答什么"。0.5B 的错误回答（"numpy ≥2.1.0"）可能造成真正的误导。

### 3.5 速度分析

7B 和 0.5B 的速度差距（2.3s vs 2.1s）比预期小很多。原因：

- 短回答场景（<100 字符）下，模型加载和 prompt 编码占据了大部分时间
- 实际生成阶段 7B ~1s, 0.5B ~0.5s，差异被固定开销掩盖
- 对于长回答（>500 字符），7B 会显著慢于 0.5B

---

## 4. 总结与推荐

### 4.1 当前配置评估

| 组件             | 评分  | 说明                                       |
| ---------------- | ----- | ------------------------------------------ |
| 检索 (bge-small) | ★★★★☆ | MRR 0.86, Recall@3 100%，但对英文缩写弱    |
| 生成 (7B BF16)   | ★★★★☆ | 准确但保守，不会幻觉                       |
| 生成 (0.5B BF16) | ★★☆☆☆ | 快速但不可靠，会编造                       |
| 整体 RAG         | ★★★★☆ | 检索 + 7B 组合可靠，需改进检索覆盖英文缩写 |

### 4.2 推荐改进路径

| 优先级 | 改进                              | 预期效果                        |
| ------ | --------------------------------- | ------------------------------- |
| **P0** | bge-large-zh-v1.5 (dim=1024)      | 英文缩写查询召回改善，MRR > 0.9 |
| **P0** | BM25 混合检索                     | 覆盖专用名词和缩写              |
| P1     | 查询改写（Query Rewriting）       | 改善长尾查询召回                |
| P1     | 更大的评估数据集（30+ 查询）      | 更稳定、更有统计意义的评估      |
| P2     | LLM-as-judge (RAGAS Faithfulness) | 自动评估回答质量，替代人工标注  |

---

## 5. 代码结构

```text
07_rag_on_npu/
├── rag_pipeline.py           # RAG 主流程
├── rag_eval.py               # 检索评估 (~180 行)
│   ├── EVAL_QUERIES          — 标注数据集
│   └── RetrievalEvaluator    — MRR/Recall@k/Precision@k
└── rag_eval_ragas.py         # 生成评估 (~260 行)
    ├── EVAL_DATASET          — 含参考答案
    ├── run_query_with_llm()  — 单次 RAG 查询
    └── main()                — 顺序加载 7B/0.5B 对比

rag_eval_results.json          # 远端评估结果
```

---

## 参考链接

- [RAGAS: Evaluation framework for RAG](https://docs.ragas.io/)
- [MRR (Mean Reciprocal Rank)](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
- [BGE Embedding Models](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [BM25 混合检索](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)
