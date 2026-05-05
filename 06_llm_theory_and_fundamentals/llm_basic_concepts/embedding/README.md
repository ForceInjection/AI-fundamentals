# 文本嵌入（Text Embeddings）技术指南

「句子相似」「检索最相关的 top-K」「异常聚类」这些看似不同的需求，归根到底都是一件事：把自然语言压成稠密向量，然后在向量空间里算几何。本目录系统梳理这条技术线——从词袋模型一路走到 Transformer 句向量，包含算法原理、距离度量、可视化技巧，以及在 RAG / 聚类 / 分类 / 异常检测等场景中的落地用法。

## 1. 核心文档

- **[深入了解文本嵌入技术](text_embeddings_comprehensive_guide.md)** 🌟 推荐主读  
  从 Bag-of-Words、TF-IDF 到 Word2Vec（CBOW / Skip-gram），再到 Transformer 句向量的完整演进图谱；配合 L2 / 曼哈顿 / 点积 / 余弦相似度四类距离度量、PCA / t-SNE 两类降维可视化，以及聚类、分类、异常检测、RAG 四类下游场景的动手样例。
- **[LLM 嵌入技术图文指南](LLM_embeddings_explained_visual_guide.zh-CN.md)**  
  以几何直觉+可视化把 Embedding 讲成「向量空间里的位置与距离」，适合在读理论前先建立直觉。
- **[文本嵌入快速入门](text_embeddings_guide.md)**  
  面向初次接触 Embedding 的读者，给出最短可行的概念讲解与调用样例。
- **[LLM 内嵌 Embedding 层 vs. 独立 Embedding 模型](embedding.md)**  
  剖析 LLM 自带的 Embedding 层与 BGE、OpenAI text-embedding-3 等外部独立模型在训练目标、输出维度、使用场景上的差异，以及在 RAG 系统里如何协同。

## 2. 图片资源

- **[`img/`](img/)** — 当前文档使用的配图：公式、聚类散点、距离度量、热力图等，命名已按主题规范化。
- **[`images/`](images/)** — 早期指南文档的历史配图，供对照阅读。

## 3. 学习路径建议

1. 先看 **[图文指南](LLM_embeddings_explained_visual_guide.zh-CN.md)**，在脑子里建立 Embedding = 向量空间里的点 这一几何直觉。
2. 再精读 **[深入了解文本嵌入技术](text_embeddings_comprehensive_guide.md)**，掌握算法演进 + 相似度计算 + 可视化的完整技能栈。
3. 最后看 **[LLM 内嵌 vs. 独立 Embedding 模型](embedding.md)**，区分 RAG 与训练语境下的用法差异。

## 4. 相关资源

- [RAG 与向量检索](../../../07_rag_and_tools/README.md)
- [Token 机制解析](../token/README.md)
