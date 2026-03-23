# 文本嵌入 (Text Embeddings) 技术指南

本目录包含大语言模型嵌入（Embeddings）技术的详细理论解析、演进历史和实践指南。嵌入技术是将自然语言转换为计算机可处理的高维稠密向量的核心基础。

## 1. 核心文档

- **[深入了解文本嵌入技术 (Text Embeddings Comprehensive Guide)](text_embeddings_comprehensive_guide.md)** 🌟推荐阅读
  - 一篇全面的深度长文，详细讲解了从词袋模型 (Bag of Words)、TF-IDF 到 Word2Vec、Transformer 句子嵌入的演进历史。
  - 涵盖了向量距离计算（L2, 曼哈顿, 点积, 余弦相似度）、降维可视化（PCA, t-SNE）以及聚类、分类、异常检测和 RAG 等实际应用场景。
- **[LLM 嵌入技术详解：图文指南 (LLM Embeddings Explained)](LLM_embeddings_explained_visual_guide.zh-CN.md)**
  - 以直观的图解方式解释 LLM 中 Embedding 的工作原理及向量空间的几何意义。
- **[文本嵌入技术快速入门 (Text Embeddings Guide)](text_embeddings_guide.md)**
  - 快速上手指南，适合初学者了解 Embedding 的基本概念与调用方式。
- **[大模型 Embedding 层与独立 Embedding 模型：区别与联系](embedding.md)**
  - 深入剖析 LLM 内部自带的 Embedding 层与如 BGE、OpenAI text-embedding-3 等外部独立 Embedding 模型的架构差异与协作关系。

## 2. 图片资源说明

- **[`img/`](img/)**: 存放文档中使用的相关配图，包括公式、可视化散点图、热力图等，已根据内容统一规范命名。
- **[`images/`](images/)**: 存放其他指南文档的历史参考图片。

## 3. 学习路径建议

1. 首先阅读 **[图文指南](LLM_embeddings_explained_visual_guide.zh-CN.md)** 建立直观的向量空间概念。
2. 随后精读 **[深入了解文本嵌入技术](text_embeddings_comprehensive_guide.md)** 掌握算法演进、相似度计算与可视化实战。
3. 了解架构差异，阅读 **[大模型 Embedding 层与独立 Embedding 模型的区别](embedding.md)**。

## 4. 相关资源

- [RAG 与向量检索技术](../../../07_rag_and_tools/README.md)
- [Token 机制解析](../token/README.md)
