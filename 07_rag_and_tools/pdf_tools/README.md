# 文档智能解析工具

RAG 系统的上限是数据输入的质量——PDF 里的表格、公式、多栏排版、扫描图如果没解析好，后面的 chunking 和 embedding 再精巧也救不回来。**Garbage in, garbage out**。本目录对比三款主流开源 PDF / Office → Markdown 工具，覆盖从「中文复杂 PDF」到「广谱办公文档」的不同取舍。

## 1. 工具对比矩阵

| 工具                        | 主打场景                 | 中文能力 | 格式支持                                                   | 技术特点                                             |
| --------------------------- | ------------------------ | -------- | ---------------------------------------------------------- | ---------------------------------------------------- |
| **MinerU**（上海 AI Lab）   | 科研文献、财报、扫描 PDF | ✅ 强    | PDF / 网页 / 电子书 → Markdown / JSON                      | 多模态 PDF 模型管线，公式转 LaTeX，乱码 PDF 自动识别 |
| **Marker**（VikParuchuri）  | 英文书籍、科研论文       | ⚠️ 弱    | PDF / EPUB / MOBI → Markdown                               | 基于深度学习的布局检测，比 nougat 快 10 倍           |
| **MarkItDown**（Microsoft） | Office / 图片 / 音频     | 一般     | PDF / Office / 图像 OCR / 音频 STT / HTML / ZIP → Markdown | 广谱格式覆盖，适合通用 ETL                           |

## 2. 核心文档

### 2.1 MinerU：复杂中文 PDF 首选

- **[MinerU 高效解析 PDF](miner_u_intro.md)** — 详解 MinerU 的四阶段管线（预处理 → 布局分析 → 内容抽取 → Markdown 组装），重点介绍其针对中文复杂版式（多栏、公式、表格、页眉页脚）的处理策略，并给出 CLI / Docker / S3 三种运行方式。

### 2.2 Marker：英文文献深度解析

- **[Marker 源码解析（译）](marker_zh_cn.md)** — 译自 _journal.hexmos.com_ 的 Marker 源码深度解读。从准备、OCR、布局检测、文本抽取、后处理到 Markdown 输出，拆解 Marker 如何用 PyMuPDF + 布局模型把 PDF 转成高精度 Markdown。适合想理解 PDF AI 解析内部机制的读者。
- ⚠️ **注意**：Marker 中文支持较弱，中文文档优先选择 MinerU。

### 2.3 MarkItDown：广谱办公文档转换

- **[MarkItDown 入门与容器化](markitdown/README.md)** — Microsoft 开源的通用文档转 Markdown 工具，支持 PDF / PowerPoint / Word / Excel / 图像 OCR / 音频转写等 9 类输入；本目录提供加速国内构建的自定义 Dockerfile 与样例 PDF。

## 3. 如何选型

- 中文 PDF（财报 / 论文 / 教材） → **MinerU**。
- 英文科研文献 / EPUB / MOBI → **Marker**。
- Office 文档、图片 OCR、音频转写等广谱需求 → **MarkItDown**。
- 多工具对比：批量跑同一批 PDF，用 [`../rag_basics/evaluating_chunking_strategies_summary.md`](../rag_basics/evaluating_chunking_strategies_summary.md) 的 token 级指标做下游检索效果评估。

## 4. 相关资源

- [RAG 基础](../rag_basics/README.md) — 解析后的 Markdown 如何喂进 RAG 管线。
- [LLM 理论与基础 · 文件格式](../../06_llm_theory_and_fundamentals/llm_basic_concepts/file_formats/llm_file_formats_complete_guide.md) — 模型侧的文件格式知识。
