# 上海人工智能实验室开源工具 MinerU 助力复杂 PDF 高效解析提取

> 参考：https://www.shlab.org.cn/news/5443982

`MinerU` 是一款开源智能文档解析工具，专注于将 **PDF**、**网页**、**电子书**等多模态内容转换为结构化数据（如 Markdown、JSON），支持 AI 训练、知识管理、RAG（检索增强生成）等场景。

`MinerU` 不仅能将混合了图片、公式、表格、脚注等在内的复杂多模态 `PDF` 文档精准转化为清晰、易于分析的 `Markdown` 格式；同时支持从包含广告等各种干扰信息或者复杂格式的网页、电子书中快速解析、抽取正式内容，有效提高`AI`语料准备效率，助力各行业利用大模型、`RAG`等技术，结合学术文献、财务报告、法律文件、电子书籍等专业文档，打造垂直领域的新知识引擎。

> **MinerU 项目地址**：`https://github.com/opendatalab/MinerU`
> 
> **PDF-Extract-Kit PDF模型解析工具链代码**: `https://github.com/opendatalab/PDF-Extract-Kit`
> 

**MinerU 在线网站**：

**_https://mineru.net/OpenSourceTools/Extractor/PDF_**

## 一、MinerU功能介绍

`MinerU` 可以将 `PDF` 转化为 `Markdown` 格式。支持转换本地文档或者位于支持`S3`协议对象存储上的文件。

**主要功能包含：**

* 支持多种前端模型输入
* 删除页眉、页脚、脚注、页码等元素
* 符合人类阅读顺序的排版格式
* 保留原文档的结构和格式，包括标题、段落、列表等
* 提取图像和表格并在`Markdown`中展示
* 将公式转换成`LaTex`
* 乱码`PDF`自动识别并转换
* 支持`CPU`和`GPU`环境
* 支持 `Windows/Linux/Mac`平台

## 二、PDF提取流程及技术架构

`PDF`文档相比网页、电子书等结构标准化的文件含有更多复杂的元素，处理更具挑战性和代表性，所以接下来，将以PDF为代表，重点介绍 MinerU 如何实现高质量文档数据提取。

![638581060581520000.jpg](https://img.shlab.org.cn/pjlab/images/2024/08/638581060581520000.jpg "638581060581520000.jpg")

**MinerU PDF文档提取，主要由4大环节构成：**

### **PDF文档分类预处理**

`MinerU`支持不同类型的PDF文档提取，包括文本型`PDF`、图层型`PDF`、扫描版`PDF`；初始阶段，输入`PDF`文档，系统会启用文档分类模块，提取`PDF`元数据，检测是否有乱码，是否是扫描版，进行`PDF`类型识别预处理。

> 注：**文本型PDF**：文字可以复制；
> **图层型PDF**：文字不可复制，解析乱码

### **模型解析，PDF内容提取**

紧接着，利用高质量`PDF`模型解析工具链进一步对`PDF`文档进行`Layout`区块布局检测，准确定位标题、正文、图片、表格、脚注、边注等重要元素位置，与此同时，结合公式检测模型定位公式区域。最后结合高质量公式识别及`OCR`技术提取准确的文本、公式内容，存储到`JSON`文件中。

![6.jpg](https://img.shlab.org.cn/pjlab/images/2024/08/638581061134730000.jpg "638581061134730000.jpg")

### **流水线处理，支持多种格式输出**

模型处理的数据会输入流水线，进行后处理：确定块级别顺序，删减无用元素，依靠版面对内容排序、拼装，保证正文流畅。处理方式包括：**坐标修复**、**高iou处理**、**图片**、**表格描述合并**、**公式替换**、**图标转储**、**Layout排序**、**无用移出**、**复杂布局过滤**等。

流水线处理好的文档信息会变为一个统一的**中间态**：`middle-json`（包含PDF解析出来的所有的信息），开发者可以按照使用需求自定义输出`Layout`、`Span`、`Markdown`、`Content list`等不同的格式。

> 注：`Content list`是作者团队开发的一套列表样的序列结构格式，比`Markdown`格式能保留更多信息，可用于多模态、`NLP`等大模型训练。

### **PDF提取结果质检**

团队利用由论文、教材、试卷、研报等多种类型文档组成的人工标注的`PDF`自测评测集，对整个流程进行检测，保证每次开发调优、算法改进后，提取效果越来越好；同时利用可视化质检工具，将`PDF`提取结果进行人工质检与标注，再反馈给模型训练，进一步提升模型能力。

**详细项目全景图如下：** ![638581061501110000.jpg](https://img.shlab.org.cn/pjlab/images/2024/08/638581061501110000.jpg "638581061501110000.jpg")

## 三、高质量PDF模型解析工具链

`MinerU PDF`模型解析工具链 P`DF-Extract-Kit`，主要由四个关键模块组成：

* **布局检测**：使用 LayoutLMv3 微调出来的检测模型进行区域检测，如图像，表格、标题、文本等；
* **公式检测**：使用基于 YOLOv8 自研的公式检测模型进行公式检测，包含行内公式和行间公式；
* **公式识别**：使用自研的 UniMERNet 公式识别模型进行公式识别；
* **光学字符识别**：使用 PaddleOCR 模型进行文本识别。

在论文、教材、研报、财报等多样性的`PDF`文档上，`MinerU`的`pipeline`都能得到准确的提取结果，对于扫描模糊、水印等情况也有较高鲁棒性。

 ![638581062272580000.jpg](https://img.shlab.org.cn/pjlab/images/2024/08/638581062272580000.jpg "638581062272580000.jpg")

## 四、评测指标

### **布局检测**

作者团队将 `MinerU` 与现有的开源 `Layout` 检测模型做了对比，包括 `DocXchain`、`Surya`、`360LayoutAnalysis` 的两个模型。而 `LayoutLMv3-SFT` 指的是他们在`LayoutLMv3-base-chinese` 预训练权重的基础上进一步做了SFT训练后的模型。论文验证集由`402`张论文页面构成，教材验证集由`587`张不同来源的教材页面构成。

![638581063007750000.jpg](https://img.shlab.org.cn/pjlab/images/2024/08/638581063007750000.jpg "638581063007750000.jpg")

### **公式检测**

作者团队将 `MinerU` 与开源的模型 `Pix2Text-MFD` 做了对比。其中，`YOLOv8-Trained` 是他们在`YOLOv8l` 模型的基础上训练后的权重。论文验证集由`255`张论文页面构成，多源验证集由`789`张不同来源的页面构成，包括教材、书籍等。

 ![638581063546060000.jpg](https://img.shlab.org.cn/pjlab/images/2024/08/638581063546060000.jpg "638581063546060000.jpg")

### **公式识别**

公式识别作者团队则直接使用了 `UniMERNet` 的权重，没有进一步的`SFT`训练，其精度验证结果可以在其`GitHub`页面获取。

![638581063850150000.jpg](https://img.shlab.org.cn/pjlab/images/2024/08/638581063850150000.jpg "638581063850150000.jpg")

### **光学字符识别**

使用了`PaddleOCR` 官方提供的权重，没有做进一步的训练和验证，因此不涉及验证代码。

**评测结果显示，MinerU在布局检测、公式检测、公式识别多个维度上性能都远超其他开源模型，识别准确率也非常不错。**

更多评测详情，请访问：

[https://github.com/opendatalab/PDF-Extract-Kit/blob/main/assets/validation/README-zh\_CN.md](https://github.com/opendatalab/PDF-Extract-Kit/blob/main/assets/validation/README-zh_CN.md)

## 五、MinerU部署及使用

**MinerU 完整部署及使用文档**：
[https://github.com/opendatalab/MinerU](https://github.com/opendatalab/MinerU)

**OpenDataLab GitHub仓库**：
[https://github.com/opendatalab](https://github.com/opendatalab)

**多模态标注工具 LabelU**：
[https://github.com/opendatalab/labelU](https://github.com/opendatalab/labelU)

**多模态对话标注管理平台Label-LLM**：
[https://github.com/opendatalab/LabelLLM](https://github.com/opendatalab/LabelLLM)