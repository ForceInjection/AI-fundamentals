# AI Fundermentals
## 相关硬件知识
- [PCIe 知识大全](https://mp.weixin.qq.com/s/dHvKYcZoa4rcF90LLyo_0A)
- [NVLink 入门](https://mp.weixin.qq.com/s/fP69UEgusOa_X4ZKLo30ig)
- [NVIDIA DGX SuperPOD：下一代可扩展的AI领导基础设施](https://mp.weixin.qq.com/s/a64Qb6DuAAZnCTBy8g1p2Q)

## 深入理解 GPU 架构
在准备在 GPU 上运行的应用程序时，了解 GPU 硬件设计的主要特性并了解与 CPU 的相似之处和不同之处会很有帮助。本路线图适用于那些对 GPU 比较陌生或只是想了解更多有关 GPU 中计算机技术的人。不需要特定的并行编程经验，练习基于 CUDA 工具包中包含的标准 NVIDIA 示例程序。

- [GPU 特性](gpu_architecture/gpu_characteristics.md)
- [GPU 内存](gpu_architecture/gpu_memory.md)
- [GPU Example: Tesla V100](gpu_architecture/tesla_v100.md)
- [GPUs on Frontera: RTX 5000](gpu_architecture/rtx_5000.md)
- 练习：
	- [Exercise: Device Query](gpu_architecture/exer_device_query.md)
	- [Exercise: Device Bandwidth](gpu_architecture/exer_device_bandwidth.md)

### GPU 架构和编程模型介绍
- [GPU Architecture and Programming — An Introduction](gpu_programming/gpu_programming_introduction.md)

### 其他相关知识点
- [深入理解 Nvidia CUDA 核心（vs. Tensor Cores vs. RT Cores)](cuda/cuda_cores_cn.md)

## CUDA 学习材料
### 快速入门
- [并行计算、费林分类法和 CUDA 基本概念](https://mp.weixin.qq.com/s/NL_Bz8JB-LdAtrQake7EdA)
- [CUDA 编程模型入门](https://mp.weixin.qq.com/s/IUYzzgt6DUYhfaDnbxoZuQ)
- [CUDA 并发编程之 Stream 介绍](cuda/cuda_streams.md)

### 参考资料
- [CUDA Reading Group 相关讲座](https://mp.weixin.qq.com/s/6sOrNzG0UeVBes8stWSoWA): [GPU Mode Reading Group](https://github.com/gpu-mode)
- [《CUDA C++ Programming Guide》](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [《CUDA C 编程权威指南》](https://mp.weixin.qq.com/s/xJY5Znv3cuQi_UCd_XjJ4A)：[书中示例代码](https://github.com/Eddie-Wang1120/Professional-CUDA-C-Programming-Code-and-Notes)
- [Nvidia 官方 CUDA 是示例](https://github.com/NVIDIA/cuda-samples)
- [《CUDA 编程：基础与实践 by 樊哲勇》](https://book.douban.com/subject/35252459/)
	- [学习笔记](https://github.com/QINZHAOYU/CudaSteps)
	- [示例代码](https://github.com/MAhaitao999/CUDA_Programming)
- [《CUDA 编程简介: 基础与实践 by 李瑜》](http://www.frankyongtju.cn/ToSeminars/hpc.pdf)
- [《CUDA 编程入门》 - 本文改编自北京大学超算队 CUDA 教程讲义](https://hpcwiki.io/gpu/cuda/)
- [Multi GPU Programming Models](https://github.com/NVIDIA/multi-gpu-programming-models)
- [CUDA Processing Streams](https://turing.une.edu.au/~cosc330/lectures/display_lecture.php?lecture=22#1)

## 监控与运维
- [nvidia-smi 入门](ops/nvidia-smi.md)
- [nvtop 入门](ops/nvtop.md)
- [Nvidia GPU XID 故障码解析](https://mp.weixin.qq.com/s/ekCnhr3qrhjuX_-CEyx65g)
- [Nvidia GPU 卡 之 ECC 功能](https://mp.weixin.qq.com/s/nmZVOQAyfFyesm79HzjUlQ)
- [查询 GPU 卡详细参数](ops/DeviceQuery.md)
- [Understanding NVIDIA GPU Performance: Utilization vs. Saturation (2023)](https://arthurchiao.art/blog/understanding-gpu-performance/)
- [GPU Utilization is a Misleading Metric](ops/gpu_utils.md)

## 性能分析与调优
- [使用 Nsight Compute Tool 分析 CUDA 矩阵乘法程序](https://www.yuque.com/u41800946/nquqpa/eo7gykiyhg8xi2gg)
- [CUDA Kernel Profiling using Nvidia Nsight Compute](profiling/s9345-cuda-kernel-profiling-using-nvidia-nsight-compute.pdf)

# LLM 基础
## Article & Video
- [Andrej Karpathy：Deep Dive into LLMs like ChatGPT（B站视频）](https://www.bilibili.com/video/BV16cNEeXEer)
- [大模型基础组件 - Tokenizer](https://zhuanlan.zhihu.com/p/651430181)
- [深入浅出文本嵌入技术](llm/text_embeddings_guide.md)
- [深入探索：AI 驱动的 PDF 布局检测引擎源代码解析](llm/marker.zh-CN.md)
- [上海人工智能实验室开源工具 MinerU 助力复杂 PDF 高效解析提取](llm/minerU_intro.md)
- [**大模型可视化指南**](https://www.maartengrootendorst.com/)

## eBook
- [大模型基础](https://github.com/ZJU-LLMs/Foundations-of-LLMs) <br>
	<img src="https://raw.githubusercontent.com/ZJU-LLMs/Foundations-of-LLMs/main/figure/cover.png" height="300"/>
- [Hands-On Large Language Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) <br>
	<img src="https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/images/book_cover.png" height="300"/>
	
## AI Infra
- [高性能 GPU 服务器硬件拓扑与集群组网](https://arthurchiao.art/blog/gpu-advanced-notes-1-zh/)
- [NVIDIA GH200 芯片、服务器及集群组网](https://arthurchiao.art/blog/gpu-advanced-notes-4-zh/)
- [深度学习（大模型）中的精度](https://mp.weixin.qq.com/s/b08gFicrKNCfrwSlpsecmQ)

## 深度学习/机器学习
- [《机器学习系统：设计和实现》](https://openmlsys.github.io/index.html)
- [《动手学深度学习》](https://zh.d2l.ai/)

## 动手实践
- [读者来信：请问7b阅读分析不同中医古籍的能力怎么样？可以进行专项训练大幅度提高这方面能力么？](llm/traditional-chinese-medicine.md)
- [ollama benchmark](llm/ollama/README.md)
- [解密大语言模型中的 Tokens](llm/token/llm_token_intro.md)
	- [Tiktokenizer 在线版](https://tiktokenizer.vercel.app/?model=gpt-4o)
- [Markitdown 入门](llm/markitdown/README.md)

## DeepSeek
- [Deepseek 3FS（ Fire-Flyer File System）设计笔记](deepseek/deepseek_3fs_design_notes.zh-CN.md)
- [DeepSeek r1 蒸馏模型和满血模型对比](deepseek/deepseek-r1-cmp.md)
- [在 Mac 上运行 DeepSeek-R1 模型](deepseek/mac-deepseek-r1.md)

## Useful Projects

- [unstructured](https://github.com/Unstructured-IO/unstructured):Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.
- [MinerU](https://github.com/opendatalab/MinerU):A high-quality tool for convert PDF to Markdown and JSON.
- [markitdown](https://github.com/microsoft/markitdown): Python tool for converting files and office documents to Markdown.
- [unsloth](https://github.com/unslothai/unsloth): About
Finetune Llama 3.3, DeepSeek-R1 & Reasoning LLMs 2x faster with 70% less memory! 
- [ktransformers](https://github.com/kvcache-ai/ktransformers): A Flexible Framework for Experiencing Cutting-edge LLM Inference Optimizations

## RAG
- [**从0到1快速搭建RAG应用**](https://mp.weixin.qq.com/s/89-bwZ4aPor4ySj5U3n5zw)
	- [配套代码](llm/rag/lession2.ipynb)

## Fine-Tuning
- [**Qwen 2 大模型指令微调入门实战**](https://mp.weixin.qq.com/s/Atf61jocM3FBoGjZ_DZ1UA)
	- [配套代码](llm/fine-tuning/train_qwen2.ipynb)

# LLM 训练
## 从零开始训练 70B 模型
* [Training a 70B model from scratch: open-source tools, evaluation datasets, and learnings](https://imbue.com/research/70b-intro/)
* [Sanitized open-source datasets for natural language and code understanding: how we evaluated our 70B model](https://imbue.com/research/70b-evals/)
* [From bare metal to a 70B model: infrastructure set-up and scripts](https://imbue.com/research/70b-infrastructure/)
* [Open-sourcing CARBS: how we used our hyperparameter optimizer to scale up to a 70B-parameter language model](https://imbue.com/research/70b-carbs/)