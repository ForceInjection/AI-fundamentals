# Stanford CS336 · Language Modeling from Scratch

> 机构：Stanford · 讲师：Tatsunori Hashimoto、Percy Liang · 最近开课：Spring 2026（spring2025 / spring2024 存档在站）
> 资源可访问性核实：2026-11-19

## 定位

讲大模型本身：不把 LLM 当黑盒，从 tokenizer 一路造到 alignment，补全「模型是怎么做出来的」这块基础。仓库里 KV Cache、推理系统、Agent 方向的文章都默认读者懂 Transformer 和训练流程，缺的这一块由 CS336 补。

## 课程主线

五个作业对应五段主线，全部从零实现：

1. **basics**：tokenizer（BPE）、Transformer 前向反向、优化器，纯 PyTorch/NumPy 手写；
2. **systems**：CUDA/Triton kernel（flash attention、环形 all-reduce）、分布式训练（数据/张量/流水线并行）、显存与通信量账本；
3. **scaling laws**：学习率/批量/参数量的 scaling 实验，Chinchilla 口径的算力分配；
4. **data**：数据采集、清洗、去重、过滤，DataComp 式的 pipeline；
5. **alignment**：SFT、RLHF、DPO，安全评估。

## 资源地图

| 资源                 | 位置                                                                                                                               | 公开情况                 |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| 课程主页（各学期版） | <https://cs336.stanford.edu/>                                                                                                      | 完全公开                 |
| 讲课录像             | YouTube playlist（主页入口）                                                                                                       | 2024 版起全公开          |
| 作业                 | <https://github.com/stanford-cs336/>（assignment1-basics 至 assignment5-alignment，含 PDF 说明）                                   | 全公开                   |
| 讲义                 | 部分 PDF 在 github.com/stanford-cs336/lectures；主体是站内交互式 Python 讲义（lecture_01 至 lecture_17，`/lectures/?trace=` 路径） | 全公开，可直接在浏览器跑 |
| 自学 GPU 指南        | 主页附录：Modal / Lambda / RunPod 等租用方案                                                                                       | 公开                     |

## 学习建议

作业 2（systems）与本仓库 `02_gpu_programming`、`09_inference_system` 的 CUDA/kernel 方向直接衔接；作业 3-5 对应 `06_llm_theory_and_fundamentals` 的量化/MoE/训练话题。交互式讲义是这门课的特色，PDF 不完整——按网页版学，别只找 slides。作业工作量大（官方建议大量 GPU 时数），自学可用 GPU 租用指南里的按小时方案。
