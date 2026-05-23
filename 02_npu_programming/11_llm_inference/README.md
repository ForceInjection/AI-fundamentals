# 11. LLM 推理 on NPU

在 NPU 上加载 Qwen2.5-0.5B-Instruct 进行本地推理，实现全链路本地化的 RAG。

## 文件

| 文件 | 说明 |
|------|------|
| `01_llm_inference_on_npu.md` | LLM 推理部署文档 |
| `02_fp16_nan_debug.md` | Qwen2.5-7B FP16 NaN 诊断报告 |
| `llm_inference.py` | LLM 推理脚本（infer / chat / benchmark） |

## 关键发现

- Qwen2.5-0.5B-Instruct 在 NPU 上正常推理（FP16），生成速度 ~18 tok/s
- Qwen2.5-7B-Instruct 在 FP16 下输出 NaN（BF16 训练的模型在仅支持 FP16 的 NPU 栈上溢出），FP32 下正常但占用 29.4 GB HBM
- `--local` 模式与 RAG pipeline 集成后，全链路（embedding + FAISS + LLM）均在 NPU 上运行
