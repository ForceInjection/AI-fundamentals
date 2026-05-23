# 11. LLM 推理 on NPU

在 NPU 上部署 Qwen2.5（默认 7B-Instruct, BF16），实现全链路本地化 RAG。0.5B 也可通过 `--llm-model` 切换。

## 文件

| 文件                         | 说明                                                  |
| ---------------------------- | ----------------------------------------------------- |
| `01_llm_inference_on_npu.md` | LLM 推理部署文档（自回归原理、ChatML、采样策略）      |
| `02_fp16_nan_debug.md`       | Qwen2.5-7B FP16 NaN 诊断报告（含 BF16 解决方案）      |
| `03_lora_finetune.md`        | LoRA 微调文档（原理、配置、训练结果）                 |
| `llm_inference.py`           | LLM 推理脚本（infer / chat / benchmark，含 NPU 检查） |
| `lora_finetune.py`           | LoRA 微调脚本（peft + BF16，HBM 峰值 16.4 GB）        |

## 关键发现

- Qwen2.5-7B-Instruct 在 BF16 下推理正常（HBM ~15 GB），与 FP16 内存相同但数值稳定
- FP16 下 NaN 根因：深层激活值 + 大 head_dim（128）→ Q·K^T 点积超 FP16 最大值（65504）
- 0.5B 不溢出的原因：head_dim 更小（64）+ 激活值更低（~1665 vs ~3400），溢出风险差 ~144×
- BF16 的 8 位指数（同 FP32）完美解决，当前 CANN 8.0.1 + torch_npu 2.1.0 栈已原生支持
- `--local` 模式已与 RAG pipeline 集成，全链路（embedding + FAISS + LLM）均在 NPU 上运行
- LoRA 微调在 NPU 上完全可行：0.07% 可训练参数，梯度检查点优化后 HBM 峰值 16.4 GB，2 epoch loss 12.08 → 5.41
