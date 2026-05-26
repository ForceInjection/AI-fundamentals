# 13. LoRA 微调

在 Ascend NPU 上对 Qwen2.5-7B-Instruct 做参数高效微调。

## 文件

| 文件 | 说明 |
|------|------|
| `01_lora_finetune.md` | LoRA 微调文档（原理、实验对比、训练结果） |
| `lora_finetune.py` | LoRA 微调脚本（CLM / SFT 双模式） |
| `gen_sft_data.py` | SFT 训练数据生成（用 7B 从文档生成 QA 对） |
| `fetch_data.py` | 训练数据获取（Wiki + 昇腾文章） |
| `sft-data-250.jsonl` | 250 条 SFT 数据集 |
| `sft-data-380.jsonl` | 380 条 SFT 数据集（最终版本） |

## 关键发现

- SFT 格式远优于 CLM：CLM 覆盖指令遵循能力，SFT 保留
- 380 QA 对接近 LoRA r=8 有效上限
- RAG + SFT 互补：RAG 提供事实，SFT 提供风格
- DDP 多卡训练可用（HCCL），HBM 16.4 GB/卡
