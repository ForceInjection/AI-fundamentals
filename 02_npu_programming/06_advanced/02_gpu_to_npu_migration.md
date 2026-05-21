# GPU 到 NPU 的迁移策略

## 1. 迁移决策树

```text
需要迁移到 NPU 吗?
│
├─ 使用标准 PyTorch 模型 (ResNet/BERT/ViT/Llama)?
│   YES → 迁移成本低。替换 .cuda() → .npu()，验证精度即可
│
├─ 有自定义 CUDA kernel?
│   YES → 需评估 kernel 复杂度:
│   │
│   ├─ 简单 kernel (element-wise, custom activation)
│   │   → 用 Ascend C 重写，成本中等 (1-2 周)
│   │
│   └─ 复杂 kernel (FlashAttention, fused ops)
│       → 检查是否已有社区 NPU 适配 (如 torch_npu 的 fused operators)
│       否则成本高 (4-8 周)，考虑替代方案
│
├─ 依赖 NVIDIA 特有库 (cuDNN/cuBLAS/NCCL)?
│   → CANN 有对应替代，但 API 不同:
│   cuDNN/cuBLAS → CANN OPP (内置算子)
│   NCCL → HCCL (API 类似)
│   TensorRT → ATC
│
├─ 使用 torch.compile / Dynamo / Triton?
│   → NPU 不直接支持 Dynamo/Triton
│   Triton kernel 需改写为 Ascend C
│   torch.compile 替换为 MindSpore Graph 模式 或 CANN GE
│
└─ 训练框架是 PyTorch Lightning / HF Trainer?
    → 需确认这些框架的 Ascend 适配状态
    PyTorch Lightning: 有限支持
    HuggingFace Trainer: 通过 accelerate + torch_npu 基本可用
```

## 2. 迁移成本估算

| 场景 | 代码改动 | 时间 |
|------|---------|------|
| 标准 CV 模型训练 | 改设备字符串 (< 5 行) | < 1 天 |
| 标准 NLP 模型训练 | 改设备字符串 + 调试环境 | 1-3 天 |
| 有自定义 CUDA kernel (简单) | 重写 kernel (Ascend C) | 1-2 周 |
| 有自定义 CUDA kernel (复杂) | 重写 + 调优 | 1-2 月 |
| 完整训练 pipeline (数据+训练+部署) | 框架适配 + 工具链切换 | 2-4 周 |

## 3. 不建议迁移的场景

- 严重依赖 NVIDIA 专有生态（如 cuQuantum、CV-CUDA 的特殊功能）。
- 使用大量 Triton kernel 且无社区替代方案。
- 模型频繁变更且算子需求不稳定（研发早期）。
- 对延迟要求极高（微秒级）的推理场景（需评估 ATC 转换后的延迟）。

## 4. 参考链接

- [昇腾社区 — 迁移工具](https://www.hiascend.com/document/detail/en/canncommercial/800/devguide/migrate/migrate_0001.html)