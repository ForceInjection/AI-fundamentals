# 诊断报告：Qwen2.5-7B FP16 在 NPU 上输出 NaN

## 1. 现象

Qwen2.5-7B-Instruct 以 `torch_dtype=torch.float16` 加载到 Ascend 910B3 上推理时，输出为乱码——反复生成 `![](https://...)` 等 Markdown 图片 URL 和特殊字符。0.5B-Instruct 在同样配置下完全正常。

环境：CANN 8.0.1, torch_npu 2.1.0.post13, PyTorch 2.1.0, transformers 4.38.2。

## 2. 排查过程

### 2.1 排除模型来源问题

从 HuggingFace 和 ModelScope 分别下载了 Qwen2.5-7B-Instruct，对比测试：

| 来源            | 文件大小 | FP16 推理结果    |
| --------------- | -------- | ---------------- |
| HuggingFace     | 15GB     | 乱码 URL         |
| ModelScope      | 15GB     | 乱码 URL（相同） |
| ModelScope 0.5B | 1GB      | 正常 "1+1等于2"  |

两者文件大小完全一致（4 个 safetensors 分片各 3.55-3.95 GB），表现也一致。排除模型来源问题。

### 2.2 排除采样策略问题

贪心解码（`do_sample=False`）与随机采样结果一致，均输出乱码。排除采样策略问题。

### 2.3 定位 NaN

直接检查模型 logits 输出：

```python
outputs = model(**inputs)
logits = outputs.logits[0, -1]  # 最后一个位置的 logits
print(f"min={logits.min()}, max={logits.max()}, mean={logits.mean()}")
# 输出: min=nan, max=nan, mean=nan
```

logits 全部为 NaN。NaN（Not a Number）会导致 `argmax` 返回 token 0，`top-k` 返回前 k 个 token，所以输出是连续的 ASCII 序 token（`!`, `"`, `#`, ...），被 tokenizer decode 后看起来像乱码 URL。

### 2.4 对比 FP32

用 `torch_dtype=torch.float32` 加载 7B 模型，推理结果正常：

```text
FP32 ANSWER: '1+1等于2。'
HBM used: 29.4 GB
```

确认问题仅在 FP16 下出现。

### 2.5 追踪 NaN 产生位置

用 PyTorch hook 监控每层输出，定位 NaN 首次出现位置：

```text
NaN first appeared in: Layer27_attn  （共 28 层，索引 0-27）
```

### 2.6 分析各层激活值

```python
outputs = model(**inputs, output_hidden_states=True)
for i, hs in enumerate(outputs.hidden_states):
    print(f"Layer {i:2d}: max={hs.max():.2f}")
```

```text
Layer  0: max=    0.05     ← Embedding 输出，正常
Layer  1: max=    3.26     ← 第 1 层，值很小
Layer  2: max=    4.34
Layer  3: max=    5.77
Layer  4: max= 2714.00     ← 突然放大 ~500 倍
Layer  5: max= 3342.00
Layer  6: max= 3360.00
...
Layer 26: max= 3436.00     ← 此后稳定在 3000-3500
Layer 27: max=  940.00     ← LayerNorm 后缩小
Layer 28: max=     nan     ← 最终输出 NaN
```

## 3. 根因分析

### 3.1 数值溢出路径

第 4 层开始，hidden states 的 max 值突然放大到 2700+，并在此后 23 层中持续维持在 3000-3500 范围。虽然这些值本身在 FP16 范围内（max 65504），但问题出在后续 Attention 的 `Q @ K^T` 计算：

```text
Q, K 中每个元素 ≈ ±500（经 LayerNorm + 线性投影后）
head_dim = 128
dot_product = Σ(q_i × k_i) ≈ 128 × 500 × 500 ≈ 32,000,000
```

**32,000,000 >> 65,504（FP16 max）** → overflow → `inf`

`softmax([inf, inf, ...])` → `inf - inf → NaN`

### 3.2 为什么 0.5B 不溢出

用同样方法测量 0.5B 的 FP16 激活值（已验证全链路正常）：

```text
Layer  0: max=    0.05
Layer  1: max=    2.08
Layer  2: max=    3.03
Layer  3: max=  778.00     ← 第 3 层开始放大
Layer  4: max= 1632.00     ← 第 4 层起稳定
Layer  5: max= 1634.00
...
Layer 21: max= 1665.00     ← 深层稳定在 ~1600-1670
Layer 22: max=   44.00     ← RMSNorm 归一化后
Layer 24: max=   65.38     ← lm_head 输出
Logits: min=-15.80, max=13.55, nan=False  ← 全部正常
```

对比两个模型：

| 指标               | 0.5B                | 7B                     |
| ------------------ | ------------------- | ---------------------- |
| hidden_size        | 896                 | 3584                   |
| head_dim           | 64                  | 128                    |
| 深层激活值 max     | ~1665（Layer 4-21） | ~3400（Layer 4-26）    |
| 第一次跳跃层       | Layer 3（0.05→778） | Layer 4（0.05→2714）   |
| Q/K 投影值典型范围 | ±10~20（估算）      | ±170（实测）           |
| Q·K^T 估算         | 64 × 20² ≈ 25,600   | 128 × 170² ≈ 3,700,000 |

0.5B 不溢出的原因有两个叠加因素：

1. **head_dim 更小**：64 vs 128，点积项数减半。即使 Q/K 值相同，点积结果也减半
2. **激活值范围更小**：深层 max 约 1665 vs 3400。经 RMSNorm 归一化后，Q/K 投影值的典型范围在 ±10-20（0.5B）vs ±170（7B）。这源于权重矩阵的缩放特性差异——更大的模型倾向于在更深层产生更大范围的特征值

两者的组合效应：`(128/64) × (170/20)² ≈ 2 × 72 ≈ 144×` 的 overflow risk 差异。0.5B 的 25,600 安全落在 FP16 范围内（< 65,504），而 7B 的 3,700,000 远超出。

### 3.3 为什么 BF16 没问题

模型训练精度是 BF16（config 记录 `torch_dtype: bfloat16`），BF16 的指数范围与 FP32 相同（8 位指数），最大值约 3.4e38，远大于 FP16 的 65504。

经实测验证，当前 NPU 栈（torch_npu 2.1.0.post13 + CANN 8.0.1）已完全支持 BF16 推理：BF16 tensor 创建、matmul、scaled_dot_product_attention 均正常。以 `torch_dtype=torch.bfloat16` 加载 7B 模型，推理结果正确（`1+1等于2`），HBM 占用仅 14.7 GB，与 FP16 相同。

BF16 与 FP32 的对比：

| 指标                 | BF16                  | FP32                    |
| -------------------- | --------------------- | ----------------------- |
| HBM 占用             | 14.7 GB               | 29.4 GB                 |
| 数值稳定性           | 正确                  | 正确                    |
| 能否同时跑 Embedding | 可以（~16 GB 总占用） | 不可以（~31 GB 总占用） |

这是最理想的方案——HBM 效率和 FP16 相当，数值稳定性和 FP32 相当。根本原因在于 LLM 推理对**动态范围**（需要指数位宽）的需求远大于对**尾数精度**的需求，而 BF16 恰好把精度位分配给了指数。

### 3.4 为什么是第 27 层而非更早

第 4 层后值达到 2700+，但 Layre 4-26 的残差连接 + Attention + FFN 组合没有再次放大这些值——RMSNorm 和残差连接的组合将 max 稳定在 3400 左右。直到最后第 27 层的 Attention，dot product 最终溢出。

具体的溢出条件：需要 Q^T·K 的点积超过 65504。在第 27 层之前，可能由于 LayerNorm 的位置、权重矩阵的缩放特性，Q 和 K 的投影值恰好较低。第 27 层的特定权重组合与已累积的激活值交互，最终触发了溢出。

## 4. 可行的解决方案

| 方案                        | 可行性                 | 代价                                              |
| --------------------------- | ---------------------- | ------------------------------------------------- |
| **BF16 加载模型**           | 已验证可用             | 无。HBM 14.7 GB，与 FP16 相同，可同时跑 Embedding |
| FP32 加载模型               | 已验证可用             | HBM 占用 29.4 GB，无法同时跑 Embedding            |
| 在关键操作前手动 scale 输入 | 需改 modeling_qwen2.py | 可能影响模型精度                                  |
| 使用 0.5B 模型              | 已验证可用             | 回答质量有限                                      |

**推荐方案**：将 `torch_dtype=torch.float16` 改为 `torch_dtype=torch.bfloat16`。无需升级 CANN 或 torch_npu——当前栈已完全支持 BF16。7B BF16（14.7 GB）+ BGE embedding（~1 GB）≈ 16 GB，64 GB HBM 完全够用。

## 5. 总结

这是一个典型的 **FP16 数值溢出** 问题：

- **直接原因**：Qwen2.5-7B 深层激活值较大（3000+），Attention 的 Q·K^T 点积超出 FP16 最大值 65504
- **根本原因**：模型用 BF16 训练，FP16 的 5 位指数无法表示中间结果；而 BF16 的 8 位指数（与 FP32 相同）可以安全容纳
- **0.5B/7B 差异**：0.5B 的 head_dim 更小（64 vs 128）且激活值范围更低（~1665 vs ~3400），叠加效应约 144× 溢出风险差异
- **解决方案**：以 `torch_dtype=torch.bfloat16` 加载 7B 模型，HBM 占用 14.7 GB（与 FP16 相同），数值稳定性等同 FP32，当前 CANN 8.0.1 + torch_npu 2.1.0 栈已完全支持，无需升级

**教训**：部署大模型时，若权重为 BF16 格式，优先使用 BF16 推理而非转换为 FP16。LLM 推理的创新计算特性（大范围中间值、长距离 attention scores）对动态范围的需求远大于尾数精度，BF16 是比 FP16 更合适的选择。
