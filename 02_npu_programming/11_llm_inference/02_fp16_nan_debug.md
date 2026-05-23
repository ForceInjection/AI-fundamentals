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

0.5B 的 hidden_size=896，head_dim=64。更小的维度意味着：

- 激活值范围更小（RMSNorm 归一化效果更好）
- Q·K^T 的点积项更少（64 vs 128），overlap 风险更低

未经实验验证的推断：0.5B 的激活值可能稳定在数百而非数千，点积 64 × 500 × 500 = 16,000,000（仍然 > 65504，但如果激活值实际是 ~200，则 64 × 200 × 200 = 2,560,000 仍可能溢出）。实际情况可能涉及更多因素（权重初始化、每个 head 内的正负抵消等），但 FP16 的面板值限制是确定的瓶颈。

### 3.3 为什么 BF16 没问题

模型训练精度是 BF16（config 记录 `torch_dtype: bfloat16`），BF16 的指数范围与 FP32 相同（8 位指数），最大值约 3.4e38，远大于 FP16 的 65504。

当前的 NPU 栈（torch_npu 2.1.0）可能不完全支持 BF16 推理，或者 BF16 路径未正确启用。升级 CANN/torch_npu 到更新的版本可能会解决此问题。

### 3.4 为什么是第 27 层而非更早

第 4 层后值达到 2700+，但 Layre 4-26 的残差连接 + Attention + FFN 组合没有再次放大这些值——RMSNorm 和残差连接的组合将 max 稳定在 3400 左右。直到最后第 27 层的 Attention，dot product 最终溢出。

具体的溢出条件：需要 Q^T·K 的点积超过 65504。在第 27 层之前，可能由于 LayerNorm 的位置、权重矩阵的缩放特性，Q 和 K 的投影值恰好较低。第 27 层的特定权重组合与已累积的激活值交互，最终触发了溢出。

## 4. 可行的解决方案

| 方案                           | 可行性                 | 代价                                  |
| ------------------------------ | ---------------------- | ------------------------------------- |
| FP32 加载模型                  | 已验证可用             | HBM 占用 29.4GB，无法同时跑 Embedding |
| 升级 CANN + torch_npu          | 理论上支持 BF16 后可解 | 需要升级远端环境，风险未知            |
| 在关键操作前手动 scale 输入    | 需改 modeling_qwen2.py | 可能影响模型精度                      |
| 使用 0.5B 模型                 | 已验证可用             | 回答质量有限                          |
| 等待新版 transformers 自动处理 | 被动                   | —                                     |

当前最佳方案：对于 RAG 全链路场景使用 0.5B 模型；如需 7B 单独推理，可使用 FP32 加载（`--llm-model <path>` + 脚本中改为 float32）。

## 5. 总结

这是一个典型的 **FP16 数值溢出** 问题：

- **直接原因**：Qwen2.5-7B 深层激活值较大（3000+），Attention 的 Q·K^T 点积超出 FP16 最大值 65504
- **根本原因**：模型用 BF16 训练（指数范围同 FP32），但在仅支持 FP16/FP32 的旧版 NPU 栈上推理时，FP16 无法表示中间结果；NPU 栈不支持 BF16 推理
- **0.5B/7B 差异**：0.5B 模型维度小，激活值范围不触发 FP16 溢出边界

**教训**：在旧版 NPU 栈上部署 BF16 训练的大模型时，应先验证 FP16 下的数值稳定性。若激活值范围超过 ~200-300，FP16 的 dot product 可能在各处溢出。优先尝试升级 CANN/torch_npu 以启用 BF16 推理，或使用 FP32 作为 fallback。
