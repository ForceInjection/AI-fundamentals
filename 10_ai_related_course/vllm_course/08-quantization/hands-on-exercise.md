# 第 8 讲课堂实验

三个实验：实验 1 LLM Compressor FP8 闭环（核心，需 GPU；压缩步骤课前完成）；实验 2 kernel 选择观察（核心，纯日志，可与实验 1 合并）；实验 3 `--kv-cache-dtype` 对照（可选）。总时长约 30 分钟（不含课前压缩）。

## 准备（课前 30-60 min）

```bash
pip install llmcompressor torch   # 压缩环境(独立 venv 更稳)
```

按 `docs/features/quantization/llm_compressor/fp8.md` 的最小示例压 Qwen3-4B（或更小模型控时长）：

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
oneshot(model="Qwen/Qwen3-4B", recipe=recipe, output_dir="./Qwen3-4B-FP8-dynamic")
```

产物目录就是实验 1 的输入（含 HF `quantization_config`）。

## 实验 1:FP8 闭环对照（15 min，需 GPU）

目的：BF16 原模型与 FP8 压缩模型的显存与吞吐对照，并读懂加载日志。

1. 起原模型记显存；起压缩产物：

```bash
vllm serve ./Qwen3-4B-FP8-dynamic --enforce-eager 2>&1 | tee /tmp/vllm-fp8.log
```

2. 在日志中找并标注：`quantization=compressed-tensors` 的方法行；kernel 选择行（`init_fp8_linear_kernel` 的结果，如 cutlass）；`# GPU blocks` 变化（KV 可用显存因权重变小而增多）。
3. 跑 `mixed_load_probe.py`（第 2 讲）一轮，记录 TPOT 对照。

**产出**：BF16 vs FP8 的显存对照、kernel 选择日志行、TPOT 数字。

## 实验 2:kernel 选择观察（10 min，纯日志）

目的：亲眼看「路由条件散布」的三个证据。

1. 用一个 GPTQ/AWQ checkpoint（社区任选；或压缩产物改 scheme）起服务，记录选中的 w4a16 kernel（应为注册表顺序的首个可用者，如 Machete/Marlin）；
2. 加 batch invariance 配置重启（`--attention-config` 相关项，或按文档），对照 kernel 是否放弃 Marlin（预期：batch-invariant 禁 Marlin，`auto_awq.py:314-321`）；
3. 强制实验：`--linear-backend <名单内的名>`，观察不兼容时的报错与回退。

**产出**：kernel 选择对照表（条件 → 选择）。

## 实验 3：`--kv-cache-dtype fp8` 对照（10 min，可选，需 GPU）

1. 同模型分别默认与 `--kv-cache-dtype fp8`(e4m3)起服务：记录 `# GPU blocks` 变化与 KV 每显存占用；
2. 跑探针看 TTFT/TPOT 是否退化（长上下文更明显）；
3. 在 DEBUG 日志里找 KV scale 相关行（对照 `kv_cache.py:189-201` 的最终写入）。

**产出**：blocks 对照 + 一句精度/速度观察。

## 常见问题

- LLM Compressor 压制 OOM：换更小模型(0.6B/1.7B)或 `--enforce-eager` 关闭目标模型干扰；压制在 CPU/单卡均可。
- 压缩产物加载报「scheme 不支持」：核对该格式的 `compressed_tensors` scheme 映射（`compressed_tensors.py:923-1080`）；换 vLLM 版本可能改变支持面。
- FP8 起服务报错指引「`fp8_per_tensor`」：这正是 `fp8.py:105-108` 的新行为——按指引改用 shorthand 即为正确操作。
