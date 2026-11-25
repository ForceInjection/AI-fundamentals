# 第 7 讲课堂实验

三个实验：实验 1 ngram 开关对照（核心，需 GPU，零额外模型）；实验 2 三口径读数（在实验 1 的日志上做）；实验 3 动态 SD（可选）。总时长约 25 分钟。

## 准备（课前完成）

```bash
cd ~/Project/ai-infra/vLLM && git checkout 2f5d8a2b1
cp ../02-scheduler/code/mixed_load_probe.py .   # 复用探针
```

## 实验 1:ngram 开关对照（15 min，需 GPU）

目的：亲眼看投机解码的加速与 accept length，并验证「重复性负载收益高」。

1. 基线服务（无投机）：

   ```bash
   vllm serve Qwen/Qwen3-1.7B --enforce-eager 2>&1 | tee /tmp/vllm-nospec.log
   ```

   跑一轮 `python3 mixed_load_probe.py`，记 TPOT 均值。

2. 开 ngram 重启：

   ```bash
   vllm serve Qwen/Qwen3-1.7B --enforce-eager \
     --speculative-config '{"method":"ngram","num_speculative_tokens":5}' \
     2>&1 | tee /tmp/vllm-ngram.log
   ```

   再跑一轮探针；日志里找 `SpecDecoding metrics` 行（走 logger.info，默认级别即可见），记 Mean acceptance length。

3. 换高重复负载：探针的短 prompt 改成同一前缀（或直接用第 3 讲的 `prefix_cache_probe.py`——它的 system 段是高度模板化的文本，天然高重复），再跑一轮，对照 accept length 与 TPOT。

**产出**：开/关两组 TPOT + 各自 accept length + 一句结论（ngram 的收益与负载重复度的关系）。

## 实验 2：三口径读数（10 min，零额外成本）

目的：分清 acceptance 的三种口径，避免以后混用。

1. 在实验 1 的 ngram 日志（`/tmp/vllm-ngram.log`）中找：
   - `SpecDecoding metrics: Mean acceptance length: X.XX`（含 bonus 口径）与 `Per-position acceptance rate`；
   - `Accepted/Drafted/Verified tokens` 三个计数。
2. 手算验证：`1 + Accepted/Drafted ≈ Mean acceptance length`？偏差多少，解释来源（异步/多请求聚合）。
3. 用 `--per-request-spec-decode-metrics summary` 重启，发一个请求，从响应的 `metrics.speculative_decoding` 读 per-request 的 `mean_acceptance_length`——注意它与日志均值的口径差（不含 bonus）。

**产出**：三口径读数表（日志均值/手算/per-request）+ 口径说明一句。

## 实验 3：动态 SD 与 adaptive verification（10 min，可选，需 GPU）

1. 在 ngram 配置上叠 `--speculative-config '{"method":"ngram","num_speculative_tokens":8,"num_speculative_tokens_per_batch_size":{"1":8,"16":2,"64":0}}'`（高并发自动降 K 到 0）；
2. 分别用 1 并发与 32 并发跑探针，对照 accept length 与 TPOT：低并发大 K、高并发小 K 是否按配置生效；
3. 再叠 `--speculative-config '{"method":"ngram",...,"enable_adaptive_verification":true}'` 观察 per-position 指标变化。

**产出**：两档并发的对照数字 + 一句「动态 SD 在保护什么」。

## 常见问题

- ngram accept length 很低（接近 1）：负载重复度低或 `prompt_lookup_min/max` 不合适（默认 5/5；代码任务可试 2-8）；ngram 本就是零成本兜底，不要对闲聊负载期待收益。
- MTP 起不来：确认模型 checkpoint 带 MTP 层（`config.json` 的 `model_type` 在 MTPModelTypes 内；Qwen3-1.7B 基座不带），换 DeepSeek/Qwen3-Next 类。
- accept length 数字忽高忽低：多请求聚合的均值受混合负载影响；用 per-request 口径看单条。
