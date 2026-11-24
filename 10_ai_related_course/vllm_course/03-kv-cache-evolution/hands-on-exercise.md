# 第 3 讲课堂实验

三个实验：实验 1 是核心（同前缀命中观察，需 GPU）；实验 2 需要较大显存看分组；实验 3 可选。总时长约 25 分钟。

## 准备（课前完成）

```bash
cd ~/Project/ai-infra/vLLM && git checkout 1acec5c009
```

## 实验 1：同前缀命中的冷热对照（15 min，需 GPU）

目的：亲眼看一次前缀缓存命中——同一 system prompt，第二轮请求 TTFT 显著下降。

1. 起服务：

```bash
vllm serve Qwen/Qwen3-1.7B --enforce-eager 2>&1 | tee /tmp/vllm-kv.log
```

2. 跑本讲脚本（见 `code/README.md`）:

```bash
python3 code/prefix_cache_probe.py --host localhost --port 8000 --model Qwen/Qwen3-1.7B
```

脚本行为：构造一个约 1.5K token 的公共前缀（system 段），附加 6 个不同问题；先发 2 个「预热」请求让公共前缀入缓存，再计时发送 6 个请求，输出每条 TTFT。

3. 观察并解释：
   - 后发请求（共享前缀部分已缓存）的 TTFT 相对首个请求下降多少；
   - `--no-enable-prefix-caching` 重启再跑一轮，对照差异。
4. 在日志（`DEBUG` 级）里找命中证据：`Cached` / prefix 相关统计行；对照 README「哈希前移」一节——命中发生在 schedule 查表，不在执行期。

**产出**：冷热两轮 TTFT 对照表 + 开/关 prefix caching 的差异说明。

## 实验 2：混合注意力模型的分组日志（15 min，需较大显存或 CPU 后端）

目的：亲眼看「多组」出现——单组模型(Unitary)与混合模型(Hybrid)的启动日志差异。

1. 先跑一个单组对照（Qwen3 家族是 full attention）:

```bash
VLLM_LOGGING_LEVEL=INFO vllm serve Qwen/Qwen3-1.7B --enforce-eager 2>&1 | grep -iE 'group|blocks|KV Cache' | head -20
```

2. 换混合注意力模型（按显存任选其一；Gemma 系带 sliding window 层，Qwen3-Next 带 GDN）:

```bash
vllm serve google/gemma-2-2b-it --enforce-eager 2>&1 | grep -iE 'group|blocks|KV Cache|hybrid' | head -30
```

3. 对照两组日志，找出：KV cache groups 数量、每组的类型(sliding window / full / mamba)、`# GPU blocks` 总数；回读 `get_kv_cache_coordinator`（`kv_cache_coordinator.py:850-904`）确认走的是哪个分支。
4. 显存不足以跑 2B 模型时：改为读 `models/gemma2.py:156`（逐层二选一）与 `kv_cache_planning.py:727-789`（晋升分支），写一条「该模型在 HEAD 上会走哪个 coordinator 分支」的推断。

**产出**：单组 vs 多组的日志对照（或推断）+ 分支判断依据。

## 实验 3：`--kv-cache-dtype` 对照（10 min，可选，需 GPU）

1. 同模型分别用默认与 `--kv-cache-dtype fp8` 起服务，记录两次启动日志的 `# GPU blocks` 与每 token KV 字节。
2. 跑实验 1 脚本，对比命中率与 TTFT（注意：FP8 KV 的命中语义与精度代价，写一句观察）。

**产出**：两组 blocks 数 + 一句观察。

## 常见问题

- 实验 1 预热不明显：公共前缀长度要超过一个 block（默认 block_size 查启动日志）；或并发发送导致部分请求未吃满缓存，先串行验证。
- 日志无 prefix 相关行：确认没加 `--no-enable-prefix-caching`，且模型未触发自动关闭（见 `attention.py:431` 等自动关闭点）。
- 显存不足跑 Gemma-2-2b：用 CPU 后端（`--device cpu`，命中/分组逻辑相同，速度慢但能看日志）。
