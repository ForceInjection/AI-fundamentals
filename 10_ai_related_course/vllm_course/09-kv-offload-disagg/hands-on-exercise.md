# 第 9 讲课堂实验

三个实验：实验 1 CPU offload 生效观察（核心，单卡）；实验 2 双路径对照；实验 3 PD 分离最小复现（可选，双机；无双机走纸面）。总时长约 25 分钟。

## 准备（课前完成）

```bash
cd ~/Project/ai-infra/vLLM && git checkout 1acec5c009
cp ../02-scheduler/code/mixed_load_probe.py . && cp ../03-kv-cache-evolution/code/prefix_cache_probe.py .
```

## 实验 1:CPU offload 生效观察（15 min，单卡）

目的：确认 offload 生效、确认命中来源被标记为 host、观察对 TTFT 的影响。

1. 开 offload 起服务：

```bash
VLLM_LOGGING_LEVEL=INFO vllm serve Qwen/Qwen3-1.7B --enforce-eager \
  --kv-offloading-size 8 2>&1 | tee /tmp/vllm-offload.log
```

2. 日志确认（三处）：`Creating offloading spec with name: CPUOffloadingSpec`（`factory.py:57`）;offloading 相关 metrics 注册；connector 角色（role=kV_producer/kv_both 类）。

3. 跑 `prefix_cache_probe.py`（同前缀命中探针，会命中 GPU 前缀缓存）与一轮长 prompt 负载（第 2 讲探针的 long prompt）；从 `/metrics` 抓 `prompt_tokens_cached_by_source`——确认出现 `source="host"` 行（且 device/host 合计守恒）。

4. 对照：去掉 `--kv-offloading-size` 重启，host 行消失；显存水位对比（`# GPU blocks` 数量随 8GB 让渡而变化）。

**产出**：生效日志行 + host 命中指标截图 + 显存对照。

## 实验 2：双路径对照（10 min，单卡）

目的：分清两条平行实现，并体会「metrics 两套」的代价。

1. 默认路径(native → OffloadingConnector)跑实验 1，记录 metrics 名（`vllm:kv_offload_*`）与日志关键词(CPUOffloadingSpec)。
2. 切 simple 路径重启：

```bash
VLLM_USE_SIMPLE_KV_OFFLOAD=1 vllm serve Qwen/Qwen3-1.7B --enforce-eager \
  --kv-offloading-size 8 2>&1 | tee /tmp/vllm-simple.log
```

3. 对照：启动日志的 connector 名(SimpleCPUOffloadConnector)与 spec 行差异；`/metrics` 里换成 `vllm:simple_kv_offload_*`(save_outcomes/load_blocks/used_blocks)；加 `--kv-cache-dtype` 无关，但可试 disk 模式特性（hugepage/num_cpu_blocks 的 extra_config）。

**产出**：两路径对照表（connector 名/metrics 名/日志关键词/特性差异）。

## 实验 3:PD 分离最小复现（10 min，可选；无双机走纸面）

目的：按 `disagg_prefill.md` 示例理解 kv_role 配对。

1. 双机（或同机双端口）各起一个实例，P 端与 D 端共用 NixlConnector:

```bash
# P 实例
vllm serve Qwen/Qwen3-1.7B --port 8001 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
# D 实例(示例形态;真实 PD 分离按文档拆 prefill/decode 实例与请求路由)
```

2. 向 D 实例发请求，观察两实例日志的 connector 角色/传输统计；
3. 无双机替代：读 `disagg_prefill.md:24-50` 示例命令，逐参数解释(kv_role/backends)；再读 `mooncake_store_connector_usage.md:93` 找到 XpYd 原文并摘录。

**产出**：双实例日志摘录（或纸面参数解释 + XpYd 出处摘录）。

## 常见问题

- `--kv-offloading-size` 后服务变慢：CPU 层容量不足触发 keys_to_retry（`cpu/manager.py:270`）；加大容量或缩短 prompt 再试。
- 找不到 offload 日志行：确认 `--kv-offloading-size` 有值（设了才启用，`cache.py:402`），且没被 `VLLM_USE_SIMPLE_KV_OFFLOAD` 换到 simple 路径（两者的日志关键词不同）。
- `/metrics` 无 host 行：确认请求确实长到能被 offload（`offload_prompt_only` 默认只 offload prefill 块，`base.py:827`）。
