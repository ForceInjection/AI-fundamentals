# 第 5 讲：命中、驱逐、换出

40 分钟，分三段：CPU 模拟 15 分钟，server 观测 15 分钟，强制驱逐 10 分钟。没有 GPU 时做 (a) 和 (c)，(b) 用示例日志填写。

## (a) CPU：把命中、加锁、驱逐走一遍

脚本只依赖 Python 标准库，不需要 torch 或模型权重。

```bash
cd 10_ai_related_course/kv_cache_course/05-kv-cache-reuse
python3 code/prefix_cache_sim.py --demo abcd
python3 code/prefix_cache_sim.py --demo share
python3 code/prefix_cache_sim.py --demo evict
python3 code/prefix_cache_sim.py --self-check
```

三个场景各自看什么：

- `--demo abcd`：整段建树、匹配 `AB` 并 split、对 `AB` 加锁、驱逐 `CD` 后解锁。看每一步的 `lock_ref` 和可驱逐节点。
- `--demo share`：同 prompt 换 LoRA、换 salt 的命中长度，以及 `page_size=16` 时 60 token 前缀只命中 48。
- `--demo evict`：12 token 容量下锁住 R2，再插入 R5，看谁被驱逐、谁被锁保护。

如果本机装好了 SGLang 依赖，可以对照官方单测。单测覆盖经典 `RadixCache`，锁沿路径累加的语义与 unified 实现一致：

```bash
cd ~/Project/ai-infra/sglang
python -m pytest test/registered/unit/mem_cache/test_radix_cache_unit.py -k "lock_ref or evict" -v
```

## (b) server：打开 HiCache，看三层命中

启动一个带 L2 host cache 和 L3 file backend 的实例。L3 目录用环境变量指定，便于下一轮复用。

```bash
SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/tmp/hicache-lab \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --port 30000 \
  --enable-metrics \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-write-policy write_through \
  --hicache-storage-backend file \
  --max-total-tokens 8192 \
  --page-size 1
```

参数含义：`--hicache-ratio 2` 让 host pool 是 device pool 的两倍；`write_through` 在插入时就做 D→H 备份；`--page-size 1` 让命中长度不被 page 对齐截断。把 `--page-size` 改成 16 再做一次，对比命中长度。

发三条请求：冷启动、同 prompt 重放、同 prompt 换 `cache_salt`。下面的客户端用流式读取，把第一个 chunk 的到达时间记为 TTFT。

```bash
python3 - <<'PY'
import json, time, urllib.request

URL = "http://127.0.0.1:30000/v1/chat/completions"
SYSTEM = ("You are a careful assistant. Answer briefly. " * 160)

def ask(tag, question, salt=None, extra=None):
    body = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        "max_tokens": 32,
        "stream": True,
        "return_cached_tokens_details": True,
    }
    if salt:
        body["cache_salt"] = salt
    if extra:
        body["extra_key"] = extra
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    meta, prompt_tokens = {}, 0
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if ttft is None:
                ttft = time.perf_counter() - t0
            meta = obj.get("meta_info", meta)
            if obj.get("usage"):
                prompt_tokens = obj["usage"].get("prompt_tokens", prompt_tokens)
    d = meta.get("cached_tokens_details", {})
    print(f"{tag}: prompt={prompt_tokens} device={d.get('device', 0)} "
          f"host={d.get('host', 0)} storage={d.get('storage', 0)} "
          f"ttft={ttft:.3f}s")

ask("R1 冷启动", "第一问")
ask("R2 同 prompt 重放", "第一问")
ask("R3 换 salt", "第一问", salt="tenant-2")
ask("R4 换 extra_key", "第一问", extra="lora-b")
PY
```

指标在同一个端口的 `/metrics`：

```bash
curl -s http://127.0.0.1:30000/metrics | grep -E \
  'sglang:(cache_hit_rate|hicache_backup_tokens_total|load_back_tokens_total|load_back_duration_seconds|hicache_host_used_tokens)'
```

| 指标 | 含义 | 看什么 |
| --- | --- | --- |
| `sglang:cache_hit_rate` | 最近窗口的前缀命中率 | 稳态接近 90%，冷启动为 0 |
| `sglang:hicache_backup_tokens_total` | D→H 备份 token 累计 | 增长说明写策略在工作，不代表 device 已释放 |
| `sglang:load_back_tokens_total` | H→D 回载 token 累计 | 应与 host 命中量同步增长 |
| `sglang:load_back_duration_seconds` | 回载耗时直方图 | 用 `_sum / _count` 算平均，和 TTFT 比量级 |
| `sglang:hicache_host_used_tokens` | host pool 已用 token | 接近 host 容量时会出现 host 驱逐 |

要看到 host 命中，需要先让 device 上的副本被驱逐或降级。继续用不同 prompt 把 `--max-total-tokens` 填满，再重放 R1，观察 `host` 从 0 变成非 0，`load_back_tokens_total` 随之增长。要看到 storage 命中，停掉 server，用同一个 `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` 重启，再发 R1。L3 查找按请求 token 的 hash 链进行，不依赖内存里的树是否还在。

## (c) 强制驱逐

CLI 路线：把 `--max-total-tokens` 压到 4096，重跑 (b) 的 R1 到 R4，观察 `evicted` 和命中层级的变化。

如果已经在 SGLang 的测试环境里拿到 `tree_cache` 对象，可以直接调驱逐接口：

```python
from sglang.srt.mem_cache.base_prefix_cache import EvictParams

result = tree_cache.evict(EvictParams(num_tokens=1024))
print(result.num_tokens_evicted)
```

`EvictParams` 的字段是 `num_tokens`、`swa_num_tokens`、`mamba_num`，普通 MHA 只填 `num_tokens`。

## 产出

一张统一观察表。示例行只示意填法，其余行由学员填写，至少覆盖冷启动、device 命中、host 命中、storage 命中和强制驱逐五种情况。

| 请求 | prompt tokens | device_hit | host_hit | storage_hit | 实际 prefill tokens | TTFT | load_back ms | evicted tokens | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 2200 | 0 | 0 | 0 | 2200 | 410 ms | 0 | 0 | 冷启动，建树 |
| R2 | 2200 | 2000 | 0 | 0 | 200 | 180 ms | 0 | 0 | device 命中 system prompt |
| R3 | 2200 | 0 | 2000 | 0 | 200 | 260 ms | 42 ms | 2000 | device 副本被驱逐后重放，host 命中 |
| R4 | 2200 | 0 | 0 | 2000 | 200 | 320 ms | 58 ms | 2000 | 重启后从 storage 命中 |

表后固定写三句：

TTFT 变化：……
吞吐变化：……
是否值得换出：……
