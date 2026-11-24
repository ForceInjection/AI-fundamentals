# 第 10 讲课堂实验

三个实验：实验 1 运维面巡检（核心，纯 CPU 可做大部分）；实验 2 基线压测（需 GPU）；实验 3 故障演练（可选，需 GPU 与容错心态）。总时长约 25 分钟。

## 准备（课前完成）

```bash
cd ~/Project/ai-infra/vLLM && git checkout 1acec5c009
```

## 实验 1：运维面巡检（10 min，大部分纯 CPU）

目的：把运维端点与指标族走一遍，形成「巡检清单」。

1. 起服务（带认证与统计开关）:

   ```bash
   VLLM_LOGGING_LEVEL=INFO vllm serve Qwen/Qwen3-1.7B --enforce-eager \
     --api-key test123 --enable-server-load-tracking 2>&1 | tee /tmp/vllm-ops.log
   ```

2. 端点巡检（每个 curl 记录返回与用途）:

   ```bash
   curl -s -H "Authorization: Bearer test123" localhost:8000/health -o /dev/null -w "%{http_code}\n"
   curl -s -H "Authorization: Bearer test123" localhost:8000/version
   curl -s -H "Authorization: Bearer test123" localhost:8000/load
   curl -s -H "Authorization: Bearer test123" localhost:8000/metrics | grep -c '^vllm'
   ```

3. 指标族盘点：`/metrics` 输出按前缀归类(counted)：引擎类（`num_requests_running/waiting`、`kv_cache_usage_perc`）、缓存类（`prefix_cache_*`、`cached_by_source` 的 source 标签）、延迟类（TTFT/TPOT/ITL 直方图）、子 logger（spec_decode——若配了；kv_connector）。
4. 认证验证：不带 Authorization 发 /health，观察 401（对照 `AuthenticationMiddleware`）。
5. 陷阱验证：`curl localhost:8000/ping` → 404（与讲义一致：不是通用端点）。

**产出**：巡检清单（端点/状态码/用途）+ 指标族计数。

## 实验 2：基线压测（15 min，需 GPU）

目的：跑通「压测 → 读数」闭环，取上线基线。

1. 起服务（默认 -O2，生产姿态）:

   ```bash
   vllm serve Qwen/Qwen3-1.7B 2>&1 | tee /tmp/vllm-bench.log
   ```

2. 压测（另开终端）:

   ```bash
   vllm bench serve --backend openai-chat --model Qwen/Qwen3-1.7B \
     --dataset-name random --random-input-len 512 --random-output-len 128 \
     --num-prompts 200 --percentile-metrics ttft,tpot,itl,e2el 2>&1 | tee /tmp/bench.log
   ```

3. 记录：request/output throughput、TTFT/TPOT/E2EL 均值与 P99;
4. 对照 `/metrics`：压测期间的 `num_requests_running` 峰值、`kv_cache_usage_perc` 峰值、TTFT 直方图分布；
5. 思考题：bench 的 TPOT 与 metrics 的 TPOT 直方图为什么不完全一致?（提示：ITL vs TPOT 口径 + 排除 ≤1 token 请求）

**产出**：基线百分位表 + metrics 峰值记录 + 思考题答案。

## 实验 3：故障演练（10 min，可选，需 GPU）

目的：观察 worker 死亡 → executor shutdown → /health 503 的完整链路。

1. 起服务（tp=1，记录 /health 200）:
2. 另开终端找到 worker 进程（`ps -ef | grep` 按 pid），`kill -9` 它；
3. 时间线记录：kill 时刻 → worker monitor 报错日志（`start_worker_monitor` 触发）→ executor shutdown → /health 变 503 的时刻；
4. 解释：为什么 K8s 部署下这个「死亡」是正确行为（对照第 1 讲的三堵墙之「故障边界」与本讲 FT 协议的分工）。

**产出**：故障时间线（带时刻）+ 分工解释。

## 常见问题

- /metrics 无 vllm 前缀指标：确认带认证头（部分中间件拦在前面）且服务版本对齐基线。
- bench 报 dataset 错：random 数据集无需下载；其他数据集（sharegpt 等）需本地文件。
- 实验 3 后服务没恢复预期：正确——worker 死亡的语义就是整组失效，K8s 的 liveness（读 /health 503）会重拉；手动重启即恢复。
