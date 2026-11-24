# 第 1 讲课堂实验

三个实验：实验 1 纯 CPU（必做），实验 2 需要一张 GPU（核心），实验 3 可选。总时长约 40 分钟。

## 准备（5 min，课前完成）

```bash
cd ~/Project/ai-infra/vLLM
git checkout 1acec5c009
python3 -c "import vllm; print(vllm.__version__)"   # 确认环境可用
```

## 实验 1：进程拓扑观察（10 min，纯 CPU）

目的：验证进程公式 `A + DP + N`，认出每个进程是谁。

1. 用最小配置起服务（DP=1，单卡 tp=1 时 `A=1, DP=1, N=1`，外加可选 coordinator）:

   ```bash
   vllm serve Qwen/Qwen3-0.6B --enforce-eager 2>&1 | tee /tmp/vllm-serve.log
   ```

2. 另开终端，跑本讲脚本给进程归类：

   ```bash
   python3 code/process_topology.py
   ```

3. 对照检查：
   - 每个进程的 cmdline 里，API / EngineCore / Worker 各自的辨识特征（`EngineCore` 进程 title 由 `run_engine_core` 设置；`MultiprocExecutor` 拉起的 worker 进程特征）;
   - `EngineZmqAddresses` 相关日志里的 IPC 端点(inputs / outputs);
   - 改配置再起一次：`--tensor-parallel-size 2`（需 2 卡）或 `--data-parallel-size 2`，数进程，验证公式。

**产出**：一张进程列表截图 + 每个进程一行归类说明（公式项 × 实际 pid）。

## 实验 2：一次请求的全链路日志追踪（15 min，需 GPU）

目的：把 README「七环节」映射到真实日志行。

1. 开调试日志重启服务：

   ```bash
   VLLM_LOGGING_LEVEL=DEBUG vllm serve Qwen/Qwen3-0.6B --enforce-eager 2>&1 | tee /tmp/vllm-debug.log
   ```

2. 发一个请求并计时：

   ```bash
   curl http://localhost:8000/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"用一句话介绍 vLLM"}],"max_tokens":32}' \
   | python3 -m json.tool
   ```

3. 在 `/tmp/vllm-debug.log` 里找并标注（行号随版本浮动，按关键词搜）:
   - 引擎启动：`init engine` 阶段、KV cache 分配（`# GPU blocks` 相关行，对应 `_initialize_kv_caches`）;
   - 请求进入：`Received request`（EngineCore 收到 add_request）;
   - 调度与执行：scheduler 输出摘要、每步日志；
   - 输出回流与 finish 原因。

4. 把每条日志对应回七个环节之一，写进标注表。

**产出**:「日志行 ↔ 环节号」标注表（≥7 行，每环节至少一条）。

## 实验 3:eager 对照（10 min，可选，需 GPU）

1. 保持实验 2 服务，重跑同一 curl 五次，记录平均耗时（或用 `time` 计 curl）。
2. `--enforce-eager` 已开（实验 1/2 均带此开关）。去掉它重启，再跑五次。
3. 记录两组平均首 token 延迟差；这正是第 5 讲 CUDA Graph 要解释的现象，先拿到体感数据。

**产出**：两组延迟数字 + 一句观察。

## 常见问题

- 单卡机器跑 `--tensor-parallel-size 2` 会报错：属预期，公式验证改用 `--data-parallel-size 2`（纯 CPU 也能起多个 EngineCore 进程，模型可换 `--runner generate` 的小配置；若本机不支持 CPU 后端，以实验 1 的单配置进程观察 + 读代码替代）。
- DEBUG 日志过大：加 `| grep -E 'EngineCore|Received request|Added request'` 过滤后再标注。
