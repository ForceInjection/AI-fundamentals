# 第 2 讲课堂实验

三个实验围绕同一组负载：实验 1 建立基准现象，实验 2 拧参数看变化，实验 3 读日志找证据。实验 1-2 需要一张 GPU；实验 3 在实验 2 的日志上做，零额外成本。

## 准备（5 min，课前完成）

```bash
cd ~/Project/ai-infra/vLLM && git checkout 2f5d8a2b1
pip list | grep -E '^(vllm|openai) '   # openai 客户端随 vLLM 依赖安装
```

## 服务端配置（两次起服务，对照用）

- 配置 A（默认，chunked prefill 开）：`vllm serve Qwen/Qwen3-4B`
- 配置 B（抑制切块效果）：`vllm serve Qwen/Qwen3-4B --long-prefill-token-threshold 100000`

> threshold 设为远大于 prompt 长度的值 = 单请求单步不设上限，配合多请求并发观察调度差异。两次实验间只需重启服务，客户端脚本不变。

## 实验 1：长短交错负载的 TTFT 对照（15 min，需 GPU）

目的：复现「长 prompt 拖累短请求」以及 chunked prefill 的缓解效果。

1. 跑本讲脚本（见 `code/README.md`）：

   ```bash
   python3 code/mixed_load_probe.py --host localhost --port 8000
   ```

   脚本行为：交错发送 4 条长 prompt（约 2K token）与 4 条短 prompt（约 40 token），流式接收，记录每条请求的 TTFT 与总耗时，输出对照表。

2. 在配置 A 下跑一次，记录：短请求 TTFT 是否被长请求拉高、长请求自身 TTFT 是多少。
3. 切配置 B 重启，再跑一次，对照两组数字。

**产出**：两组 TTFT 对照表 + 一句结论（短请求 TTFT 在哪种配置下被保护）。

## 实验 2:threshold 扫描（10 min，需 GPU）

目的：感受 `--long-prefill-token-threshold` 作为「单请求单步上限」的旋钮作用。

1. 配置 B 的 threshold 分别取 `512`、`2048`、`100000` 各跑一次脚本。
2. 记录三个数字：短请求 TTFT 均值、长请求 TTFT 均值、总墙钟时间。
3. 思考题（写进产出）：threshold 越小，长请求自身 TTFT 怎么变？为什么这个旋钮不是越大越好也不是越小越好？

**产出**：三档 threshold 的三个数字 + 思考题答案。

## 实验 3：日志找证据（10 min，零额外成本）

目的：把 README 六环节映射到真实日志。

1. 用 `VLLM_LOGGING_LEVEL=DEBUG` 重启（任一配置），跑实验 1 脚本。
2. 在日志中找并标注（按关键词搜）：
   - chunked prefill 生效：同一请求分多步 prefill 的调度记录（搜 `chunk` / 多次 `Received request` 后的增量 prefill 行）；
   - 抢占事件（若触发）：搜 `preempt`（对应 `_preempt_request` 的日志与计数）；
   - cache-aware 准入（若配置了 `--cache-aware-admission-window`）：准入顺序变化；
   - async scheduling：启动日志里 `async_scheduling` 的解析结果（默认自动 True）。
3. 每条日志对应回环节号，写进标注表。

**产出**：「日志行 ↔ 环节号」标注表（至少覆盖 chunked prefill 与 async scheduling 两项）。

## 常见问题

- 脚本报 openai 连接错误：确认服务起来了（`curl localhost:8000/v1/models`）。
- 4B 模型显存不够：换 `Qwen3-1.7B`，长 prompt 长度相应减半，现象不变。
- 看不到抢占：正常——抢占需要显存压力。可把 `--gpu-memory-utilization` 压低（如 0.5）或加大 `--max-model-len` 再用长 prompt 挤；看到 `preempt` 日志即为成功。
