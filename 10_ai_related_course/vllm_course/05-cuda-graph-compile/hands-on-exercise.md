# 第 5 讲课堂实验

三个实验：实验 1 三态对照（核心，需 GPU，复用第 2 讲探针）；实验 2 capture 日志解析（脚本在 `code/`）；实验 3 调试二分演练。总时长约 25 分钟。

## 准备（课前完成）

```bash
cd ~/Project/ai-infra/vLLM && git checkout 2f5d8a2b1
cp ../02-scheduler/code/mixed_load_probe.py .   # 复用第 2 讲探针
```

## 实验 1：三态 TPOT 对照（15 min，需 GPU）

目的：eager / 默认 -O2 / -O0 三态的 TPOT 差异，并解释每种差异来自哪层机制。

三态依次起服务（每次起完跑一轮 `python3 mixed_load_probe.py`，记录 TPOT 均值；同时记录**启动耗时**）：

```bash
vllm serve Qwen/Qwen3-1.7B --enforce-eager                 # 态 1:全关
vllm serve Qwen/Qwen3-1.7B -O0                             # 态 2:-O0(无编译无图)
vllm serve Qwen/Qwen3-1.7B                                 # 态 3:默认 -O2(FULL_AND_PIECEWISE)
```

观察与解释（写进产出）：

1. 态 3 与态 1/2 的 TPOT 差：来自 CUDA Graph（消 launch）+ 编译融合（第 5 讲机制）；
2. 态 2 与态 1 几乎一样：两层全关，`-O0` 未显式给 mode 时同样落到 NONE（`vllm.py:1937-1942`）。若数字有差，来自 enforce-eager 额外关掉的 JIT kernel warmup（`vllm.py:1896`），是启动期差异，不影响稳态 TPOT；
3. 启动耗时：态 3 明显更久（capture 5~20 秒 + 编译）——生产上这笔启动开销换来稳态收益。

**产出**：三态 TPOT 对照柱状图（或表）+ 三条解释，每条注明机制层（图/编译/启动）。

参考实测（2026-11-25，基线 `2f5d8a2b1`，H100 80GB，Qwen3-0.6B，随机负载 200/32 × 12 并发，单次采样）：态 1 TPOT 17.16ms / 态 2 16.76ms / 态 3 2.35ms——态 3 约快 7×，启动开销 capture 5 秒。量级供参照，非基准结论。

## 实验 2:capture 日志解析(10 min)

目的：从 serve 日志提取 capture 事实，验证两族尺寸网格的存在。

1. 默认配置起服务（`-O2`，不加 enforce-eager），日志 tee 到文件：

   ```bash
   vllm serve Qwen/Qwen3-1.7B 2>&1 | tee /tmp/vllm-capture.log
   ```

2. 跑解析脚本：

   ```bash
   python3 code/capture_log_parser.py /tmp/vllm-capture.log
   ```

3. 记录：capture 总耗时、占显存 GiB、图数量；`--max-num-seqs 64` 重启再跑，对照 decode 族图数量的变化（网格随上限变化）。

**产出**：两次解析输出 + 一句对照结论。

## 实验 3：调试二分演练（10 min，可选）

目的：演练「二分关旗标」定位流程。

1. 假设故障：「-O2 下某个 batch shape 触发报错」。按二分表逐级关：`-cc.cudagraph_mode=NONE`（若消失 → 图的锅）→ 不消失再 `-cc.mode=0`（若消失 → compile 的锅）→ 不消失 → `-cc.backend=eager`（若消失 → Inductor 的锅）。
2. 每一步记录：旗标、现象、结论。
3. 进阶（可选）：`TORCH_TRACE=/tmp/trace` 复现一次编译，`tlparse` 打开 HTML，找 attention custom op 的切分位置。

**产出**：二分过程表（旗标 × 现象 × 结论）。

## 常见问题

- 小模型上三态差异可以很大（参考实测约 7×）：0.6B/1.7B 的 kernel 快而多，launch 占比高于大模型，CUDA Graph 消除的正是这一层。若你的三态数字接近，先确认态 3 确实走进了图（capture 行）；确认走进图后差异仍小，换 `--max-model-len` 更长的负载放大 decode 占比再看。
- `-O2` 首次启动很慢：编译缓存冷启动，第二次起会快（`~/.cache/vllm`）；实验前预热一次可排除干扰。
- 实验 2 日志里没有 capture 行：确认没加 `--enforce-eager`，且模型不是自动置 eager 的 ROCm 特例（`model.py:1640-1647`）。
