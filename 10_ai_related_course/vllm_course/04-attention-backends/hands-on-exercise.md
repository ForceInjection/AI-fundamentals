# 第 4 讲课堂实验

三个实验：实验 1 强切对照（核心，需 GPU）；实验 2 决策树复算（纸面 + 日志对答案）；实验 3 可选。总时长约 30 分钟。

## 准备（课前完成）

```bash
cd ~/Project/ai-infra/vLLM && git checkout 1acec5c009
nvidia-smi --query-gpu=compute_cap --format=csv   # 记下你的算力代际(如 8.9)
```

## 实验 1：`--attention-backend` 强切对照（15 min，需 GPU）

目的：亲历「合法强切看吞吐、非法强切读理由」。

1. 起默认服务，记录自动选择的后端：

   ```bash
   vllm serve Qwen/Qwen3-1.7B --enforce-eager 2>&1 | tee /tmp/vllm-attn.log
   # 启动日志里找:平台优先级列表与最终选择(搜 attention / backend 关键词)
   ```

2. 合法强切（对照吞吐）：重启时加 `--attention-backend TRITON_ATTN`，跑第 2 讲的 `mixed_load_probe.py`，对比默认（TRITON_ATTN 机器上即 FLASH_ATTN/FLASHINFER）的吞吐差。
3. 非法强切之「名字不存在」:

   ```bash
   vllm serve Qwen/Qwen3-1.7B --attention-backend FLASH_ATTN_V99
   # 预期:Unknown attention backend ... Valid options are: ...(registry.py:18-31)
   ```

4. 非法强切之「配置冲突」：先起一个 Qwen3 服务确认默认后端，再用 MLA 模型（如 DeepSeek 蒸馏系，若显存允许）强切标准后端：

   ```bash
   vllm serve <某 MLA 模型> --attention-backend FLASH_ATTN
   # 预期:Selected backend ... is not valid for this configuration. Reason: ...(cuda.py:472-488)
   ```

**产出**：默认选择记录；吞吐对照数字；两张报错截图（各含理由列表）。

## 实验 2：决策树复算（10 min，纸面 + 日志对答案）

目的：验证你能手推五层决策树。

1. 纸面推演（不看日志）：你的机器（算力代际 **）+ Qwen3-1.7B（GQA，非 MLA，无滑窗）→ 走非 MLA 分支第 ** 段 → 抄优先级列表：** → 逐项校验，第一个通过者：**。
2. 起服务对答案：日志中的实际选择与你推的是否一致?
3. 换假想硬件再推两次：B200（SM100，因果 GQA）首选应为 FLASHINFER；假想 SM120 非因果混合批——注意两个 composite 分别要求 major==9/10,SM120 两者都不满足（预期答案：无 composite 命中，走优先级列表尾部）

**产出**：复算过程照 + 与日志的对照结论（一致/不一致及原因）。

## 实验 3:Gemma-2 滑窗观察（5 min，可选，显存许可）

1. 起 `google/gemma-2-2b-it`，日志确认后端（应为 FLASH_ATTN 或 FLASHINFER——两者都支持滑窗，`flash_attn.py:483-484` / `flashinfer.py:761-762`）;
2. 强切一个不支持滑窗的后端，观察报错理由中的 sliding window 项（`backend.py:350-351`）。

**产出**：滑窗拒绝理由截图（若触发）。

## 常见问题

- 默认后端日志找不到选择过程：自动选择无冲突时不打印理由列表，只打最终结果；想看理由就用非法强切逼出来。
- TRITON_ATTN 吞吐明显低：正常，Triton 是兜底实现；实验目的是看机制差异，不是评优。
- 强切后服务起不来：读报错理由列表——这正是本讲教的技能。
