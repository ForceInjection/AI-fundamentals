# 第 6 讲课堂实验

三个实验：实验 1 配置组合校验（核心，纯 CPU）；实验 2 启动日志走读（单卡 GPU，可选）；实验 3 双卡观察（可选）。总时长约 25 分钟。本讲实验重在「配置与日志」，不需要跑满推理负载。

## 准备（课前完成）

```bash
cd ~/Project/ai-infra/vLLM && git checkout 1acec5c009
nvidia-smi -L   # 记下本机 GPU 数(实验报错预期依赖它)
```

## 实验 1：并行配置组合校验（15 min，纯 CPU）

目的：`_validate_parallel_config`（`parallel.py:681-770`）的错误信息就是学习材料。

三个必试组合（`vllm serve <小模型或任意模型名>` 均可，报错发生在配置层，不需要真的加载模型；必要时加 `--model Qwen/Qwen3-0.6B` 走完整解析）:

1. **超卡数**：`--tensor-parallel-size 8`（单机不足 8 卡时）→ 预期：world_size 超过本机 GPU 数的报错（对照 `parallel.py:1284-1296` 的条件）;
2. **DCP 不整除**：`--decode-context-parallel-size 3 --tensor-parallel-size 4` → 预期：TP 必须被 DCP 整除的报错（`:756-758`）;
3. **EPLB 前置缺失**：`--enable-eplb`（不带 `--enable-expert-parallel`）→ 预期：EPLB 需要 enable_expert_parallel 的报错（`:723-748`）。

每条记录：完整报错信息 + 对应的校验代码行 + 一句「怎么改才对」。

**产出**：三组「报错 → 代码行 → 修正」记录。

## 实验 2：启动日志走读（10 min，单卡 GPU 或 CPU 后端）

目的：认出九种进程组在 init 日志里的出现方式；理解 dense 模型为何没有 EP 组。

1. 单卡起服务：

```bash
VLLM_LOGGING_LEVEL=INFO vllm serve Qwen/Qwen3-0.6B --enforce-eager 2>&1 | tee /tmp/vllm-par.log
```

2. 在日志里找 init 阶段的 group 分配输出（搜 `DP rank` / `TP rank` / `EP rank`，对照 `parallel_state.py:2462-2476` 的打印）；数一数出现了几种 group、EP 组在不在（dense 模型预期缺席，`parallel_state.py:2402` 的 `is_moe` 判定）。
3. 找 executor 选择日志(uni/mp)，对照第 1 讲。

**产出**:group 分配标注表（出现了哪些、EP 为何缺席）。

## 实验 3:DP attention 观察（10 min，可选，需 ≥2 卡）

1. 双卡起 MoE 或普通模型：`--data-parallel-size 2`（观察：进程数翻倍；coordinator 进程出现；DP 组日志）。
2. 在日志里找：`dp_sync_interval` 相关同步行（每 16 步）、coordinator 的 finish-sync;
3. 对照 `docs/serving/data_parallel_deployment.md:5-16` 的动机描述，写一句「DP rank 之间什么共享、什么独立」。

**产出**：进程拓扑观察 + DP 共享/独立清单。

## 常见问题

- 实验 1 的组合 1 在 8 卡机器上不报错：把 TP 调到 16 再试；报错点在 `parallel.py:1284-1296` 的卡数检查。
- CPU 后端起 MoE 报 kernel 不支持：EP 相关实验（组合 3）在 CPU 后端可能提前失败于别处；报错发生在配置层即为成功，后置失败可忽略。
- 单卡想看 all2all：看不到——本讲明确教过「纯 TP-EP 无 all2all」；这就是答案本身。
