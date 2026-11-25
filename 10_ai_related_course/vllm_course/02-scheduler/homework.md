# 第 2 讲课后作业

## 作业（源码阅读题）

**题 1（必做）**：nano-vllm 先修课的思考题 2 问过：「如果允许任意 seq 分块，`postprocess` 的逻辑需要怎么改？」。现在对照 vLLM 的答案，写一份对照分析：

- vLLM 的 `schedule()`（`scheduler.py:700-1815`）如何做到「任意请求都可分块」而不用改输出处理？给出代码证据（行号）。
- V1 请求状态里的 `is_prefill_chunk` 与 `num_computed_tokens` 各自承担了 nano 里哪个变量/分支的职责？
- nano 的 `postprocess`（`scheduler.py:125-136`）与 vLLM 的 `update_from_output`（`scheduler.py:2334` 起）在「块哈希写回」时机上的差异（本讲只要求指出位置差异，机制细节属第 3 讲）。

**题 2（必做）**：参数决策备忘。给定负载：Agent 场景，80% 请求 prompt ≈ 300 token,20% 请求 prompt ≈ 8K token，输出都很短；单卡 24GB，`Qwen3-4B`。回答：

1. `--max-num-batched-tokens` 取多少（默认值是多少？本机默认呢？查 `vllm serve --help` 或日志）；
2. `--long-prefill-token-threshold` 取多少，为什么；
3. `--scheduling-policy` 用 fcfs 还是 priority，给一条业务理由；
4. 每条决策附「预期收益 + 预期代价」各一句。

## 附加题（不计考核）

实验 2 的 threshold 扫描里，若同时开 `--async-scheduling-depth 4`（默认 2），TTFT 数字会怎么变？先预测再实测，解释或证伪你的预测（提示：深度影响的是调度与执行的重叠，不是切块粒度）。

## 自测题（不提交）

1. V1 调度器里没有「prefill 请求对象」和「decode 请求对象」的区分，它是靠请求身上的哪两个字段推断「这一步给这个请求排多少」？
2. FCFS 策略下抢占为什么选 `running[-1]` 而不是 `running[0]`？这个选择和「被抢占者 prepend 到 waiting 队首」组合起来，对最早到达的请求意味着什么？
3. `--max-num-batched-tokens` 与 `--max-num-scheduled-tokens` 的默认关系是什么？什么场景下需要把后者单独调小？
