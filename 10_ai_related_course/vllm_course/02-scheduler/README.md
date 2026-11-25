# 第 2 讲 · 调度器：从 FIFO 到策略化

> 120 分钟 | 基线：nano-vllm `b9dd55a`；vLLM `2f5d8a2b1`(v0.34.1rc0-84,2026-11-25)

本讲回答一个核心问题：**长短请求混着来，每一步的 batch 怎么拼，才不互相拖累？**

知识链：请求进入带策略的等待队列 → `schedule()` 先遍历 `running` 再放行 `waiting` → 双预算(`token_budget` / `input_budget`)控制每步规模 → chunked prefill 把长 prompt 切块 → prefill 与 decode 混进同一个 `SchedulerOutput` → 显存不足时抢占（只有 recompute）→ 被抢占者插队 waiting 队首 → async scheduling 把调度工作与上一步 GPU 执行重叠。

## 一、nano 已学形态（回顾）

先修课第 3 课你在 nano-vllm 里走过、甚至亲手改过这套调度器：

- 三队列：`Scheduler.__init__`（`nanovllm/engine/scheduler.py:10-25`）维护 waiting / running / swapped——swapped 这条还是你 fork 里加的（第 9 讲会回收这条线）。
- **互斥调度**：prefill 分支（`scheduler.py:40-62`）只要本轮产出了 prefill 就直接 return,swap-in 与 decode 都不执行；decode 分支（`scheduler.py:80-94`）每 seq 只前进一格。
- **chunked prefill 仅限首条 seq**：`if remaining < num_tokens and scheduled_seqs: break`（`scheduler.py:52-53`）——这个 `break` 是你亲手写过的。
- 抢占两层决策：`preempt`（`scheduler.py:97-116`）按情况走 recompute（`_recompute`，`scheduler.py:118-123`）或 swap。

nano 的互斥调度很直观，但代价明确：一个 2 万 token 的长 prompt 进来，期间所有 decode 请求全部停摆（TTFT 换整批 TPOT）。生产要解的就是这个。

## 二、演进：从「两阶段」到「无阶段」

**V0**:prefill 与 decode 分开调度，调度器里有 `Policy` 类（优先级函数 + 每步重排），抢占分 recompute / swap 两可（`--preemption-mode`）。

**V1**：方向反过来——不再有阶段。`schedule()` 开头的注释就是设计声明（`vllm/v1/core/sched/scheduler.py:702-711`）：

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the `num_computed_tokens` and `num_tokens_with_spec`…

每个请求只有「已算到第几个 token」的计数，每步预算内能排多少排多少，长 prompt 自然被切成多个 chunk 与 decode 混批——chunked prefill 从 V0 的可选优化变成 V1 的地基（`enable_chunked_prefill` 默认 True）。学术源头是 Sarathi-Serve(OSDI'24,arXiv:2403.02310)的 stall-free 调度：切块让 GPU 每步都有活干，decode 不会因为一个长 prefill 而停摆。

**async scheduling（默认已开）**：配置为 None 时自动解析为 True（`vllm/config/vllm.py:1755-1794`，pooling 模型等场景除外）。它把「下一步的调度」与「上一步的 GPU 执行」重叠起来，代价是调度时还不知道上一步的真实采样结果——这是第 5 环节的内容。

## 三、现状走读

### 环节 1:schedule() 主流程

调用点在第 1 讲已看到：`EngineCore.step`（`core.py:701-731`）的 `:711` 调 `schedule()`，`:725` 调 `update_from_output`。本讲打开这个函数本体——它是一个**约 1,116 行的单一函数**（`scheduler.py:700-1815`），没有子函数拆分。内部结构按注释分四段：

| 段             | 行号      | 做什么                                        |
| -------------- | --------- | --------------------------------------------- |
| 初始化         | 701-775   | 双预算就位、LoRA/collector 清点               |
| 遍历 `running` | 776-1037  | 续跑请求（decode 或 prefill chunk）逐个排入   |
| 遍历 waiting   | 1038-1669 | 新请求准入（含被抢占者的重新准入）            |
| 组装输出       | 1670-1815 | 打包 `SchedulerOutput`（`output.py:240-349`） |

关键认知：没有 `_schedule_prefills()` / `_schedule_decode()` 这类子函数（V0 教程的说法），prefill chunk 和 decode 写进**同一个** `num_scheduled_tokens` dict、同一个 `SchedulerOutput`——「混合批」在代码形态上就是没有分支。

### 环节 2：双预算

每步规模由两个**局部变量**控制（`scheduler.py:728-734`，V1 没有 budget 类）：

- `token_budget` = `max_num_scheduled_tokens`（默认等于 `max_num_batched_tokens`）：本步总调度 token 上限；
- `input_budget` = `max_num_batched_tokens`：本步输入 token 上限（spec decode 的 draft 槽位也从这里扣）。

遍历中每排入一个请求就同步扣减（`:989`、`:1619`）；`running` 循环条件 `while req_index < len(self.running) and token_budget > 0`（`:775-776`）；预算耗尽后 waiting 阶段只放行「不占 running 槽位的异步 KV load」，其余 break（`:1073-1096`）。收尾有三条断言兜底（`:1671-1675`）。

### 环节 3:chunked prefill 切块

切块发生在 running 阶段（`:824-842`）与 waiting 阶段（`:1346-1390`），核心一行：

```python
num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)
```

`--long-prefill-token-threshold` 的语义容易误读：它是**单个请求单步的 prefill 上限**（`:829-830`），不是总预算；而且只在 chunked prefill 开启**且 batch 内请求数 > 1** 时生效（`:757-774`，注释原话「nobody to starve」——只有一个请求时没有需要保护的别人，切块只会白白拖长它自己）。`_adaptive` 变体则以 `input_budget // 请求总数` 做动态下限（`:771-776`）。

### 环节 4：抢占——只剩 recompute

`allocate_slots` 拿不到块时触发抢占决策（`:896-975`）：victim 的选取依赖策略——FCFS 下 `self.running[-1]`（最新的请求先牺牲），PRIORITY 下按 `(priority, arrival_time)` 选最大者。随后 `_preempt_request`（`:1829-1876`）：释放全部 KV 块、`num_computed_tokens = 0`（完全重算）、计数器加一、**`waiting.prepend_request`**（`:1874`）——插回队首而非队尾，抢占即最高优先级。同一步内可以连环抢占直到腾够（`:967-969`）。

V0 的 `--preemption-mode`（recompute / swap 两可）已删除：V1 抢占只有 recompute。swap 的能力没有消失，而是移出了调度器——变成 KV offloading 子系统，那是第 9 讲的内容。被抢占者在 waiting 队首被重新准入时「按新请求全量重发」（`:1606-1613`）。

### 环节 5：策略——队列类型加两处 if

V0 的 `Policy` 类没有对应物。V1 的策略降维成三样东西：`SchedulingPolicy` enum（FCFS / PRIORITY，`vllm/v1/core/sched/request_queue.py:13-17`）、队列数据结构（`create_request_queue`，`request_queue.py:200-207`：FCFS 用 deque,PRIORITY 用按 `(priority, arrival_time, request_id)` 排序的堆）、以及两处 if 分支（抢占 victim，`:915-921`；双等待队列归并，`:2881-2893`）。CLI 是 `--scheduling-policy`，默认 `fcfs`。PRIORITY 下 running 内不做按优先级重排——优先级只影响入队顺序和谁被牺牲。

另有一条「非策略的插队」：`cache_aware_admission_window`（`:2841`）让前缀缓存命中的请求提前准入——prefix cache 不只是省算力，还反过来影响调度公平性，这是第 3 讲的伏笔。

### 环节 6:async scheduling：默认开的流水线

名字有误导性：`AsyncScheduler`（`vllm/v1/core/sched/async_scheduler.py:10`）只覆写了一个方法 `_update_after_schedule`（`:18-62`），给请求预填「输出占位数」，让引擎在不知道上一步真实采样结果的情况下继续调度。真正的重叠机制在 EngineCore：`step_with_batch_queue`（`core.py:746 起`）在队列未满时**先调度下一批并提交执行，再阻塞收上一批的结果**——调度/输出处理的 host 工作与 GPU 执行重叠，深度由 `--async-scheduling-depth`（默认 2）控制。这与你第 1 讲看过的 `run_busy_loop` 是同一处代码的两种模式（PP 的 `batch_queue` 与它共用机制）。

## 四、V0 教程 vs HEAD

| V0 教程说法（大量旧文仍在传）                      | HEAD 上的事实                                                        |
| -------------------------------------------------- | -------------------------------------------------------------------- |
| 「prefill 队列 + decode 队列分开调度」             | 单一 `running` 列表（`scheduler.py:250`）；无阶段划分注释 `:702-711` |
| 「Policy 类，每步按优先级重排」                    | 无 Policy 类；enum + 队列类型 + 两处 if;running 内不重排             |
| 「抢占可选 recompute/swap（`--preemption-mode`）」 | 配置已删除；只有 recompute（swap 归第 9 讲 offloading）              |
| 「被抢占请求回到 waiting 队尾」                    | `prepend_request` 插队首（`:1874`）                                  |
| 「async scheduling 是需要显式开启的实验特性」      | 默认自动开启（None → True，`config/vllm.py:1791`）                   |
| 「token budget 是一个对象」                        | `schedule()` 的两个局部变量（`:729-730`）                            |

## 五、实验与产出

见 [hands-on-exercise.md](hands-on-exercise.md)：实验 1 用脚本制造长短交错负载测 TTFT（需 GPU）；实验 2 改 `--long-prefill-token-threshold` 对照；实验 3（可选）日志里找抢占与 cache-aware 准入。课后作业见 [homework.md](homework.md)。

## 六、边界

| 内容                                | 本讲                      | 去处    |
| ----------------------------------- | ------------------------- | ------- |
| `allocate_slots` 内的 KV 块分配细节 | 只看它失败触发抢占        | 第 3 讲 |
| spec decode 的 draft 槽位与占位符   | 只看预算扣减点            | 第 7 讲 |
| `step_with_batch_queue` 进程模型    | 第 1 讲已铺，本讲用其结论 | —       |
| offloading（swap 的现代形态）       | 只讲「移出了调度器」      | 第 9 讲 |

## 七、前后衔接

第 1 讲的 `EngineCore.step` 五步里，本讲打开了第一步(schedule)；第 3 讲打开 `allocate_slots` 背后的 KV 块世界；第 9 讲把你 fork 里亲手写的 swap 抢占与 vLLM 把 swap 移出调度器的决定对照起来。
