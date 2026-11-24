# 第 2 讲：调度器 — PPT 大纲

## 设计说明

- **总页数**：约 24 页
- **演讲时长**:120 分钟（nano 回顾 10 min + 演进 15 min + 现状走读 45 min + 实验 40 min + 答疑 10 min）
- **核心视觉**:nano↔V1 同负载调度时序对照图（本讲锚点，第 14 页，实验 1 实测数据）;schedule() 1,114 行行号地图；双预算扣减流程；V0 教程失效对照表
- **教学策略**：从学生亲手写过的 `break`（`nanovllm/engine/scheduler.py:52-53`）切入——你在 nano 里做的折中，就是 vLLM 用无阶段模型消灭的对象；走读按六环节，每环节先给「一句结论」再给行号
- **配色**:nano 对照橙色系、V1 调度器蓝色系、抢占红色系、async scheduling 绿色系；行号统一灰底等宽
- **边界**：`allocate_slots` 的 KV 分配细节留第 3 讲；spec decode 预算只出现扣减点；offloading 只讲「移出了调度器」，细节第 9 讲

---

## 开场（第 1–2 页）

### 第 1 页：封面

- **标题**：调度器：从 FIFO 到策略化
- **副标题**:vLLM 十讲 · 第 2 讲
- **基线信息**:nano-vllm `b9dd55a` + vLLM `1acec5c009` | 核对日期 2026-11-24
- **一张图**：左侧 nano 三队列（waiting/running/swapped，橙色），右侧 vLLM 单一 `running` 列表（蓝色），中间大问号「同一步的 batch 怎么拼?」

### 第 2 页：本讲核心问题与议程

- **核心问题**（大字）：长短请求混着来，每一步的 batch 怎么拼，才不互相拖累?
- **四段议程**:
  - nano 回顾：互斥调度与那个 break（10 min，第 3–5 页）
  - 演进：两阶段 → 无阶段（15 min，第 6–9 页）
  - 现状走读：六环节（45 min，第 10–20 页）
  - 实验：混合负载 TTFT 对照（40 min，第 22 页）

## 第一部分：nano 回顾（10 min，第 3–5 页）

### 第 3 页：nano 回顾：互斥调度

- **复用先修课第 3 课锚点图**（三队列 + schedule 分支）
- **三个锚点**（橙色系）:
  - `Scheduler.__init__`（`nanovllm/engine/scheduler.py:10-25`）:waiting / running / swapped 三队列——swapped 是你们 fork 里加的
  - prefill 分支（`scheduler.py:40-62`）:**本轮有 prefill 就直接 return**,swap-in 与 decode 都不执行
  - decode 分支（`scheduler.py:80-94`）：每 seq 前进一格；`BlockManager.can_append` 询问（`block_manager.py:107-108`）
- **口播**:「互斥很直观：一步要么读题要么写答案。代价马上看」

### 第 4 页：你亲手写过的那个 break

- **上稿**（代码摘录，`nanovllm/engine/scheduler.py:52-53`）:

  ```python
  if remaining < num_tokens and scheduled_seqs: break
  ```

  - 语义：batch 里已有其他 seq 时，新 seq 不允许分块 prefill
  - 这是 nano 对「长 prompt 拖全场」的第一个折中：只保护 batch 里已有的请求

- **口播**:「这个 break 是你们第 3 课的练习题；先修课思考题 2 还问过：如果允许任意 seq 分块，要改哪里?今天看 vLLM 的答案」

### 第 5 页：问题的代价

- **一张图**:TTFT/TPOT 时序图（时间轴上：一条 2 万 token 长请求的 prefill 长条，期间 8 条短请求的 decode 空窗）
- **上稿**:
  - 长请求 prefill 期间，所有 decode 停摆 → 短请求 TPOT 尖刺
  - nano 的缓解（首 seq 分块）只保护「已在 batch 里」的请求；waiting 里的短请求仍要排队等长请求完整 prefill
- **数据位**：实验 1 实测截图占位（课堂上现场替换为真实数字）

## 第二部分：演进（15 min，第 6–9 页）

### 第 6 页：V0 的解法与它的复杂度

- **上稿**（三行，每行「解法 → 新复杂度」）:
  - prefill/decode 分开调度 → 两个队列、两套逻辑
  - `Policy` 类（优先级函数 + 每步重排）→ 可插拔但难测
  - `--preemption-mode`：recompute / swap 两可 → 分支翻倍
- **口播**:V0 把「两种工作模式」做进了调度器；V1 把「模式」删了

### 第 7 页：V1 的解法：无阶段

- **上稿**（设计声明，代码摘录，`vllm/v1/core/sched/scheduler.py:704-713`）:

  ```python
  # NOTE(woosuk) on the scheduling algorithm:
  # There's no "decoding phase" nor "prefill phase" in the scheduler.
  # Each request just has the num_computed_tokens and
  # num_tokens_with_spec. ...
  ```

  - 请求 = 两个计数器（`num_computed_tokens` / `num_tokens_with_spec`），没有「阶段」属性
  - 每步在预算内排「任意请求的任意一段」——chunked prefill 从可选优化变成地基

- **红字**：这四行注释是本讲最重要的一段代码；背下它，后面所有设计都顺了

### 第 8 页：演进时间线

- **时间轴**:V0（两阶段 + Policy + preemption_mode）→ V1（单 running 列表；抢占只剩 recompute；`--preemption-mode` 删除；async scheduling 默认开）
- **三个删除标记**：`Policy` 类 ✗、`preemption_mode` ✗、`vllm/core/` 目录 ✗（git考古：V0 调度器代码已物理删除）

### 第 9 页：Sarathi-Serve

- **上稿**:arXiv:2403.02310(OSDI'24)——chunked prefill + stall-free 调度
  - 切块：长 prompt 分多步排，GPU 每步都有活干
  - stall-free:decode 请求与 prefill chunk 混在同一 batch,decode 不再为长 prefill 停摆
  - vLLM 的吸收：`enable_chunked_prefill` 在 V1 默认 True（`config/scheduler.py:160-166`）
- **口播**：论文里的「stall-free」翻译成 vLLM 的语言，就是「无阶段 + 混合批」

## 第三部分：现状走读（45 min，第 10–20 页）

### 第 10 页：schedule() 全景（行号地图）

- **一张图**:1,114 行函数的纵向地图（`scheduler.py:702-1816`），四段着色：
  - 初始化预算 715-748
  - 遍历 running 778-1027（蓝色）
  - 遍历 waiting 1040-1670（紫色）
  - 组装 SchedulerOutput 1686-1812（灰色）
- **红字**：没有 `_schedule_prefills()` / `_schedule_decode()` 子函数——V0 教程的函数名在这里不存在；prefill chunk 与 decode 写进同一个 `num_scheduled_tokens` dict（`:978` 的 `prefill_scheduled |= request.is_prefill_chunk` 是混合批的直接证据）
- **口播**:「无阶段」在代码形态上就是「没有分支」

### 第 11 页：双预算

- **上稿**（`scheduler.py:728-734`，代码摘录）:

  ```python
  token_budget = self.max_num_scheduled_tokens
  input_budget = self.scheduler_config.max_num_batched_tokens
  ```

  - 两个**局部变量**（没有 budget 类）：`token_budget` 本步调度 token 上限；`input_budget` 本步输入 token 上限（spec draft 槽位从这里扣）
  - 扣减点：running 阶段 `:991-992`、waiting 阶段 `:1621-1622`
  - 耗尽行为：running 循环条件 `:780`；waiting 阶段只放行异步 KV load（`:1073-1096`）；收尾断言 `:1673-1677`

- **默认值表**：`max_num_batched_tokens` 按硬件 2048/8192/16384(arg_utils.py:3000-3033)；`max_num_scheduled_tokens` 默认 = 前者

### 第 12 页：chunked prefill 切块

- **上稿**（核心一行，`scheduler.py:824-842` running / `:1348-1388` waiting 两处）:

  ```python
  num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)
  ```

  - 长请求被切成多步，每步排 `min(剩余未算, 剩余预算)` 个 token
  - 被切的请求留在 `running`，`is_prefill_chunk` 标记，下一步继续

- **口播**：对照 nano——同样叫 chunked prefill,nano 只允许首条 seq,nano 的 `break` 在 V1 里没有对应物

### 第 13 页：threshold 的真实语义

- **上稿**（`scheduler.py:761-776`）:
  - `--long-prefill-token-threshold`：**单个请求单步的 prefill 上限**，不是总预算（`:829-830`）
  - 生效条件：chunked prefill 开启 **且** 请求数 > 1——注释原话「nobody to starve」：只有一个请求时切块只会白白拖长它自己
  - `_adaptive` 变体：`input_budget // 请求总数` 做动态下限（`:771-776`）
- **易错框**（红字）：把它读成「超过这个长度的请求被拒绝」——错；它是切片粒度旋钮
- **对应实验**：实验 2 的三档扫描

### 第 14 页：时序对照（本讲核心图）

- **一张图**（实验 1 实测数据，现场出图）:
  - 上半：nano 互斥时序（长请求 prefill 长条，期间 decode 空窗，短请求 TTFT 尖刺）
  - 下半：V1 混合批时序（长请求切成 N 段，decode 穿插其间，短请求 TTFT 平稳）
  - 标注：两组短请求 TTFT 均值对比
- **口播**：这张图是本讲全部内容的物理验证；数据来自你们的实验 1

### 第 15 页：抢占：只剩 recompute

- **上稿**（`scheduler.py:909-969` 触发 → `:1831-1877` 执行）:
  - 触发：`allocate_slots` 返回 None（KV 块不够——细节第 3 讲）
  - `_preempt_request`（`:1831-1877`）：释放全部 KV 块（`:1849`）、`num_computed_tokens = 0`（`:1854`，完全重算）、`num_preemptions += 1`
  - **`--preemption-mode` 已删除**：`PreemptionMode`/SWAP 全仓零命中；swap 能力移出了调度器（offloading 子系统，第 9 讲回收）
- **红色系**：抢占相关全页用红色，视觉上建立「代价」感

### 第 16 页：victim 与插队

- **上稿**（victim 选择，`scheduler.py:909-969`）:
  - FCFS：`self.running[-1]`——最新的请求先牺牲
  - PRIORITY：`max(running, key=(priority, arrival_time))`；且会把 victim 本步已排的份额回滚归还预算（`:929-957`）
  - 同一步内连环抢占直到腾够（`:967-969`）
- **上稿**（去向，`:1876`）：`waiting.prepend_request(request)`——**插队首**，不是队尾；重新准入时「按新请求全量重发」（`:1606-1613`）
- **口播**:FCFS 下「最年轻先死 + 死了排最前」组合的公平性含义，留给课后作业

### 第 17 页：策略：队列类型 + 两处 if

- **上稿**:
  - `SchedulingPolicy` enum:FCFS / PRIORITY（`request_queue.py:13-17`）
  - 队列即策略：`create_request_queue`（`:200-207`）——FCFS=deque,PRIORITY=按 `(priority, arrival_time, request_id)` 排序的堆（`:131-198`）
  - 策略的全部作用点只有两处 if：抢占 victim（`scheduler.py:917-923`）、双等待队列归并（`:2883-2893`）
- **对照框**:V0 的 `Policy` 类（优先级函数 + running 每步重排）无对应物——PRIORITY 下 running 内不重排
- **CLI**：`--scheduling-policy`，默认 `fcfs`（`config/scheduler.py:195-201`）

### 第 18 页：四条等待队列

- **一张图**：`Scheduler.__init__`（`:231-251`）的队列字段图
  - `waiting` / `kv_holding_waiting` / `async_kv_waiting`（三条按策略创建）+ `deferred_waiting`(set)
  - 入队分流：`_enqueue_waiting_request`（`:2826-2841`）——blocked 进 deferred；持块的进 kv_holding；可异步 load 的进 async_kv；其余 waiting
  - 归并：`_next_waiting_queue`（`:2883-2893`，策略相关的两处 if 之一）
- **口播**:「一条 waiting 队列」是简化说法；生产版按「KV 在不在手」分流——第 9 讲会回收 kv_holding

### 第 19 页：非 policy 的插队

- **上稿**：`cache_aware_admission_window`（`config/scheduler.py:247-268`；实现 `scheduler.py:2843-2881`）
  - 前缀缓存命中的请求可以在窗口内提前准入
  - 含义：prefix cache 不只省算力，还影响调度公平性——缓存命中者插队，未命中者相对延后
- **伏笔框**（蓝色）：这是第 3 讲 prefix caching 的调度面；实验 3 会在日志里找它
- **对应 nano**:nano 没有这个概念——先修课里 prefix cache 只影响算力账

### 第 20 页：async scheduling

- **上稿**:
  - `AsyncScheduler`（`async_scheduler.py:10`）**只覆写一个方法** `_update_after_schedule`（`:18-62`）：预填输出占位（`num_output_placeholders`），让引擎在结果未知时继续调度
  - overlap 真身在 EngineCore：`step_with_batch_queue`（`core.py:759-886`）——队列未满时先调度下一批并 non_block 提交，再阻塞收上一批结果
  - 默认已开：`async_scheduling=None` → 自动 True（`config/vllm.py:1755-1794`）；深度 `--async-scheduling-depth`（默认 2）；不兼容场景(ROCm DeepEP DBO、pooling)显式报错
  - 代价：被抢占时在途输出变 stale（`num_stale_output_tokens`，`:1857-1868`）
- **绿色系**：第 1 讲的 busy loop 在这页变成流水线图（同步串行 vs 重叠）

### 第 21 页：V0 教程 vs HEAD

- **六行对照表**（README 同款，红/绿双色）:
  1. 「prefill/decode 两队列」→ 单 running 列表；无阶段注释 `:704-713`
  2. 「Policy 类每步重排」→ enum + 队列 + 两处 if;running 不重排
  3. 「preemption-mode 两可」→ 已删除；只剩 recompute
  4. 「被抢占去队尾」→ prepend 队首
  5. 「async scheduling 要显式开」→ 默认自动 True
  6. 「token budget 是对象」→ 两个局部变量
- **口播**：拿到任何 V0 时代的调度器文章，先对照这张表再决定信多少

### 第 22 页：本讲实验

- **三行实验卡**（对应 hands-on-exercise.md）:
  - 实验 1(GPU)：`mixed_load_probe.py` 长短交错负载，配置 A/B 对照短请求 TTFT
  - 实验 2(GPU):threshold 三档扫描(512 / 2048 / 100000)
  - 实验 3（零成本）:DEBUG 日志找 chunked prefill、抢占、cache-aware 准入、async 解析结果
- **产出要求**：两组 TTFT 对照表；三档 threshold 三个数字 + 思考题；日志标注表

### 第 23 页：结课与预告

- **本讲你应能**：解释「无阶段」模型；调三个调度旋钮并说出预期；辨认 V0 时代教程的六类失效
- **作业**：源码对照题（nano 允许任意分块要改什么 vs vLLM 的答案）+ 参数决策备忘（Agent 负载四问）
- **预告**:schedule() 每排一个请求都要过 `allocate_slots`——它背后是 KV 块的世界。第 3 讲：KV cache 管理，PagedAttention 的生与死
- **下节材料**：第 3 讲 syllabus 阅读清单（课前预读）

### 第 24 页（备页）:nano/V1 概念对照速查

- **一张表**（答疑用，不正式讲）:
  - nano waiting ↔ vLLM waiting（+三条分流队列）
  - nano swapped+swap 抢占 ↔ 已删除（V1 offloading，第 9 讲）
  - nano 「本轮有 prefill 就 return」↔ 无阶段混合批
  - nano 首 seq 分块 break ↔ 所有请求可分块
  - nano preempt 两层决策 ↔ 只有 recompute

---

## 图资产清单（自制）

| 图                         | 页    | 说明                            |
| -------------------------- | ----- | ------------------------------- |
| nano 三队列锚点图          | 3     | 先修课第 3 课复用               |
| TTFT/TPOT 停摆时序         | 5     | 长请求 prefill 条 + decode 空窗 |
| schedule() 行号地图        | 10    | 四段着色的纵向地图              |
| 双预算扣减流程             | 11    | 局部变量 + 两个扣减点           |
| nano↔V1 时序对照           | 14    | 核心图；实验 1 实测数据现场替换  |
| 抢占流程(victim→prepend)   | 15-16 | 红色系                          |
| 四条等待队列分流图         | 18    | 入队分流 + 归并                 |
| 同步串行 vs overlap 流水线 | 20    | 绿色系；busy loop 两种模式       |
| V0 失效对照表              | 21    | 红绿双色表                      |
