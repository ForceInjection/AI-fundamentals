# 第 5 讲：CUDA Graph 与编译 — PPT 大纲

## 设计说明

- **总页数**：约 23 页
- **演讲时长**:120 分钟（nano 回顾 10 min + 机制回顾 10 min + 现状走读 45 min + 实验 25 min + 答疑 10 min）
- **核心视觉**：三态对照柱状图（第 6 页，实验 1 实测）；两族尺寸网格示意（第 11 页）;piecewise 切分图（第 13 页，attention 摘出图外）；八条旧说法失效表（第 20 页）
- **教学策略**：用第 1 讲实验 3 存下的 eager 对照数据开场（「你已经有体感了」）；机制 30 秒版先立住 launch 开销，再讲「图为什么救它、又带来什么代价」；调试二分表当工具教
- **配色**:nano 橙色系、机制图蓝色系、piecewise/FULL 对比绿色系、调试红色系
- **边界**:Inductor 产码细节不展开；draft 图只讲融合结论（spec 语义第 7 讲）;AttentionCGSupport 与第 4 讲矩阵交叉引用

---

## 开场（第 1–2 页）

### 第 1 页：封面

- **标题**:CUDA Graph 与编译
- **副标题**:vLLM 十讲 · 第 5 讲
- **基线信息**:nano-vllm `b9dd55a` + vLLM `1acec5c009` | 核对日期 2026-11-24
- **一张图**：左「eager：每 kernel 一次 launch（火柴棒图）」右「图：一次 launch 全播放」中间数字「TPOT 差 N 倍（实验 1 实测）」

### 第 2 页：本讲核心问题与议程

- **核心问题**:decode 为什么离不开 CUDA Graph,piecewise 又解决了「整图」的什么死穴?
- **五段议程**:nano 回顾（10 min，第 3–4 页）| 机制 30 秒版（10 min，第 5 页）| 现状走读六环节（45 min，第 7–19 页）| 实验（第 21–22 页）| 调试表（并入走读）

## 第一部分：nano 回顾（10 min，第 3–4 页）

### 第 3 页：nano 回顾：整段录像

- **复用先修课第 8 课锚点图**（capture_cudagraph → run_model 分支）
- **锚点**（橙色）:
  - capture：`model_runner.py:239-274`，非 eager 时为各 batch size 录整段 decode 前向
  - replay：写 slot_mapping/context_lens/block_tables 进缓冲 → `graph.replay()`（`:212-229`）
- **口播**:「录一遍、回放 N 次——消的就是 CPU 逐 kernel launch」

### 第 4 页：nano 的三个硬编码折中

- **上稿**（三行，每行「折中 → 生产撞墙」）:
  - prefill 不进图（prefill 形状千变）→ 生产里 prefill 也想要图
  - batch > 512 直接 eager → 上限拍脑袋
  - capture 尺寸集手工有限 → 真实 batch 连续，padding 浪费不可控
- **口播**:V2 的解法逐条对着这三点：网格采样、FULL_AND_PIECEWISE、attention 摘出

## 第二部分：机制 30 秒版（10 min，第 5–6 页）

### 第 5 页：CUDA Graph 语义 30 秒

- **上稿**（`debug_vllm_compile.md:346-360` 的五句）:
  - 图装 1+ 个 kernel;replay 一次 launch 播全部
  - replay 读写**同一片内存** → 新数据要先 copy 进图引用的固定 buffer
  - 只捕 CUDA kernel，不捕 CPU 工作
  - launch 开销：batch 小、kernel 短时占主导（`cuda_graphs_multimodal.md:54`）；融合收益实证「2 launches → 1」（`fusions.md:292`）
- **口播**:「地址固定」是全部代价的根源——后面 piecewise、FULL 降级都由它派生

### 第 6 页：三态对照（实验 1 预告）

- **一张柱状图**:TPOT 对比（eager / 默认 -O2 / -O0——注意 -O0 关图也关编译，与 eager 有别）
- **上稿**：实验 1 会亲手测；第 1 讲实验 3 存的 eager 数据先垫底
- **口播**：数字课堂上现场出；「你已经有体感了，现在要的是解释」

## 第三部分：现状走读（45 min，第 7–19 页）

### 第 7 页：环节① capture 流程

- **一张图**：启动期时序（compile warmup → capture_model → 日志「finished in X secs, took Y GiB」→ lock_workspace）
- **上稿**（`gpu_worker.py:1155-1157` → `model_runner.py:1172-1252`）:capture **不走 `_dummy_run`**（`:914` 只服务 profiling;「dummy run 触发 capture」是旧说法）——由闭包工厂自造 dummy 输入（`cudagraph_utils.py:743-857`）
- **口播**:capture 完锁 workspace（`:1241`），防 resize 让图指针失效——一个容易踩的坑

### 第 8 页：输入静态化

- **一张图**：固定 input_buffers + 每步 copy_in → replay → 从持久输出 buffer 切片
- **上稿**（`cudagraph_utils.py:743-857` factory；`model_runner.py:2172-2180` 注释）：图内只引用 buffer 固定地址（`:759-771` 切片）；运行时真实数据先拷进 buffer；输出按最宽图预分配持久 buffer（`store_capture_output`，`:707-741`）

### 第 9 页：环节② 两族尺寸网格

- **一张表 + 一张网格示意**（两族，`compilation.py`）:
  - decode 请求族：(n, q) 二维；≤6 全捕，之后 `0.65·n^(2/3)` 取 2 幂（`:122-125`）；上限 max_num_seqs(SM100)
  - prefill token 族：1,2,4 → 每 8 到 64 → 倍增 4 档 → 每 256/512，共 54 尺寸（`:89-99`）；上限 8192
- **红字**：旧教程「1,2,4,…,512 共 25 档」已废；旧字段 deprecated（`:788-798`）

### 第 10 页：环节③ 每步的中央裁判

- **一张图**:execute_model 开头的 dispatch 流（eager 判定 → dispatch 落桶「最小能容纳的图」→ DP 对齐 → padding 补齐 → replay；无图 → NONE eager）
- **上稿**：`dispatch_cg_and_sync_dp`（`dp_utils.py:232-369`；调用点 `model_runner.py:1933-1949`）;padding 补齐（`:1502`）;PIECEWISE 不需要 request padding（`cudagraph_utils.py:367-375`）

### 第 11 页：环节④ piecewise:attention 在图外

- **一张核心图**：模型 forward 横条，attention 段标红（图外），其余段标绿（图内小图）;splitting_ops 清单截选（`compilation.py:962-987`，~20 个 op）
- **上稿**:
  - attention 被包成 custom op `vllm::unified_attention_with_output`，列入 splitting_ops → Dynamo 看到全图，切分后 attention 留图外
  - 辨析：piecewise **编译**（每子图过 Inductor 产 kernel，`piecewise_backend.py:86`）与 piecewise **图**（每段捕小图消 launch）——同一套切分，两个用途
  - breakable cudagraph 变体：不走 compile，运行时断点续捕（`breakable_cudagraph.py`）

### 第 12 页：FULL 模式与降级链

- **一张图**:AttentionCGSupport 四级(ALWAYS > UNIFORM_BATCH > UNIFORM_SINGLE_TOKEN_DECODE > NEVER)→ `resolve_cudagraph_mode_and_sizes`（`compilation.py:1588-1737`）降级链：FULL → FULL_AND_PIECEWISE / FULL_DECODE_ONLY / PIECEWISE / NONE
- **上稿**:FULL（attention 也在图内）需后端声明 ALWAYS/UNIFORM_BATCH;draft 只支持 FULL_DECODE_ONLY，且**多步融合进一张 FULL 图**（`speculator.py:38-52`；`model_runner_v2.md` §9）——「逐步各捕一张」已过时
- **口播**：第 4 讲的支持矩阵在这里长出了「图支持」这一列

### 第 13 页：环节⑤ torch.compile 四档

- **一张表**（`vllm.py:301-384`）:
  - O0：不编译不捕图；融合全关
  - O1：编译 + PIECEWISE；条件融合；autotune 开
  - **O2（默认）**:FULL_AND_PIECEWISE;TP 加 allreduce+rms / rope+kv 融合
  - O3：当前 == O2（`vllm.py:343-384`）
- **上稿**：融合 pass 落在 `compilation/passes/fusion/`；compile 与 capture 尺寸分开的理由（`compilation.py:537-546`：inductor 一份通用 shape，图必须逐尺寸）
- **易错框**：`--enforce-eager` 直接关的只是 capture（`gpu_worker.py:1155-1157`）;compile 看 `-O` 档位——两者在 V2 里解耦了

### 第 14 页：环节⑥ 调试：二分旗标表

- **一张表**（`debug_vllm_compile.md:8-15`）：`--enforce-eager`（全关）→ `-cc.mode=0`（关编译）→ `-cc.cudagraph_mode=NONE`（关图）→ `-cc.backend=eager`（关 Inductor）
- **上稿**:TORCH_TRACE + tlparse 出编译 HTML（报 bug 附上）；`TORCH_LOGS=output_code`；graph break 按 Dynamo 报告；缓存清理 `~/.cache/vllm`
- **红色框**：`vllm compile` 子命令、`VLLM_COMPILE_DEBUG`、`--cudagraph-mode` 三个**都不存在**

### 第 15 页：V0/旧教程 vs HEAD（上）

- **四行对照表**：默认 PIECEWISE → -O2 FULL_AND_PIECEWISE;dummy run 触发 capture → capture_model 闭包工厂；尺寸 25 档 → 两族网格；FA3 不支持图 → AttentionCGSupport 自动降级
- **续表见下页**（留出呼吸空间）

### 第 16 页：V0/旧教程 vs HEAD（下）

- **四行对照表（续）**:draft 逐步捕图 → 一张 FULL 融合；--cudagraph-mode/vllm compile/VLLM_COMPILE_DEBUG → 不存在；enforce_eager 关 compile+图 → 只关 capture;CudagraphDispatcher 类 → 不存在（dispatch 在 CudaGraphManager）
- **口播**：八条里六条是「默认值与 CLI 变了」——版本迁移最常翻车的就是这类

### 第 17 页：与第 4 讲的互锁

- **一张图**：第 4 讲支持矩阵 + 「AttentionCGSupport」一列 → 决定 FULL 可用性 → `resolve_cudagraph_mode_and_sizes` 降级 → 本讲图模式
- **上稿**：后端声明（第 4 讲）→ 管理器降级（本讲）→ 运行时 dispatch——一条链

### 第 18 页：数字与口径

- **上稿**：本讲可引用的量化口径清单——capture 5~20 秒（`model_runner.py:1246`）、Inductor 断言 ~2ms/步（`debug_vllm_compile.md:269`）、融合 2→1 launch（`fusions.md:292`）、小 batch launch 主导（`cuda_graphs_multimodal.md:54`）；其余「launch 占比 X%」类数字仓库内无官方口径，引用须外查（PyTorch 官方文档/vLLM 博客）
- **口播**：本讲纪律——仓库里没有的数字，不编

### 第 19 页：与第 7 讲的接口

- **一张小图**:draft 多步融合一张 FULL 图（目标模型 + draft 两个 manager）
- **上稿**:spec decode 的图语义（draft 只 FULL_DECODE_ONLY）留待第 7 讲展开；本讲记住「融合」这个动作

## 第四部分：实验与收尾（第 20–23 页）

### 第 20 页：本讲实验

- **三行实验卡**：实验 1（核心）三态 TPOT 对照；实验 2 capture 日志解析（`capture_log_parser.py`）；实验 3（可选）旗标二分演练
- **产出要求**：三态数字柱状图；解析输出截图；二分过程记录

### 第 21 页：实验 1 详解

- **上稿**：三次起服务（enforce-eager / 默认 / -O0）× 同一探针（`mixed_load_probe.py`）；记录 TPOT 均值与启动时长（capture 5~20 秒的体感）
- 思考题：-O0 与 enforce-eager 数字几乎一样，为什么?（答案：-O0 也是 NONE 模式；差异在 compile 产物是否存在）

### 第 22 页：实验 2 详解（capture 日志解析）

- **上稿**：`capture_log_parser.py` 读 serve 日志，输出图数量（两族）、capture 耗时、显存 GiB；改 `--max-num-seqs` 再起，看 decode 族图数量变化（对照网格公式）
- 思考题：为什么 prefill 族 54 个尺寸而 decode 族少得多?(答案：请求族二维 (n,q) 已经覆盖 token 连续区间；且 decode 才是 replay 主场)

### 第 23 页：结课与预告

- **本讲你应能**：三态对照并解释；读懂 capture 日志；用二分表定位编译/图问题
- **预告**：图与编译把每步算得更快了——但 batch 里装谁、装多少，还有一整个调度世界（第 2 讲）之上的并行扩展。第 6 讲：并行，TP 到 EP
- **下节材料**：第 6 讲 syllabus 阅读清单

---

## 图资产清单（自制）

| 图 | 页 | 说明 |
| --- | --- | --- |
| launch 火柴棒对比 | 1 | eager vs 图 |
| nano 录像锚点图 | 3 | 先修课第 8 课复用 |
| 三折中对照 | 4 | 折中 → 生产撞墙 |
| CUDA Graph 语义五句 | 5 | 官方口径摘录 |
| capture 时序 | 7 | compile 先行 → capture → lock |
| 静态 buffer 图 | 8 | copy_in → replay → 切片 |
| 两族网格示意 | 9 | decode (n,q) 与 prefill token |
| dispatch 中央裁判流程 | 10 | 每步落桶 |
| piecewise 切分横条 | 11 | 核心图；attention 红色图外 |
| FULL 降级链 | 12 | 四级 support → 模式 |
| 四档对照表 | 13 | 默认 O2 高亮 |
| 二分旗标表 | 14 | 红色系 |
| busy loop 流水线 vs 串行 | 20 | 绿色系（第 1 讲图复用） |
