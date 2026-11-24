# 第 4 讲：注意力后端 — PPT 大纲

## 设计说明

- **总页数**：约 22 页
- **演讲时长**:120 分钟（nano 回顾 10 min + 演进 15 min + 决策树走读 40 min + 支持矩阵 15 min + 实验 30 min + 答疑 10 min）
- **核心视觉**：五层决策树（本讲锚点，第 9 页）；后端×特性支持矩阵（第 15 页，每格带行号）；两个走查例对比卡（Qwen3 vs DeepSeek V3.2，第 13 页）
- **教学策略**：从 nano 的一处 if-else 切入，「一个分支 vs 一棵树」开场；决策树按五层讲，每层给一行结论；支持矩阵当查表工具教（不是背）
- **配色**:nano 橙色系、决策树蓝色系（五层深浅）、拒绝/互斥红色系、矩阵表格灰底
- **边界**:kernel 数学不展开（FA/MLA 原理是深读素材）;page table 传递只承接第 3 讲；量化 attention 不讲

---

## 开场（第 1–2 页）

### 第 1 页：封面

- **标题**：注意力后端：从 if-else 到后端矩阵
- **副标题**:vLLM 十讲 · 第 4 讲
- **基线信息**:nano-vllm `b9dd55a` + vLLM `1acec5c009` | 核对日期 2026-11-24
- **一张图**：左「nano：`context.is_prefill` 一处 if（橙）」右「vLLM:35 文件后端目录（蓝）」中间箭头「矩阵怎么长出来」

### 第 2 页：本讲核心问题与议程

- **核心问题**：同一个模型部署到不同硬件，引擎怎么自动决定用哪个注意力 kernel?强制指定时，它凭什么拒绝你?
- **四段议程**:nano 回顾（10 min，第 3–4 页）| 演进：if-else → 列表+谓词（15 min，第 5–7 页）| 决策树五层 + 两个走查例（40 min，第 8–14 页）| 支持矩阵（15 min，第 15–17 页）+ 实验（第 21 页）

## 第一部分：nano 回顾（10 min，第 3–4 页）

### 第 3 页：nano 回顾：一处 if-else 决定一切

- **复用先修课第 7 课锚点图**（store_kvcache → prefill/decode 分支）
- **锚点**（橙色）:
  - `store_kvcache`（Triton kernel，`attention.py:10-30`；slot==-1 跳过）
  - prefill：`flash_attn_varlen_func`（`:64-70`）
  - decode：`flash_attn_with_kvcache`（`:71-75`）
- **口播**:「nano 的后端选择就是这一处 if;kernel 固定 FlashAttention——没有第二选项」

### 第 4 页：三个生产约束逼出矩阵

- **上稿**（三栏）：模型架构多样化(GQA/MLA/Mamba-GDN)| 硬件多样化(SM80-120/ROCm/CPU)| 特性组合（sliding window/稀疏/batch invariance）
- **结论行**：一个 if-else 每加一个维度就翻一倍——控制流撑不住了

## 第二部分：演进（15 min，第 5–7 页）

### 第 5 页：旧世界：if-else 短路链

- **上稿**：旧版按平台逐个试，命中即用；新后端 = 改选择逻辑 = 全量回归
- **口播**：问一个问题——「如果让它变成数据，加后端还要不要改逻辑?」

### 第 6 页：新世界：优先级列表 + 统一校验

- **一张图**：选择 = 生成列表（arch×算力）→ 逐个过谓词（校验）→ 取第一个通过者
- **上稿**：加新后端 = 注册(registry)+ 覆盖几个 `supports_*` 方法；选择逻辑零改动
- **红字**:「控制流变数据」——本讲的软件工程结论

### 第 7 页：注册机制

- **上稿**（`vllm/v1/attention/backends/registry.py`）:
  - `AttentionBackendEnum`（`:34`）:**成员名即后端名，成员值即默认 qualname**（不是 dict；`:44-179` 全部默认映射）
  - 运行时覆盖表 `_ATTN_OVERRIDES`（`:357`）；`get_class` 经 `resolve_obj_by_qualname` importlib 拿类（`:233-244`）
  - `register_backend`（`:361-433`）：装饰器或显式注册；字符串名动态建 enum 成员
- **易错框**:composite 后端(TRITON_FLASH_ATTN/TRITON_FLASHINFER)是工厂生成，**源码里搜不到 class**——搜不到不等于不存在

## 第三部分：决策树五层（40 min，第 8–14 页）

### 第 8 页：决策树全景（本讲锚点）

- **一张大图**：五层树（蓝色系五段）:①显式 `--attention-backend` → ②平台入口（`cuda.py:462`）→ ③优先级列表（`cuda.py:82-179`）→ ④`validate_configuration` 统一过滤（`backend.py:296-389`）→ ⑤取第一个通过；全拒 → ValueError + 每个后端的拒绝理由（`:500-517`）
- **口播**：后面五页逐层放大；第 ④ 层是灵魂

### 第 9 页：层①②：显式指定与平台入口

- **上稿**:
  - `--attention-backend` → `AttentionBackendEnum[名.upper()]`（`config/attention.py:273-285`；「auto」→None）
  - 非法名：enum 元类报 Unknown + 合法选项列表（`registry.py:18-31`）
  - 平台入口：CUDA `cuda.py:462` / ROCm `rocm.py:770` / CPU `cpu.py:148` / XPU `xpu.py:175`
- **红色警示框**：旧教程的 `VLLM_ATTENTION_BACKEND` 环境变量**已删除**(PR #32812)——这是本讲第一个「过期说法」

### 第 10 页：层③：优先级列表（arch × 算力）

- **一张表**:MLA 分支（`cuda.py:94-153`）与非 MLA 分支（`:154-179`）按 cc 代际展开：
  - MLA:SM100 fp8 → [FLASHINFER_MLA_SPARSE, FLASHMLA_SPARSE];SM90 → [FLASH_ATTN_MLA, FLASHMLA, FLASHINFER_MLA, TRITON_MLA,…]
  - 非 MLA:SM100 因果 → [FLASHINFER, FLASH_ATTN, TRITON_ATTN,…]；其余 → [FLASH_ATTN, FLASHINFER,…]
  - cc==9 且图文混合批 → 头部插 TRITON_FLASH_ATTN(composite)
- **口播**：列表顺序 = 厂商的默认偏好；校验才是个性的裁决

### 第 11 页：层④:validate_configuration（灵魂）

- **上稿**（`backend.py:296-389`）：十余项逐一查——head_size（`:320`）/ dtype（`:322`）/ kv_cache_dtype（`:324`）/ block_size（`:326`）/ **MLA 匹配（`:332-336`，`use_mla != cls.is_mla()` 即拒）**/ **稀疏匹配（`:339-343`）**/ cc（`:346`）/ **滑窗（`:350-351`）**/ non-causal / batch-invariant / kv_connector / pcp-dcp / 组合项（`:376-388`）
- **红色重点框**:MLA 互斥——`use_mla == cls.is_mla()` 严格相等；基类 `is_mla()=False`，标准后端永远不会接 MLA
- **口播**：每个被拒后端的理由都被收集——排错时这个理由列表比结论有用

### 第 12 页：层⑤ + 兜底

- **上稿**：取 priority 最小的通过者（`cuda.py:495-527`）；全拒 → `ValueError: No valid attention backend found` + 理由列表（`:500-517`）；`--block-size` 挤掉高优先后端时给 warning（`:529-548`）
- **口播**:「它拒绝你的时候，理由都给你了——读懂这份列表，比背矩阵有用」

### 第 13 页：两个走查例（对比卡）

- **左右对比卡**（各 5 行）:
  - **Qwen3(dense GQA)**：非 MLA 分支 → H100:FLASH_ATTN（SM90 默认 FA3）| B200:FLASHINFER 优先
  - **DeepSeek V3.2(MLA+DSA)**:sparse 标记拒掉全部 dense MLA → H100:FLASH_ATTN_MLA_SPARSE（仅 cc==9）| B200 bf16:FLASHMLA_SPARSE | fp8 KV:FLASHINFER_MLA_SPARSE
  - 补充行：V3.2 的 indexer 后端由模型直连（`mla/indexer.py:221`）;MLA prefill 另有独立选择器（`mla/prefill/selector.py:46`）
- **口播**：同一份代码，模型与硬件不同，答案完全不同——这就是「矩阵」的含义

### 第 14 页：执行期贯通

- **一张图**:ModelRunner（`model_runner.py:2080` prepare_attn）→ `build_attn_metadata`（`attn_utils.py:464`，CommonAttentionMetadata 构建在 `:540-563`）→ `set_forward_context`（`model_runner.py:2188`）→ `Attention.forward`（`attention.py:537-655`）→ `impl.forward`（`flash_attn.py:1284`；block_table 读取 `:1387`）
- **上稿**：选择发生在 `__init__`（一次）；元数据每步构建；forward 经 torch 自定义算子取 context
- **口播**：第 3 讲的 block_table 在这里被 attention「看见」

## 第四部分：支持矩阵（15 min，第 15–17 页）

### 第 15 页：后端×特性矩阵（查表工具）

- **一张大表**（灰底；README 同款，每格带行号）:
  - FLASH_ATTN:GQA ✓(cc≥8.0)| MLA ✗ | 滑窗 ✓ | Mamba ✗
  - FLASHINFER:GQA ✓(SM80-12.1)| MLA ✗ | 滑窗 ✓ | Mamba ✗
  - TRITON_ATTN:GQA ✓（任意 cc）| MLA ✗ | 滑窗 ✓ | Mamba ✗
  - FLASHMLA:GQA ✗ | MLA ✓ | 滑窗 ✗（仅 TRITON_MLA 支持）| Mamba ✗
  - GDN_ATTN：独立注册表(is_ssm=True)
- **教法**：当查表工具教；会查 > 会背

### 第 16 页：两个结构性结论

- **上稿**:
  1. **MLA 与标准后端互斥**：`is_mla()` 严格相等——GQA 模型永远不会选到 FLASHMLA，反之亦然
  2. **Mamba/GDN 是另一棵树**：`MambaAttentionBackendEnum`（`registry.py:260-354`）独立；GDN 来源是模型层返回（`mamba/gdn/base.py:49-51`）
- **一张小图**：两棵决策树并排（标准树 + Mamba 树），混合模型(Qwen3-Next)同时走两棵

### 第 17 页：V0/旧教程 vs HEAD

- **六行对照表**（README 同款）：环境变量已删 / FA3-FA4 非后端名 / get_attn_backend 在 selector.py / MLA 预填解码两个选择器 / AttentionMetadata 更名 Common / 文档自动生成零滞后
- **绿色框**：`attention_backends.md` 由 gen 脚本从 validate_configuration 生成——本课程目前发现的唯一零滞后设计文档；手写文档要当「可能过期」对待

## 第五部分：实验与收尾（第 18–22 页）

### 第 18 页：本讲实验

- **三行实验卡**：实验 1 强切对照（合法值吞吐对比 + 非法值两种报错）；实验 2 决策树复算（手推优先级列表对日志答案）；实验 3（可选）Gemma-2 滑窗观察
- **产出要求**：强切对照数字；复算过程纸面照；报错截图

### 第 19 页：实验 2 详解（复算方法）

- **一张流程图**：手推模板（模型架构 → MLA? → cc 代际 → 抄优先级列表 → 逐项校验 → 首个通过）
- 两个练习：Gemma-2 on A100;Qwen3-Next on B200（预期走两棵树）
- 答案来源：启动日志 `init engine` 阶段 + `--attention-backend` 试错

### 第 20 页：常见坑

- **上稿**：强切 FLASHINFER 到 MLA 模型 →「not valid + 理由」；复合后端名写错 → Unknown + 合法列表；batch invariance + FLASHINFER → prefix caching 自动关闭（`attention.py:431`）
- **口播**：报错信息是文档的一部分——vLLM 的报错质量相当高

### 第 21 页：结课与预告

- **本讲你应能**：手推决策树；读懂拒绝理由列表；辨认可疑的 V0 说法（环境变量/FA3 后端）
- **预告**:attention 前后夹着的两件事——CUDA Graph 捕获与 torch.compile，第 5 讲：为什么 decode 离不开图，piecewise 又解决了它的什么死穴
- **下节材料**：第 5 讲 syllabus 阅读清单

### 第 22 页（备页）：后端名速查

- **一张表**（答疑用）：标准 5 个(FLASH_ATTN/FLASHINFER/TRITON_ATTN/FLEX_ATTENTION/TURBOQUANT)+ MLA 常用 4 个 + Mamba 5 个 + 平台件说明；NO_ATTENTION 为死条目（registry.py:157 指向的类不存在）

---

## 图资产清单（自制）

| 图 | 页 | 说明 |
| --- | --- | --- |
| 一处 if vs 35 文件目录 | 1 | 开场对比 |
| 列表+谓词示意 | 6 | 「控制流变数据」 |
| 注册机制图 | 7 | enum → overrides → qualname |
| 五层决策树 | 8 | 锚点图；后续页高亮 |
| 优先级列表两分支表 | 10 | MLA/非 MLA × 代际 |
| validate 十余项清单 | 11 | 每项标行号 |
| 走查例对比卡 | 13 | Qwen3 vs V3.2 左右卡 |
| 执行期贯通链 | 14 | ModelRunner → forward |
| 后端×特性矩阵 | 15 | 每格带行号 |
| 两棵决策树并排 | 16 | 标准 + Mamba |
