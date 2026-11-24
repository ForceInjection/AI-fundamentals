# 第 8 讲：量化 — PPT 大纲

## 设计说明

- **总页数**：约 22 页
- **演讲时长**:120 分钟（nano 回顾 5 min + 演进 15 min + 现状走读 45 min + 实验 30 min + 答疑 10 min）
- **核心视觉**：加载分发链五步图（本讲锚点，第 9 页）;kernel 优先级列表（第 12 页，与第 4 讲决策树呼应「同构模式」）；方法速查表（第 15 页）;Marlin 部分漂移案例（第 18 页，CUDA 准/Python 变）
- **教学策略**：从「同一个 GPTQ checkpoint 有时走 Marlin 有时不走」这个悬念开场；三大结构变化先立框架；fp8 全走查贯穿；速查表当查表工具
- **配色**:nano 橙色系、分发链蓝色系、kernel 注册表绿色系、失效对照红色系
- **边界**:GPTQ/AWQ 数学不展开；Marlin CUDA 内部引用深读文；精度方法论一句带过

---

## 开场（第 1–2 页）

### 第 1 页：封面

- **标题**：量化
- **副标题**:vLLM 十讲 · 第 8 讲
- **基线信息**:nano-vllm `b9dd55a` + vLLM `1acec5c009` | 核对日期 2026-11-24
- **一张图**：左「nano:BF16 直通」右「vLLM：格式识别 → 方法分发 → kernel 路由 → 后处理」四步链

### 第 2 页：本讲核心问题与议程

- **核心问题**:FP8 的模型和 BF16 的模型，vLLM 的加载与执行路径差在哪?同一个 GPTQ checkpoint，为什么有时走 Marlin 有时不走?
- **四段议程**:nano 回顾（5 min，第 3 页）| 三大结构变化（15 min，第 4–6 页）| 现状走读五环节（45 min，第 7–14 页）| 速查表与 Marlin 案例（15 min，第 15–18 页）+ 实验

## 第一部分：nano 回顾（5 min，第 3 页）

### 第 3 页：nano 回顾：BF16 直通

- **上稿**：权重怎么存就怎么加载；没有 quantization_config、没有 kernel 路由、没有 KV 量化
- **三个新问题**（蓝色）：格式怎么认?kernel 怎么选?运行中能重量化吗?

## 第二部分：三大结构变化（15 min，第 4–6 页）

### 第 4 页：变化一：方法文件合并

- **一张对照表**：旧四文件(gptq/gptq_marlin/awq/awq_marlin)→ 新两文件（auto_gptq/auto_awq；三名字共用一 Config，`__init__.py:137-140`）;bitsandbytes/GGUF 出树 OOT 插件（`:66-101`）;fbgemm deprecated
- **口播**：旧教程的文件路径全部失效——这是本讲修正量最大的部分

### 第 5 页：变化二：kernel 注册表独立

- **一张图**：`vllm/model_executor/kernels/linear/` 注册表：W4A16 顺序（CutlassW4A8 → Machete → **Marlin** → Conch → Exllama → TritonW4A16，`:565-589`）+ 逐个试探（`choose_mp_linear_kernel`，`:1006-1080`）+ `--linear-backend` 强制
- **呼应框**：与第 4 讲注意力决策树同构——「列表 + 谓词」模式再次出现

### 第 6 页：变化三：online 量化子系统

- **上稿**（`quantization/online/` + `config/quantization.py:60-77`）：`--quantization fp8_per_tensor` / `nvfp4_per_token` 等 shorthand 加载时量化 BF16;requantize 已量化 checkpoint；`Fp8Config` 拒绝无 checkpoint 的 fp8（`fp8.py:105-111`）

## 第三部分：现状走读（45 min，第 7–14 页）

### 第 7 页：环节① 注册表与全集

- **上稿**：`QuantizationMethods`（`__init__.py:12-50`）全集概览（awq 系/gptq 系/fp8/w4afp8/modelopt×5/compressed-tensors/mxfp4/humming/inc/quark/torchao/online+10 shorthand）
- **口播**：方法总数比教程说的多得多；但常用的就六七个

### 第 8 页：环节② 分发链总图（锚点）

- **一张大图**：五步分发链（HF quantization_config → overrides 优先级 → Config 类 → resolve_quant_method → get_quant_method → 选 kernel → 后处理 → apply）
- **口播**：后面每页高亮一步

### 第 9 页：环节③ overrides 优先级表

- **上稿**（`config/model.py:1476-1499`）:inkling2 > gptq 系 > awq 系 > … > mxfp4 > deepseek_v4_fp8 > humming；`auto_awq.py:265`「认领」awq checkpoint 改写为 auto_awq；用户显式指定与探测不一致报错（`:1551-1601`）
- **口播**：为什么 GPTQ checkpoint 的方法名会被改写?——旧名字是别名，新名字是「带探测的规范名」

### 第 10 页：环节④ fp8 全走查（一页贯全程）

- **一张流程图**：`Fp8Config.from_config`（`fp8.py:157`）→ `get_quant_method` 三分（`:178-217`：Linear/MoE/Attention）→ `create_weights` 选 kernel（`:355`）→ weight_loader 处理 scale 分片 → `process_weights_after_loading`（scale requant `:397`/marlin 转置 `:376-381`）→ `apply`（`:420-463`）
- **口播**：这五步是所有 quant method 的通用骨架；fp8 是最好的样本

### 第 11 页：kernel 路由的条件散布

- **上稿**:Marlin 路由三处：`auto_awq.py:314-321`（batch-invariant 禁用 + 平台 + 支持）→ `check_marlin_supports_layer`（`marlin_utils.py:305`）→ 失败回退 Triton AWQ（`:324-331`）;MoE 回退 moe_wna16（`:344-357`）；老函数 `cutlass_scaled_fp8_mm` 改名 `ops.cutlass_scaled_mm`
- **回答开场悬念**:GPTQ checkpoint 走不走 Marlin = 注册表顺序 + 三个条件的合取

### 第 12 页：环节⑤ KV cache 量化

- **上稿**（`config/cache.py:47-64` 合法值；`kv_cache.py:21/78-162/189-201`）:fp8(=e4m3)/e5m2 及平台件；scale 的 8 条改名正则（`base_config.py:231-264`）;k/v 互相补与 fnuz ×2;per-token-head scale 新能力；FA3+fp8 时 Q 也量化
- **口播**：第 3 讲的 `--kv-cache-dtype` 在这里有了完整的 scale 世界

### 第 13 页：环节⑥ LLM Compressor 衔接

- **上稿**:recipe 只在离线侧（oneshot 产 checkpoint）;vLLM 只看 `quantization_config`；scheme 映射核心 `_get_scheme_from_parts`（`compressed_tensors.py:787-960`：NVFP4/fp8 w8a8/降级 w8a16/wNa16/int8）;schemes 目录 1:1
- **外部衔接行**:LLM Compressor 官方文档 + DeepLearning.AI 短课（官方背书）

### 第 14 页：在线量化

- **上稿**：`--quantization fp8_per_tensor` 等 shorthand（`config/quantization.py:60-77`）；加载时量化 + requant（`base_config.py:430`）;**`--quantization fp8` 已拒绝无 checkpoint**（`fp8.py:105-111`）
- **口播**:「在线量化」这个词现在专指加载时权重量化；前向的动态激活量化一直有，是另一回事

## 第四部分：速查表与 Marlin 案例（15 min，第 15–18 页）

### 第 15 页：方法速查表（查表工具）

- **一张大表**（README 同款精选）：方法 × W/A × 特点 × 配置来源
- **教法**：当查表工具教；重点关注 fp8/compressed-tensors/gptq/awq/modelopt_fp4/online 六行

### 第 16 页：KV 量化与后端交互

- **上稿**:FA3+fp8（Q 进 fp8 域）；部分后端 + batch invariance 自动关 prefix caching（引用第 4 讲 `attention.py:431`）;turboquant KV(Lloyd-Max)为 KV 专用后端
- **口播**:KV 量化的生效与否是「第 3 讲 × 第 4 讲 × 本讲」三方合议

### 第 17 页：Marlin 部分漂移案例

- **一张对照卡**（红绿分栏）:
  - CUDA 层（未变）：`csrc/quantization/marlin/` 全部文件；`marlin_utils.py` 的 permute 复现对象——仓库旧深读文（`01_marlin-deep-dive.md`，基线 43d691ec6b）的 CUDA 部分仍准
  - Python 层（已变）：封装收敛 auto_gptq/auto_awq;kernel 入口迁 `kernels/linear/mixed_precision/marlin.py:29`；新增 FP8/MXFP4/NVFP4 的 marlin 变体
- **口播**:「部分漂移」的核对方法：分层验证，哪层变了只改哪层——CUDA 深读资产不用作废

### 第 18 页：V0/旧教程 vs HEAD

- **八行对照表**（README 同款）：方法文件合并 / kernel 注册表 / cutlass 函数改名 / fp8 拒绝在线 / bnb-gguf OOT / Marlin 恒 16-bit 激活 / KV scale 全套 / deepseekv2.py 不存在

## 第五部分：实验与收尾（第 19–22 页）

### 第 19 页：本讲实验

- **三行实验卡**：实验 1 LLM Compressor FP8 闭环（课前预处理 + 课上加载对照）；实验 2 kernel 选择观察（纯日志）；实验 3（可选）`--kv-cache-dtype fp8`
- **产出要求**：压缩产物加载成功；kernel 选择日志行；显存对照

### 第 20 页：实验 2 详解（kernel 选择观察）

- **上稿**：同一 GPTQ checkpoint 分别在 batch-invariant 开/关下起服务，日志中的 kernel 选择行差异(marlin vs dequant+matmul)；`--linear-backend` 强制实验
- 思考题：为什么 batch-invariant 模式必须放弃 Marlin?（答案：Marlin 非 batch 不变，逐位一致性要求排除它）

### 第 21 页：实验 1 详解（LLM Compressor 闭环）

- **上稿**：课前：`llmcompressor oneshot` 压 Qwen3-4B FP8（fp8.md 最小示例）；课上：显存对照(BF16 vs FP8)+ 两问（每层 method 是什么?kernel 选了哪个?）——答案都在启动日志
- 精度提示：本实验只看显存与吞吐；精度评估方法论见 `kv_cache/08_methodology`

### 第 22 页：结课与预告

- **本讲你应能**：读分发链日志；解释 kernel 路由；为负载选量化方案
- **预告**：单引擎的全部机制到本讲讲完——第 9 讲回收第 2 讲的伏笔（抢占 swap）:KV 分载与 PD 分离，你 fork 的 kv_swap 与 vLLM 三代 offloading 对照
- **下节材料**：第 9 讲 syllabus 阅读清单

---

## 图资产清单（自制）

| 图 | 页 | 说明 |
| --- | --- | --- |
| BF16 直通 vs 四步链 | 1 | 开场对比 |
| 三大变化卡 | 4-6 | 每变化一页 |
| 分发链五步总图 | 8 | 锚点图 |
| overrides 优先级表 | 9 | 表格直接上稿 |
| fp8 五步流程 | 10 | 一页贯全程 |
| W4A16 优先级列表 | 11 | 与第 4 讲风格呼应 |
| KV scale 处理图 | 12 | 互相补/×2/per-head |
| scheme 映射图 | 13 | compressed-tensors |
| Marlin 部分漂移红绿卡 | 17 | CUDA 准/Python 变 |
| 八条失效对照表 | 18 | 红色系 |
