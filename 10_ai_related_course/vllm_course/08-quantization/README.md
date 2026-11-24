# 第 8 讲 · 量化

> 120 分钟 | 基线：nano-vllm `b9dd55a`；vLLM `1acec5c009`（v0.34.0rc0-1153,2026-11-24 核对）

本讲回答一个核心问题：**FP8 的模型和 BF16 的模型，vLLM 的加载与执行路径差在哪?以及——为什么同一个 GPTQ checkpoint，有时走 Marlin、有时不走?**

知识链：HF checkpoint 的 `quantization_config` → `_verify_quantization` 的 overrides 优先级表（`config/model.py:1446-1601`）→ `get_quantization_config`（`quantization/__init__.py:94-176`）→ `resolve_quant_method`（在线/离线合成，`base_config.py:311-436`）→ 按层类型 `get_quant_method`（Linear / RoutedExperts / Attention 三种）→ `create_weights` 时选 kernel（`kernels/linear` 注册表）→ `process_weights_after_loading` → `apply`。

## 一、nano 已学形态（回顾——本讲反向）

nano 只有裸 BF16：权重怎么存就怎么加载，一次乘法一次反量化都不需要。没有量化路径、没有 kernel 路由、没有 KV 量化。生产版的三个新问题：checkpoint 的量化格式怎么认?每种格式在不同硬件上用哪个 kernel?运行中还能不能把 BF16 权重量化掉（在线量化）?

## 二、演进：三大结构变化

1. **方法文件合并**：旧教程的 `gptq.py` / `gptq_marlin.py` / `awq.py` / `awq_marlin.py` 都不存在了——GPTQ 系收敛进 `auto_gptq.py`（`gptq`/`auto_gptq`/`gptq_marlin` 三个名字共用一个 Config，`__init__.py:137-140`）,AWQ 系同理进 `auto_awq.py`。bitsandbytes 与 GGUF **出树为 OOT 插件**（`vllm-bnb-plugin` / `vllm-gguf-plugin`，经 `register_quantization_config` 注册，`__init__.py:66-101`）。
2. **kernel 选择独立成注册表**：从「quant 方法文件内 if-else」迁到 `vllm/model_executor/kernels/linear/`——w4a16 类按平台优先级列表（CUDA:CutlassW4A8 → Machete → **Marlin** → Conch → Exllama → TritonW4A16，`__init__.py:565-589`）逐个 `is_supported()` + `can_implement()` 试探（`choose_mp_linear_kernel`，`:1006-1080`）。Marlin 的 kernel 本体（CUDA 层）没变——仓库旧深读文 `01_marlin-deep-dive.md` 的 CUDA 部分仍准，变的是 Python 封装路径。
3. **在线量化子系统**：`quantization/online/` + `vllm/config/quantization.py` 的 shorthand（`--quantization fp8_per_tensor` / `nvfp4_per_token` / `int4_per_group_32` 等，`:60-77`）,**加载时**把 BF16 权重量化掉，还能对已量化 checkpoint requantize。配套变化：`Fp8Config` 已拒绝「无 checkpoint 的 fp8」（`fp8.py:105-111` 直接报错并指引改用 shorthand）——老版本「fp8 顺手在线量化」的口径已移除。

## 三、现状走读

### 环节 1：注册表与全集

`--quantization` 合法值全集在 `QuantizationMethods`（`__init__.py:12-50`）:awq 系、gptq 系、fp8、w4afp8(DeepSeek W4A8)、modelopt 五件(NVIDIA)、compressed-tensors、mxfp4/gpt_oss_mxfp4、deepseek_v4_fp8、humming（跨平台）、inc/quark/torchao（平台）、**online + 10 个 shorthand**；`fbgemm_fp8` 标记 deprecated（需 `--allow-deprecated-quantization`）。

### 环节 2：加载分发链（FP8 完整走查）

1. **认格式**：`ModelConfig._verify_quantization`（`config/model.py:1446-1601`）读 HF `quantization_config`，按 **overrides 优先级表**（`:1476-1499`，inkling2 > gptq 系 > awq 系 > … > mxfp4 > deepseek_v4_fp8 > humming）探测——`auto_awq.py:265` 会「认领」`quant_method=="awq"` 的 checkpoint 并改写为 `auto_awq`；用户显式 `--quantization` 与探测结果不一致时报错（`:1551-1601`）。
2. **解析**：`get_quantization_config` → `Fp8Config.from_config`（`fp8.py:157`，读 `activation_scheme`/`weight_block_size`/`ignored_layers`）。
3. **按层给 method**：`resolve_quant_method`（`base_config.py:311-436`）先看 online 合成（`:338-357`，requant 场景调 `set_requantization_source`），再 `get_quant_method`（`fp8.py:178-217`）:LinearBase→`Fp8LinearMethod`；RoutedExperts→`Fp8MoEMethod`；Attention→`Fp8KVCacheMethod`（KV 量化入口）；`ignored_layers` 命中→保持未量化。
4. **选 kernel**：`create_weights`（`:290-362`）内 `init_fp8_linear_kernel`（`:355`）——按 `kernels/linear` 的两张表（W8A8:FlashInfer → Cutlass → B12x → Torch → MarlinFP8 fallback → Humming，`__init__.py:483-520`；128×128 block 的 deepseek 场景另有表，`:520-560`）选第一个可用者，**启动日志会打印选中的 kernel**（docs/kernels.md 口径：日志是权威答案）。
5. **权重后处理**：`process_weights_after_loading`（`:366-418`）：非 block 方案把 TP 分片的 N 份 scale requant 为单一 scale（`:397`）;marlin 路径转置权重成 (K,N)（`:376-381`）。推理走 `apply`（`:420-463`）。

### 环节 3:kernel 路由的关键变化

「同一 GPTQ checkpoint 有时走 Marlin 有时不走」的条件散在三处：`auto_awq.py:314-321`（`use_marlin = not VLLM_BATCH_INVARIANT and is_cuda() and check_marlin_supported(...)`——batch-invariant 模式禁用 Marlin）；`check_marlin_supports_layer`（`marlin_utils.py:305`，tile 不对齐可 padding 处理）失败则回退 Triton AWQ kernel（`auto_awq.py:324-331`）;MoE 不支持时回退 `moe_wna16`（`:344-357`）。`--linear-backend` 可以全局或按量化类型强制（`kernels.md` 官方口径）。`cutlass_scaled_fp8_mm` 这个老函数名已不存在，现名 `ops.cutlass_scaled_mm`（`scaled_mm/cutlass.py:255`）。

### 环节 4:KV cache 量化（承接第 3 讲）

合法值（`config/cache.py:47-64`）:auto/float16/bfloat16/**fp8(=e4m3)/fp8_e4m3/fp8_e5m2** 及平台特有（fp8_inc/nvfp4_ds_mla/turboquant_* 等）。scale 的加载与处理：`KVCacheScaleParameter`（`kv_cache.py:21`，哨兵 -1.0）+ 8 条改名正则（`base_config.py:231-264`，如 `kv_scale→attn.k_scale`）；`process_weights_after_loading` 决定最终 scale（`kv_cache.py:78-162`，k/v 缺一时互相补，fnuz 平台 ×2）。新能力：per-token-head scale（`:97-104`，kernel 写 cache 时动态算）。与后端的交互引用第 4 讲：FA3 + fp8 时 Q 也进 fp8 域；部分后端自动关 prefix caching（`attention.py:431`）。

### 环节 5:LLM Compressor 衔接

关系一句话：**recipe 只存在于离线侧**——LLM Compressor 的 `oneshot(model, recipe)` 产出 compressed-tensors checkpoint（HF `quantization_config: {"quant_method": "compressed-tensors", ...}`）,vLLM 只看产出的 config、不看 recipe。加载侧的 scheme 映射核心是 `_get_scheme_from_parts`（`compressed_tensors/compressed_tensors.py:787-960`）:NVFP4 → W4A4Fp4;fp8 w8a8 → W8A8Fp8（算力不足降级 W8A16Fp8）;wNa16 pack → WNA16（W4A16 主路径）;int8 → W8A8Int8。schemes 目录与映射结果 1:1（`compressed_tensors/schemes/`）。文档 `docs/features/quantization/` 的表格由 gen 脚本从 kernel 注册表自动生成——与第 4 讲一样属于零滞后文档。

## 四、方法速查表（精选）

| 方法名 | W/A | 特点 | 配置来源 |
| --- | --- | --- | --- |
| fp8 | W8/A8（per-tensor 或 128×128 block） | cap 75+;ROCM fnuz 需归一 | `quant_method=fp8` + `activation_scheme` + `weight_block_size` |
| compressed-tensors | W4A16/W4A8/W8A8/W8A16/NVFP4/MXFP4 按 scheme | LLM Compressor 官方格式 | `quant_method=compressed-tensors` + config_groups |
| gptq / auto_gptq / gptq_marlin | W4A16(g_idx) | 三名一 Config;batch-invariant 禁 Marlin | `quant_method=gptq` |
| awq / auto_awq / awq_marlin | W4A16(uint4, zp) | marlin 条件三连；失败回退 Triton | `quant_method=awq` |
| modelopt_fp4 / modelopt_mxfp8 | NVFP4 / MXFP8 | cap 100+ | `quant_algo=NVFP4/MXFP8` |
| w4afp8 | W4A8（FP8 激活） | Hopper cutlass_w4a8 | `quant_method=w4afp8` |
| fp8_per_tensor 等 shorthand | 加载时量化 BF16 | 无 checkpoint 场景 | `--quantization` CLI，无 HF 字段 |
| bitsandbytes / gguf | 8/4bit / GGUF | **OOT 插件** | 插件注册 |

## 五、V0/旧教程 vs HEAD

| 旧说法 | HEAD 事实 |
| --- | --- |
| `quantization/gptq.py` / `awq_marlin.py` 等文件 | 不存在；收敛 `auto_gptq.py`/`auto_awq.py`，别名共用 Config |
| Marlin 路由按 `capability >= 80` 的 if | kernel 注册表顺序 + `is_supported/can_implement`（`kernels/linear/__init__.py:1006-1080`） |
| `cutlass_scaled_fp8_mm` | 改名 `ops.cutlass_scaled_mm`；入口抽象成 `init_fp8_linear_kernel` |
| `--quantization fp8` 可在线量化 BF16 | 报错（`fp8.py:105-111`）;online 归 shorthand |
| bitsandbytes / GGUF 内置 | OOT 插件 |
| 「Marlin 做 INT8/FP8 激活量化」 | 已移除；Marlin 恒 16-bit 激活（W4A8 归 humming，官方 README 更正） |
| KV 只认 `kv_scale` 单标量 | per-head scale、q_scale、prob_scale 全套（`kv_cache.py`） |
| DeepSeek FP8 config 在 `deepseekv2.py` | 在 `vllm/models/deepseek_v41/quant_config.py:62` |

## 六、实验与产出

见 [hands-on-exercise.md](hands-on-exercise.md)：实验 1 LLM Compressor 压 FP8 与加载对照（需 GPU 与时间，可课前预处理）；实验 2 kernel 选择观察（纯日志）；实验 3（可选）`--kv-cache-dtype fp8` 对照。课后作业见 [homework.md](homework.md)。

## 七、边界

| 内容 | 本讲 | 去处 |
| --- | --- | --- |
| GPTQ/AWQ 算法数学 | 不展开 | 深读：`inference_quantization.md` |
| Marlin kernel CUDA 内部 | 只讲路由 | `02_gpu_programming/06_quant_gemm/01_marlin-deep-dive.md` |
| 精度评估方法论 | 一句带过 | `kv_cache/08_methodology/02_inference-benchmark-methodology.md` |

## 八、前后衔接

第 4 讲的「量化 attention 后端」（turboquant 等）在本讲体系里只是众多 quant method 的平行项；第 7 讲的 draft 模型量化（`draft_lm_head_quantization`）同理。第 10 讲生产运维里，「选什么量化」是成本核算的第一变量。
