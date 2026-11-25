# 第 8 讲教学大纲：量化

## 基本信息

| 项目     | 内容                                                                |
| -------- | ------------------------------------------------------------------- |
| 课时     | 120 分钟                                                            |
| 基线     | nano-vllm `b9dd55a`；vLLM `2f5d8a2b1`(v0.34.1rc0-84)            |
| 先修     | 第 1 讲（权重加载）、第 3 讲(`kv_cache_dtype`)、第 4 讲（后端交互）   |
| 授课形式 | 加载分发链走读 + kernel 路由 + 实验                                 |

## 教学目标

- **知识目标**：画出加载分发链（overrides 优先级表 → Config → `resolve_quant_method` → `get_quant_method` → 选 kernel → 后处理）；说出三大结构变化（方法合并/kernel 注册表/online 子系统）；知道 Marlin 的路由条件与 CUDA 层未变。
- **能力目标**：读启动日志判断「checkpoint 被认领成了哪个方法、kernel 选了哪个」；为给定负载选量化方案并写配置；用二分表排查 kernel 不可用。
- **素养目标**：零滞后文档（gen 自动生成）与手写教程的取舍；CUDA 层与 Python 封装层的「部分漂移」核对方法（Marlin 案例）。

## 时间安排

| 部分            | 时长    | 内容                                                                   |
| --------------- | ------- | ---------------------------------------------------------------------- |
| nano 回顾（反向） | 5 min   | 裸 BF16；三个新问题（格式认知/kernel 路由/在线量化）                    |
| 演进            | 15 min  | 三大结构变化：方法合并/kernel 注册表/online 子系统 + OOT 插件          |
| 现状走读：五环节 | 45 min  | 见下方函数级阅读清单；课堂重点讲分发链与 kernel 路由，fp8 全走查        |
| 实验            | 30 min  | LLM Compressor FP8 + kernel 日志观察（见 hands-on-exercise.md）         |
| 答疑            | 10 min  |                                                                        |

## 函数级阅读清单（课前预读）

| 文件                                                  | 函数 / 类                        | 行号            | 看什么                                                 |
| ----------------------------------------------------- | -------------------------------- | --------------- | ------------------------------------------------------ |
| `vllm/model_executor/layers/quantization/__init__.py` | `QuantizationMethods`            | 12-50           | 方法名全集；fbgemm deprecated(57);OOT 注册(66-101)   |
| `vllm/model_executor/layers/quantization/__init__.py` | `get_quantization_config`        | 94-176          | 名字→Config 类映射（含别名合并）                        |
| `vllm/config/model.py`                                | `_verify_quantization`           | 1446-1601       | HF config 读取；overrides 优先级表(1476-1499)         |
| `vllm/model_executor/layers/quantization/base_config.py` | `resolve_quant_method`        | 311-436         | 在线/离线合成；requant 来源(430)                      |
| `vllm/model_executor/layers/quantization/base_config.py` | `QuantizationConfig` ABC      | 121-308         | 六个抽象方法；cache scale mapper(231-264)              |
| `vllm/model_executor/layers/quantization/fp8.py`      | `Fp8Config.get_quant_method`     | 178-217         | 按层类型三分(Linear/MoE/Attention)                    |
| `vllm/model_executor/layers/quantization/fp8.py`      | fp8 全走查                       | 157-463         | `from_config`→`create_weights`（选 kernel 355）→后处理→apply |
| `vllm/model_executor/kernels/linear/__init__.py`      | `_POSSIBLE_KERNELS`              | 565-589         | W4A16 优先级顺序（Marlin 第三位）                       |
| `vllm/model_executor/kernels/linear/__init__.py`      | `choose_mp_linear_kernel`        | 1006-1080       | 逐 kernel 试探；`VLLM_DISABLED_KERNELS`                  |
| `vllm/model_executor/kernels/linear/__init__.py`      | `init_fp8_linear_kernel`         | 851-936         | FP8 两张表(483-520/520-560)                          |
| `vllm/model_executor/layers/quantization/auto_awq.py` | marlin 三连条件                  | 314-331         | batch-invariant 禁 Marlin；失败回退 Triton             |
| `vllm/model_executor/layers/quantization/kv_cache.py` | scale 处理                       | 21 / 78-162 / 189-201 | KV scale 全套（per-head/互相补/最终写入）        |
| `vllm/config/cache.py`                                | `CacheDType`                     | 47-64           | 合法值；平台限制                                       |
| `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py` | `_get_scheme_from_parts` | 787-960 | LLM Compressor 的 scheme 映射核心                      |
| `vllm/config/quantization.py`                         | online shorthand                 | 60-77 / 276-334 | `fp8_per_tensor` 等名字与参数                           |

## 重点与难点

- **重点**：分发链五步；overrides 优先级表（为什么 GPTQ checkpoint 会被认领成 `auto_gptq`）；kernel 注册表与 `--linear-backend`。
- **难点**：online 与 checkpoint 方法的合成(requant);kernel 路由的条件散布（`auto_awq` 三连 + 注册表顺序）；KV scale 的多形态（per-head/互相补/fnuz ×2）。
- **易错点**：引用 `gptq.py/awq_marlin.py` 等旧文件；写 `cutlass_scaled_fp8_mm`；以为 `--quantization fp8` 能在线量化 BF16；去找 bnb/gguf 的内置实现。

## 参考材料

- 官方：`docs/features/quantization/`（README + formats + kernels 表格自动生成，零滞后；README 有两条老口径更正：Marlin 恒 16-bit 激活、`fp_quant`/QuTLASS 已删）、`llm_compressor/` 子目录（fp8/int4/int8_w4a8/int8_w8a8 四篇）
- 深读：仓库 `02_gpu_programming/06_quant_gemm/01_marlin-deep-dive.md`（CUDA 层，基线 43d691ec6b 仍准）、`09_inference_system/model_optimization/inference_quantization.md`
- 外部：DeepLearning.AI 短课量化模块；LLM Compressor 仓库文档
