# 第 5 讲教学大纲：CUDA Graph 与编译

## 基本信息

| 项目     | 内容                                                                |
| -------- | ------------------------------------------------------------------- |
| 课时     | 120 分钟                                                            |
| 基线     | nano-vllm `b9dd55a`；vLLM `2f5d8a2b1`(v0.34.1rc0-84)            |
| 先修     | 第 1 讲(Worker/Model Runner)、第 2 讲（`mixed_load_probe` 脚本复用）  |
| 授课形式 | 机制对照 + 源码走读 + 三态实验 + 调试演练                           |

## 教学目标

- **知识目标**：说清 CUDA Graph 省的是什么（launch 开销）与代价（地址固定、输入必须进 buffer）；讲出 piecewise 的切分机制(`splitting_ops`)与 FULL 模式的可用条件（AttentionCGSupport 四级）。
- **能力目标**：三态对照(eager / -O2 / -O0)测 TPOT 并解释差异；读懂 capture 日志（图数量/耗时/GiB）；用旗标二分表定位一个编译/图问题。
- **素养目标**：识别 V0/V1 时代教程在本主题上的八类失效（默认值、触发方式、尺寸网格、CLI 全都变过）。

## 时间安排

| 部分              | 时长    | 内容                                                                   |
| ----------------- | ------- | ---------------------------------------------------------------------- |
| nano 回顾         | 10 min  | 整段录像 + 三个硬编码折中（prefill 不进图/512 上限/有限尺寸集）         |
| 机制回顾          | 10 min  | launch 开销 30 秒版；capture 语义（1+ kernel 装一张图，输入进 buffer）   |
| 现状走读：六环节   | 55 min  | 见下方函数级阅读清单；课堂重点讲环节 2（两族网格）、4(piecewise)、5（四档） |
| 实验              | 35 min  | 三态对照 + capture 日志解析                                            |
| 答疑              | 10 min  |                                                                        |

## 函数级阅读清单（课前预读）

| 文件                                          | 函数 / 类                          | 行号            | 看什么                                                 |
| --------------------------------------------- | ---------------------------------- | --------------- | ------------------------------------------------------ |
| `vllm/v1/worker/gpu_worker.py`                | `compile_or_warm_up_model`         | 1068-1090       | compile 先行；capture 触发条件(1159-1162)             |
| `vllm/v1/worker/gpu/model_runner.py`          | `capture_model`                    | 1172-1252       | capture 总入口；日志与 `lock_workspace`(1241)           |
| `vllm/v1/worker/gpu/model_runner.py`          | `_dummy_run`                       | 914-1013        | 只服务 profiling/warmup——capture 不走它                |
| `vllm/v1/worker/gpu/cudagraph_utils.py`       | `CudaGraphManager`                 | 154;dispatch:561-605 | 图字典、共享 pool；落桶查表；NONE 兜底            |
| `vllm/v1/worker/gpu/cudagraph_utils.py`       | `ModelCudaGraphManager`            | 641;factory:743-857 | 模型特化；输入静态化切片(759-771)               |
| `vllm/v1/worker/gpu/cudagraph_utils.py`       | `_init_candidates`                 | 289-429         | 两族尺寸网格枚举                                       |
| `vllm/v1/worker/gpu/dp_utils.py`              | `dispatch_cg_and_sync_dp`          | 232-369         | 每步中央裁判：eager 判定/落桶/DP 对齐                  |
| `vllm/v1/worker/gpu/model_runner.py`          | padding 与静态 buffer              | 1502 / 1682 / 2172-2180 | 落桶后补 padding；输入进固定 buffer            |
| `vllm/config/compilation.py`                  | `CUDAGraphMode`                    | 163-220         | 五值枚举；`decode_mode`/mixed_mode 拆分                   |
| `vllm/config/compilation.py`                  | capture 网格                       | 87-98 / 120-126 | prefill token 族 54 尺寸；decode 请求族 2 幂          |
| `vllm/config/compilation.py`                  | `splitting_ops`                    | 978-1003         | ~20 个图外算子；attention 全在图外                     |
| `vllm/config/compilation.py`                  | `resolve_cudagraph_mode_and_sizes` | 1580-1729       | 按 AttentionCGSupport 四级降级                         |
| `vllm/config/vllm.py`                         | `OptimizationLevel` 与四档字典     | 178-190 / 303-386 | 默认 O2;O3==O2；各档融合开关                        |
| `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` | capture           | 36-52 / 118-139 | draft 只支持 `FULL_DECODE_ONLY`；多步融合一张图          |
| `docs/design/debug_vllm_compile.md`           | 旗标二分表                         | 8-15            | 六个调试旗标从粗到细                                   |

## 重点与难点

- **重点**：piecewise 切分(`splitting_ops`)；`FULL_AND_PIECEWISE` 的含义（decode 整图/其余分段）；两族尺寸网格。
- **难点**：piecewise 编译与 piecewise 图的关系（同一套切分，一个产 kernel 一个消 launch）；输入静态化（为什么图内地址必须固定）；`enforce_eager` 的两层全关与覆盖顺序（`__post_init__` 硬置双 NONE，`-O` 档位默认只填 None 字段、显式 `-O` 也翻不回来）。
- **易错点**：默认档位说成 PIECEWISE（实为 -O2=`FULL_AND_PIECEWISE`）；去找 `--cudagraph-mode`/`vllm compile`/`VLLM_COMPILE_DEBUG`（都不存在）；以为 capture 走 dummy run。

## 参考材料

- 官方：`docs/design/cuda_graphs.md`、`torch_compile.md`、`optimization_levels.md`、`debug_vllm_compile.md`（四篇与 HEAD 基本一致；BatchDescriptor 原型略滞后）
- 博客：「Introduction to torch.compile and How It Works with vLLM」(blog.vllm.ai,2025-08-20)
- PyTorch:torch.cuda.graphs 官方文档（launch 开销机制的出处）
- 对照：nano-vllm 先修课第 8 课（`capture_cudagraph` 与 `run_model` 分支）
