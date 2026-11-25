# 第 4 讲 · 注意力后端：从 if-else 到后端矩阵

> 120 分钟 | 基线：nano-vllm `b9dd55a`；vLLM `2f5d8a2b1`（v0.34.1rc0-84,2026-11-25 核对）

本讲回答一个核心问题：**同一个模型部署到不同硬件，引擎怎么自动决定用哪个注意力 kernel？强制指定时，它又凭什么拒绝你？**

知识链：`Attention` 层 `__init__` → `selector.get_attn_backend`（`selector.py:111`）→ `platform.get_attn_backend_cls`（`cuda.py:462`）→ arch×算力优先级列表（`cuda.py:82-179`）→ `validate_configuration` 统一过滤（`backend.py:296-389`，收集每个被拒后端的理由）→ 取优先级最小者 → registry 按 qualname 拿类。执行期：`GPUModelRunner` 构建 `CommonAttentionMetadata` → `set_forward_context` → `Attention.forward` → `impl.forward`。

## 一、nano 已学形态（回顾）

先修课第 7 课你看过的 nano「后端选择」（`nanovllm/layers/attention.py`）：

- 写入：`store_kvcache`（Triton kernel，`:10-30`，按 `slot_mapping` 写，`slot == -1` 跳过）；
- **prefill 分支**（`:64-70`）：`flash_attn_varlen_func`（变长 batch）；
- **decode 分支**（`:71-75`）：`flash_attn_with_kvcache`（增量读 cache）。

全部选择逻辑就是 `context.is_prefill` 的一处 if-else，而且 kernel 固定用 FlashAttention——没有第二个选项。生产版的三个新约束把它逼成了矩阵：模型架构多样化(GQA/MLA/Mamba)、硬件多样化（NVIDIA 各代/ROCm/CPU）、特性组合多样化（sliding window、稀疏、batch invariance）。

## 二、演进：if-else 短路链 → 优先级列表 + 统一校验

旧版（及大量旧教程）的选择是 if-else 短路链：按平台一个个试。HEAD 的 CUDA 选择（`vllm/platforms/cuda.py`）已经是另一种形态：

1. 显式指定（`--attention-backend`）→ 走校验，不合法直接报错**并列出每个后端的拒绝理由**（`cuda.py:472-488`）；
2. 否则 `_get_backend_priorities`（`cuda.py:82-179`，带缓存）按 **模型架构 × 算力代际** 生成一个**优先级列表**；
3. 逐个跑 `validate_configuration`（`vllm/v1/attention/backend.py:296-389`）——一个统一的过滤器，十余项特性逐一检查，**被拒后端的理由全部保留**；
4. 取列表中第一个通过者（`cuda.py:495-527`）；全被拒 → `ValueError: No valid attention backend found...` 附全部理由（`:513-517`）。

「if-else 变矩阵」的本质：选择从**控制流**变成了**数据**（列表）+ **谓词**（校验函数）。加一个新后端 = 注册 + 覆盖几个 `supports_*` 方法，不改选择逻辑。

## 三、后端矩阵：你需要认识的那些

`vllm/v1/attention/backends/` 下有 33 个 py 文件（`mla/` 子目录 25 个）。不用都记，分四组认识：

| 组                 | 后端（注册名）                                                                                                                                | 一句话                                                                                                                  |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 标准通用           | FlashAttentionBackend(`FLASH_ATTN`)/ FlashInferBackend(FLASHINFER)/ TritonAttentionBackend(`TRITON_ATTN`)                                         | GQA 三主力；FA 版本（SM90 默认 FA3、SM100+ 默认 FA4）是 `FLASH_ATTN` 内部的 `flash_attn_version` 选项，**不是独立后端名** |
| MLA 专用（`mla/`） | `FLASHINFER_MLA` / FLASHMLA / `TRITON_MLA` / `CUTLASS_MLA` / `FLASH_ATTN_MLA` + 稀疏系(`FLASHMLA_SPARSE`、`FLASHINFER_MLA_SPARSE`、`FLASH_ATTN_MLA_SPARSE`) | 与标准后端**互斥**（见下），按算力代际细分                                                                              |
| Mamba 族           | MambaAttentionBackendEnum:MAMBA1/MAMBA2/GDN_ATTN/LINEAR/SHORT_CONV                                                                            | **独立的第二张注册表**（`registry.py:260-354`），不进上面的决策树                                                       |
| 平台件             | ROCm 系 6 个 / CPU / B12X(SM120)/ `HPC_ATTN`（腾讯 Hopper）等                                                                                   | 按平台各自有优先级表                                                                                                    |

另有动态生成的 composite 类（`TRITON_FLASH_ATTN` 只在 Hopper+图文混合批出现、`TRITON_FLASHINFER` 只在 Blackwell）——工厂合成，源码里搜不到 `class`，别找。

## 四、决策树五层（本讲重点）

1. **显式指定层**：`--attention-backend`（或 Python 侧 `AttentionConfig.backend`）→ `AttentionBackendEnum[名.upper()]`（`config/attention.py:273-285`）。名字不存在 → enum 元类报「Unknown attention backend」并列出合法选项（`registry.py:18-31`）。
2. **平台静态层**：CUDA/ROCm/CPU/XPU 各有 `get_attn_backend_cls` 入口（`cuda.py:462`、`rocm.py:770`、`cpu.py:148`、`xpu.py:179`）。
3. **arch×算力优先级层**（`cuda.py:82-179`）：
   - **MLA 分支**：SM100 fp8 KV → [`FLASHINFER_MLA_SPARSE`, `FLASHMLA_SPARSE`];SM90 → [`FLASH_ATTN_MLA`, FLASHMLA, `FLASHINFER_MLA`, `TRITON_MLA`, …];dense MLA 序 `FLASHINFER_MLA` → … → `TRITON_MLA`（`:120-131`）
   - **非 MLA 分支**：SM100 因果 → [FLASHINFER, `FLASH_ATTN`, `TRITON_ATTN`, …]；其余 → [`FLASH_ATTN`, FLASHINFER, …]（`:158-179`）
4. **特性过滤层**：`validate_configuration`（`backend.py:296-389`）逐项查：`head_size`、dtype、`kv_cache_dtype`、`block_size`、**MLA 匹配（`:332-336`，`use_mla != cls.is_mla()` 即拒）**、**稀疏匹配（`:339-343`）**、cc、**sliding window（`:350-351`）**、non-causal、batch-invariant、`kv_connector`、pcp/dcp……每项都是后端类上的 `supports_*` 方法。
5. **兜底层**：全被拒 → ValueError 附每个后端的拒绝理由（`cuda.py:512-516`）——这份理由列表本身就是排错工具。

两个走查例（背下来，面试级）：

- **Qwen3(dense GQA)**：非 MLA 分支。H100(SM90)→ `FLASH_ATTN` 第一且校验通过（SM90 默认 FA3）；B200(SM100)→ FLASHINFER 优先。
- **DeepSeek V3.2(MLA+DSA)**:sparse 标记让 dense MLA 成员全被 `:339-343` 拒掉。H100 → `FLASH_ATTN_MLA_SPARSE`（仅 cc==9 可用）；B200 bf16 → `FLASHMLA_SPARSE`；fp8 KV → `FLASHINFER_MLA_SPARSE`。另外它的 **indexer 后端由模型直连**（`DeepseekV32IndexerBackend`，`mla/indexer.py:220`），不走本决策树；MLA 的 prefill 与 decode 还是**两个选择器**（`mla/prefill/selector.py:46`）。

## 五、后端 × 特性矩阵（CUDA，每格有行号依据）

| 后端        | full GQA                               | MLA                            | sliding window                                                    | Mamba(GDN)                             |
| ----------- | -------------------------------------- | ------------------------------ | ----------------------------------------------------------------- | -------------------------------------- |
| `FLASH_ATTN`  | ✓ cc≥8.0（`flash_attn.py:551-552`）    | ✗（`backend.py:332-336` 互斥） | ✓（`flash_attn.py:483-484`）                                      | ✗（`backend.py:398`）                  |
| FLASHINFER  | ✓ SM80-12.1（`flashinfer.py:835-842`） | ✗（走 `FLASHINFER_MLA`）         | ✓（`:761-762`）                                                   | ✗                                      |
| `TRITON_ATTN` | ✓ 任意 cc（`triton_attn.py:405-406`）  | ✗（走 `TRITON_MLA`）             | ✓（`:363-364`）                                                   | ✗                                      |
| FLASHMLA    | ✗（非 MLA 被拒）                       | ✓                              | ✗（无 override；仅 `TRITON_MLA` 支持滑窗，`triton_mla.py:182-184`） | ✗                                      |
| `GDN_ATTN`    | —（独立注册表）                        | —                              | —                                                                 | ✓（`gdn_attn.py:103-104` `is_ssm=True`） |

矩阵的两个结构性结论：**MLA 与标准后端是互斥集合**（`is_mla()` 严格相等才通过）；**Mamba/GDN 不进同一棵决策树**（经 `MambaLayer.get_attn_backend` → `get_mamba_attn_backend`（`selector.py:236-255`）按 `mamba_type` 直取）。Gemma-2 的 sliding window 来自模型 config（`models/gemma2.py:156`），CUDA 上 `FLASH_ATTN`/FLASHINFER 都支持，自动选择不受影响。

## 六、V0/旧教程 vs HEAD

| 旧说法                                      | HEAD 事实                                                                                                        |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `VLLM_ATTENTION_BACKEND=...` 环境变量       | **已删除**(PR #32812)；现行 `--attention-backend` / `-ac.backend`，与 `attention_config.backend` 互斥            |
| 「FA3 后端」「FA4 后端」                    | FA 版本是 `FLASH_ATTN` 内的选项（`flash_attn_version`），不是后端名                                                |
| 「`get_attn_backend` 在 `backends/utils.py`」 | 在 `vllm/v1/attention/selector.py:111`；utils 只装 metadata 工具                                                 |
| 「MLA 用 `VLLM_MLA_BACKEND` 选」            | 预填/解码分离：`mla/prefill/selector.py:46` 管预填，主后端管解码                                                 |
| `AttentionMetadata`                         | 更名 `CommonAttentionMetadata`（`backend.py:409-410`）+ 各后端 builder 产物                                      |
| 文档 `attention_backends.md`                | 由 `validate_configuration` **自动生成**（gen 脚本），与 HEAD 强绑定、无滞后——本课程目前发现的唯一零滞后设计文档 |

## 七、实验与产出

见 [hands-on-exercise.md](hands-on-exercise.md)：实验 1 `--attention-backend` 强切对照（含非法值两种报错）；实验 2 决策树复算（两种模型 × 两种假想硬件，手推优先级列表再与日志对答案）；实验 3（可选）Gemma-2 滑窗观察。课后作业见 [homework.md](homework.md)。

## 八、边界

| 内容                             | 本讲                                   | 去处                                                                                                                                                                          |
| -------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FlashAttention / MLA kernel 数学 | 不展开（只讲选择）                     | 仓库 `09_inference_system/vllm/module_analysis/` 下的 `flashattention_deep_dive.md`、`attention_mha_mla_nsa.md`、`deepseek_attention_evolution_mla_to_csa_hca.md`（演进素材） |
| page table 怎么传给 kernel       | 只到 `block_table` 参数（第 3 讲已铺） | —                                                                                                                                                                             |
| 量化 attention（turboquant 等）  | 不展开                                 | —                                                                                                                                                                             |

## 九、前后衔接

第 3 讲留下的「块找到了，kernel 怎么读」在本讲闭环：选择发生在模型层 `__init__`（一次），执行发生在每步 forward（经 `set_forward_context` 传元数据）。第 5 讲回到 Worker:attention 前后夹着的 CUDA Graph 与编译。
