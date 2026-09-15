# 模型部署实战

从"模型发布"到"可用服务"，中间还隔着并行策略选择、硬件适配、SLO 验证、量化精度取舍等一系列工程决策。同一个 DeepSeek-V3 模型，在 16 卡 H20 上能跑到 15,800+ tokens/s，在 32 卡 H20 上达成什么 SLO 目标才是合理预期？同一个 Qwen2-VL-7B 视觉多模态模型，从 NVIDIA 生态迁移到华为昇腾 MindIE，需要经过哪些版本匹配与算子适配？当前代的 B300 又有哪些反直觉的坑？本目录不讲泛泛的"部署指南"，而是给出三个**带 SLO 数字、带硬件型号、带实测数据** 的端到端参考方案，以及一个可复用的 SLO 验证脚本。

> 更多 SGLang 相关的大规模推理调优案例，请参见 [SGLang 推理引擎](../sglang/README.md)。

## 1. NVIDIA H20 集群：DeepSeek-V3 MoE

以 DeepSeek-V3（671B MoE，激活 37B）在 H20 集群上的推理部署为案例。参考腾讯太极团队 16 卡 H20 达成 15,800+ tokens/s 的实测数据，对照 vLLM 源码逐一分析 PD 分离、EPLB、DP 适配、MTP 加速及 FP8 量化等关键技术的实现状态。

- **[DeepSeek-V3 H20 推理优化：基于 vLLM 源码的深度分析](deepseek_v3_h20_vllm_deep_dive.md)**：对照太极团队四大技术方向（PD 分离、EP/EPLB、DP 适配、MTP 加速），逐一分析 vLLM 源码中的对应实现——KV Connector 框架、EPLB 三层打包算法、Batched DP MoE 同步机制、DeepSeekMultiTokenPredictor 结构，以及 FP8 量化路径与 w4a8c8 的差距
- **[`slo_calc_v2.py`](slo_calc_v2.py)**：SLO 目标可达成性验证脚本——输入并发 / 上下文 / GPU 数量，输出 TTFT、TPOT、吞吐的预期达成情况

## 2. 华为昇腾平台：Qwen2-VL-7B 视觉多模态

- **[Qwen2-VL-7B-Instruct 昇腾部署指南](qwen2_vl_7b_huawei.md)**：Atlas 800I A2（32G / 64G）硬件 + MindIE 1.0.0+ / CANN 8.0.RC1+ / OpenEuler 24.03 LTS 的完整软件栈，覆盖视觉 token 压缩、多分辨率图像输入、超 20 分钟视频理解等多模态推理的国产硬件适配要点

## 3. NVIDIA B300：部署配方与 KV Cache 实践

B300（Blackwell Ultra，SM103）是当前代的旗舰，但它的部署实践有两个反直觉之处：**世代红利只给了 FP4**——FP8/BF16 相对 B200 零提升，而 INT8 因为 PTX 未授权 sm_103a 而完全不可用；**且 MLA 模型上加 TP 会复制而非切分 KV**。这篇不讨论可能性，只整理 vLLM 与 SGLang 官方手册推荐的配置、调优判据和现成配方，每条推荐都标注出处。

- **[B300 上的模型部署与 KV Cache：官方手册最佳实践](b300-deployment-and-kv-cache.md)**：Kimi-K3 与 DeepSeek-V4 的完整启动命令、SM103 独立 target 等三道门槛、vLLM 的 `-O0`~`-O3` 优化等级与 `2 + N` 物理核公式、Deep PP 为什么用 `--tp-size 1`、SGLang 的 KV 量化官方精度对照表、HiCache 的 canonical 参数与布局兼容性、两家的调优判据（`available_gpu_mem` / `token usage` / `num_preemptions`），以及 NVFP4 KV 为何不能进生产
- 同篇 §三 **扩展账：Kimi-K3 从 16 卡到 64 卡**：按 cookbook 预设源码与 SGLang 计算器逐轴核算——`attnTP ≡ 8` 的形状锁定、每卡权重 186→25 GB 的降落曲线、**每卡 KV 与 state 三档一个字节不变**（13,824 B/token、110.8 MiB/请求）、MoE all-to-all 跨节点扇出 2.4× 的账单
- 相关原理：[从 H200 到 Blackwell 的飞跃](../vllm/hardware_optimization/deepseek_blackwell_wide_ep.md)（WideEP / NVFP4 / Weight Offloading v2）

## 4. 方法论抽象

两份案例共同展示了一条端到端部署与运维方法论：

1. **SLO 目标量化**：明确并发数、上下文长度、TTFT/TPOT/吞吐的 P50/P95/P99 分位数
2. **硬件与并行策略匹配**：依据显存总量（见 [`memory_calc/`](../memory_calc/README.md)）与模型结构选 TP/EP/PP 组合
3. **实测基准外推**：优先使用同规模 / 同模型的公开实测数据（如腾讯太极 16 卡 H20）做理论外推，而非凭理论峰值估算
4. **SLO 可达成性验证**：用 [`slo_calc_v2.py`](slo_calc_v2.py) 这类脚本在部署前做一次"纸上验证"，避免投入硬件后才发现目标不可达
5. **国产化 / 合规场景**：参考华为昇腾 MindIE 适配路径做软件栈与算子兼容性评估

## 5. 精度验证与排错

上线前的端到端验证常被「输出几乎一致、只有一两个 token 不同」卡住：这是硬件浮点噪声，还是缓存链路丢了数据？本小节提供一套定量的判别方法。

- **[输出差了一点点？用 logprobs 分清「噪声」还是「bug」](logprobs-precision-diagnosis.md)**：基于真实排查故事（KV Cache 端到端验证，Falcon-H1 7B），用 `logprobs` 参数的三个定量判别（gap、跨路径差、全局 vs 局部）把「采样噪声 vs 数据 bug」分开，再以写/介质/读三对照把根因收敛到一行代码

> 相关阅读：并行策略理论参见 [核心推理优化技术深度解析](../reference_design/03-核心推理优化技术深度解析.md)；SLO 指标定义参见 [性能评估指标体系](../reference_design/05-性能评估指标体系.md)。
