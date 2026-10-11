# SolidAttention 精读报告（FAST'26）

> **勘误（2026-10-11，二次修订）**：初判「11 页 PDF 为 USENIX 文件本身缺页」**有误**——实测该文件可读出 p.77–82 全部内容（§8.4.3 收尾、§8.4.4–8.7 消融与敏感性、§9 相关工作、§10 结论），实为 17 页全本；「11 页」是 PDF 页数元数据误报（与同批 CacheSlide/Bidaw 同款）。本报告 §8.1–8.4.2 的评估内容有效；「截断/不可见」标注已失效，相关工作与结论见原文（精读文 solidattention.md 已按全本修订）。

# SolidAttention 精读报告

**论文**：SolidAttention: Low-Latency SSD-based Serving on Memory-Constrained PCs（FAST'26，上海交大 IPADS：Xinrui Zheng, Dongliang Wei, Jianxiang Gao, Yixin Song, Zeyu Mi, Haibo Chen）
**版本说明**：本 PDF 共 11 页（论文页 67–76），在 §8.4.3 中途截断，Figure 17b 之后内容、相关工作与结论章节**不可见**。

## 一、问题与动机

AI PC 本地部署 LLM 受内存制约：市售 PC 多为 8–16 GB DRAM + iGPU 或 6–8 GB VRAM（§1 引市场分析 [17,37,47]）；而 128k 上下文已成默认配置，即便 8B 模型仅 KV cache 就需超 16 GB，是权重的 4 倍以上（§1；Figure 1a，Llama-3.1-8B INT4）。llama.cpp/Ollama「整个 KV cache 驻留内存」的假设失效。两条既有出路均不佳：KV INT4 量化明显掉精度（§1）；FlexGen 类 SSD offload 是吞吐导向，靠 batching/zigzag 调度用其他请求的计算隐藏 I/O（§2.3 Figure 2），而本地场景请求并发低、batch 小，计算远快于 I/O，无法隐藏 SSD 延迟——加载 1k-token KV cache（128 MB）约需 40ms，几乎占一个 decode step 的一半时间（§2.3）。

## 二、核心洞察

**根因是稀疏注意力计算与 SSD 特性的根本冲突**：SSD 只在粗粒度顺序操作下达到最优性能，而动态稀疏注意力产生大量细粒度随机 I/O；既有工作把稀疏算法与存储管理当孤立问题处理（§1）。解法是**协同设计**：对齐数据访问粒度 + 策略性编排 computation–I/O overlap。三个支撑观察使方案成立：(1) K 与 V 同 shape，可在 token 粒度交错（interleave），传输单元翻倍、I/O 次数减半且每块 token 数不变、不掉精度（§4，Figure 7/8）；(2) 相邻迭代块选择约 **81% 相似**（§5，Figure 9，1k context budget，LongBench 多模型），投机预取有依据；(3) 注意力对 KV pair 全局顺序无约束，预取错了无需重排，直接原位覆写即可（§5.2）。

## 三、系统设计

- **KV Consolidator**（§4）：K/V 按 token 交错成 k/v pair 数据单元；模型初始化时把 K、V 投影权重预拼接，一次矩阵乘生成交错输出（Figure 8），kernel 以固定 stride 2H 逻辑分离 K/V，无需物理重排，延迟开销 ≤2%（§4）。代价：需定制 kernel 与改推理框架。
- **Speculative Prefetcher**（§5）：init blocks（初始窗口）与 local blocks（滑动局部窗口）确定性预载入 GPU；selected blocks（top-k）按上一迭代历史投机预取。错误预取块被新块直接覆写（无需额外显存与重排），缺失块补读（Figure 10）。
- **SSD-aware Scheduler**（§6）：attention 分解为 microtasks（q_proj/kv_proj/store/prefetch/select/load/attention），建 DAG 找关键路径，非关键 I/O 尽早发起；按 LST（最晚开始时间）优先调度以消除临界路径瓶颈（Algorithm 1）。同步点复用：非关键 store 与后续 prefetch 共享注意力完成后的同一同步点；不同块类型同步点合并（§6.2）。iGPU 统一内存可进一步省掉显式同步。
- **实现**（§7）：llama.cpp + liburing，约 25k 行 C++/CUDA（12k GPU kernel）；2 个专用核（I/O、SSD-GPU 协调）；32KB 每层写缓冲抗 SSD 磨损；block size 32；context budget：<4k 输入取 25%，否则 1k，一半给 init/local、一半给 selected。

## 四、评估

- **配置**（§8.1）：CUDA 机 = Ultra 9 185H + RTX 4070 Laptop 8 GB；SYCL 机 = Ultra 7 255H + Arc 140T iGPU；均 64 GB DDR5 + 1TB Samsung 990 PRO PCIe 4.0，DRAM 限 16 GB。模型：Llama-3.1-8B / Llama-3.2-3B / Qwen-2.5-7B，权重 INT4、KV FP16，batch=1，输出上限 512。基线：Offload（全量 offload）、Offload+Sparse（InfLLM）、FlexGen（扩展支持 Llama/Qwen，稀疏比 10%）。
- **吞吐**（§8.2，Figure 12/13）：128k 输入对 Offload+Sparse 加速 **2.8×/3.1×/2.4×**（CUDA 三模型）、SYCL 最高 **2.1×/2.5×/1.9×**；对 FlexGen 16k 时最高 **58.9×**（FlexGen >16k CUDA OOM）。
- **内存**（Figure 14）：KV cache 内存占用降最高 **61.9×/62.0×/61.9×**；Qwen2.5-14B INT4（SYCL）降 **98%**，128k 下仍比 Offload+Sparse 快 **1.7×**。
- **精度**（Table 1）：平均分 Origin 57.82/65.76/71.39 vs Ours 57.67/65.93/70.76，而 KV INT4 量化为 54.70/62.31/**18.63**（Qwen 崩溃）。
- **消融**：预取降低阻塞延迟最高 SYCL **3.1×**、CUDA **3.9×**（Figure 16）；K/V 交错降注意力延迟最高 **22%**（Figure 17a）；对比全内存 InDRAM+sparse 吞吐损失 ≤**11%**（Figure 15）。

## 五、局限与适用边界

论文承认：14B 模型 FFN 计算强度高会稀释收益（§8.2）；小 KV 模型（Qwen-2.5-7B）预取收益下降（§8.4.2）；KV 频繁写有写放大（§7）；对比全内存方案有 ≤11% 吞吐损失。精读员注：(1) 论文声称精度与原模型相当，但 Table 1 中 LongBench 单项普遍略降（如 Llama-3.2-3B 40.10→38.25），block-wise 稀疏（每步只算约 1k token）对需全局精确注意力的任务本质上有损；(2) 81% 相似度意味着约 1/5 块仍需补读，若任务的选择模式逐层漂移剧烈，预取会退化；(3) 仅两台笔记本、单一 SSD 型号、batch=1，未覆盖更慢 SSD 与并发场景；(4) 11 页版无相关工作与结论章节，成本/能耗分析论文未覆盖。

## 六、关联

与 KV 生态（精读员注，11 页版无 related work 章节）：FlexGen/InfLLM 是其直接对照与继承对象——采用 InfLLM 的代表向量块选择与「sink+local」分类（对应 StreamingLLM 的 attention sink）；FlexGen 的吞吐导向 zigzag 思路被论证不适配 batch=1。与 vLLM PagedAttention 同以「块」为管理单位，但目的不同：PagedAttention 解显存碎片，SolidAttention 的块是 SSD 传输粒度。与 Mooncake/LMCache 方向相反：后者面向集群、以吞吐/命中率/复用为目标、依赖请求级并发；SolidAttention 证明低并发端侧必须改走「稀疏算法 × 存储粒度 × 调度时序」协同设计路线，这对端侧 KV 分层存储（如 LMCache 的本地 tier）是有参照价值的设计约束。同目录另有 hao、hu-shipeng、liu-yang 三篇 FAST'26 论文（未读），与本文的关联论文未覆盖。
