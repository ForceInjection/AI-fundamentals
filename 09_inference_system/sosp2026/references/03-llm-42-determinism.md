# 03 · LLM-42：用验证投机实现确定性 LLM 推理

> SOSP 2026 论文精读笔记。来源：`../papers/03_llm-42-determinism.pdf`（arXiv:2601.17768v2 预印本，2026-11-14 下载）。文中数字均为论文口径。

## 一句话

快路径照常跑动态 batching，验证器按固定形状重放候选 token，不一致就回滚重算，把确定性做成按请求计费的调度性质。

## 问题与背景

同一 prompt 跑两次，输出可能不同。根源是浮点加法不结合与动态 batching 的组合：batch size 一变，GPU kernel 的并行化跟着变，同一逻辑运算走了不同的归约顺序（§2.2 p.3）。自回归解码放大漂移：在 ShareGPT 350 条请求上对比 batch size 1 与 6 qps 动态 batching，多数请求初始几百个 token 一致，一旦有一个 token 分歧，第二一致段的长度接近零（§3 p.5，Figure 6）。

确定性用于隔离 batching bug、RL 奖励稳定、集成测试、科学复现（p.1）。禁动态 batching 重创吞吐；batch-invariant kernel（§2.3 p.3）要求所有形状共用一个归约策略，GEMM 慢 63%，RMSNorm 最差慢 70 倍（Figure 4）。惩罚全局生效：11 个请求里 1 个要确定性，整批进慢 kernel，吞吐掉 56%（Figure 5 p.4）。

## 核心设计

**四条观察**（§3 p.4-6）打底。O1：序列处于一致状态时，下一个 token 大概率仍一致。O2：多数 GPU kernel 用 shape-consistent reduction，同形状同策略；batch size 固定时，输入元素的输出与其在 batch 中的位置无关（Figure 7）。前向 kernel 不用原子归约是前提（p.5 脚注）。O3：确定性只要求位置一致，不同 token 位置可用不同归约策略。O4：评测、审计、回归测试才需要确定性，全局开关是过度设计。

**DVR 协议**（§4.2 p.6-7，Figure 8）：decode-verify-rollback。解码照常走快路径产出候选，验证器按固定窗口 T 重放，序列尾用 dummy token 补齐保证形状恒定。成功则窗口全部提交，验证器生成的下一个 token 一并收下；失败则只提交到最后一个匹配 token，其余丢弃，KV cache 截断继续解码。两个细节：解码阶段写入的 KV 即使验证通过也不可信，要用验证 pass 的 KV 覆盖（p.7）；每次验证至少产出一个新的一致 token。与投机解码的差别（Table 1 p.2）：无近似、无草稿模型、深度可到 32-64 token。

**prefill 不验证**（§4.1 p.6）：由 O3，prefill 只要避免随意凑批就天然确定，验证器只管 decode 产出的 token。**按请求开关**（O4）：is_deterministic 默认 False。11 个请求中 1 个要确定性时，解码吞吐 911 tok/s，是 SGLang 确定性模式的 2.2 倍，比非确定性最优只差 3%。

**grouped verification**（§4.3 p.7-8）：验证 pass 内存受限，窗口小则每 token 最高 0.75 ms，窗口 512 降到 0.05 ms；窗口大则回滚贵，window 256 重算开销 46.41%，window 32 只有 6.81%（Figure 9）。折中是多请求各出小窗口拼成一次大 batch 验证，兼得两边的优点。

**算子与采样配置**（§4.4 p.8-9）：attention 用 FlashAttention-3，验证时固定 num_splits=1；AllReduce 优先 multimem 版，否则固定配置的 tree AllReduce；非 greedy 用 multinomial_with_seed 替换有 Gumbel 噪声 bug 的 torch.multinomial。GEMM kernel 全部复用，最重的工程负担被绕开。

## 效果与代价

环境：Llama-3.1-8B-Instruct，4 张 H100 PCIe，SGLang v0.5.3rc0（§5 p.9）。

离线吞吐（§5.1 p.9）：SGLang-Deterministic 慢 24% 到 36%；LLM-42 在全部八种负载上高于确定性基线，最差只比非确定性最优慢 6%；对确定性基线最高快 41%。回滚与重算（Table 4 p.10）：最坏 ArXiv 100% 时 4096 请求 3351 次回滚；重算占比最高 10.97%。

在线延迟（§5.2 p.10-11）：12 QPS 下确定性基线 P50 4.64 s、P99 28 s，LLM-42 在 2% 占比时 P50 2.21 s，只比非确定性基线（2.15 s）高 3%；短板在高负载加全量确定性，延迟反超基线，归因于验证 pass 全局暂停与 prefill 未凑批（p.10）。消融（§5.3 p.11-12）：单请求验证窗口非单调，window 16 的 P99 达 615 s；grouped 最优每步约 256 token，P99 34-35 s。

局限（p.12）：验证 pass 拖慢所有在途请求；prefix sharing 不可用；未与投机解码集成；多 GPU 性能未充分评估。

## 总评

这篇论文把确定性从 kernel 工程问题挪回了调度问题。batch-invariant 的成本是全局固定税，kernel 还要跟着硬件重写；LLM-42 是按需税，只对声明需要确定性的流量付验证与重算，GEMM 等大头原样保留。O3 是理论上的放松，O1 是开销可控的经验保证，两者合起来验证才便宜。对评测复现、RL 奖励稳定、合规审计，价值在于用原版 SGLang kernel 拿到确定性并保住大部分吞吐。

没回答的也明确：prefix sharing 缺失对多轮对话是实打实的代价，未量化；全局暂停的频率与时长没有数据；性能限于单节点 4 卡 PCIe 的 8B 模型。折扣清单：吞吐优势多在 10%-50% 占比区间取得，100% 时收窄甚至转负；温度大于 0 依赖自建采样器与 seed 传递。

## 疑点与待核

1. §6 p.12 提到的 Llama-4B-Instruct-2007 型号名含混，正确性验证的模型清单需对照原稿。
2. 验证 pass 的触发节奏与全局暂停时长没有量化数据。
3. 温度大于 0 时 seed 如何传递、API 是否暴露，论文未展开。
4. O1 的接受率无单独统计，整窗失配的频率未给出。
5. Figure 5 场景下，同批不混两类请求的调度中间方案未被讨论。
