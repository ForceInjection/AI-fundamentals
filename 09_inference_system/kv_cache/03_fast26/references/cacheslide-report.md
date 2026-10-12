# CacheSlide 精读报告（FAST'26）

**论文**:CacheSlide: Unlocking Cross Position-Aware KV Cache Reuse for Accelerating LLM Serving。Yang Liu、Yunfei Gu(共同一作)等,上海交大 + 浪潮济南 + 北大 + 华为云;通讯 <gu.yunfei@sjtu.edu.cn> / <wuct@cs.sjtu.edu.cn>。FAST'26 论文集 pp.83–99,带 Artifact Evaluated–Available 徽章。

## 一、问题与动机

论文把智能体 prompt 抽象为 **System prompt + Updated prompt + Fixed prompt** 三段(§2.1,p.84),并归纳三类范式:CoT 型(Reflexion、Self-Refine,更新段=逐步累积的推理记录)、记忆型(MemGPT,工作缓冲区经 memory API 修改、FIFO 后缀不变)、工具增强型(SWE-Agent/AutoGen,函数规格动态注入、历史消息在后缀累积)(§2.2)。基于 GitHub 日志分析与实验验证,各 agent 输入中可复用片段占比都很高,Figure 3(p.85)同时给出占比方差:Reflexion 0.025、Long-term memory 0.070、SWE-Agent 0.080、MemGPT 0.076、Self-Refine 0.010、AutoGPT 0.021、Function Calling 0.026——方差大说明更新段长度不可预测,这是后文否定 window padding 的伏笔。

核心问题是 **RPDC(Relative-Position-Dependent Caching)**:可复用片段「保持一致相对顺序、但绝对位置漂移」(§1,p.83)。正式定义在 §3.3(p.88):一类 KV 复用场景,上下文由多段组成,**相对顺序必须固定(任意重排会带来严重退化),而可复用段之间的夹层内容持续更新**。此时既要保序(PDC 的约束),又要容忍位移(PIC 的灵活性)。

## 二、核心洞察

现有两类方法在 RPDC 下双输(§3.1):

- **PDC**(ContextCache、PromptCache)只许在预定位置(通常前缀)复用。Figure 2(p.85)证明「重排换位」不可行:prompt 头部吸引更多注意力([34,71],MemGPT 特意把 Working Window 放头部);SWE-Agent 的更新槽与不变段有运行时数据依赖(Updated Slot 1 产出被 Immutable 1 消费的参数),换位即破坏程序语义。PromptCache 逐段隔离预计算、不复原跨段注意力,还导致 KV 版本膨胀。
- **PIC**(CacheBlend、EPIC)把复用段位置清零,产生 **PMKD**(定义于 §3.2.1,p.86:同一文本因位置编码不同导致的 KV 相似度偏差);靠预选 token 重算补救,但多头注意力各头关注的 token 不同、输出在 decode 阶段才融合,「prefill 期预选无法保证稳定精度」(§3.1.2)。系统层面还有:主流框架(vLLM/SGLang/TensorRT-LLM)逐层串行 KV 读写形成层内 load-before-write 锁、无法重叠(§3.1.3);LRU 淘汰不感知脏页,部分更新页驱逐带来随机写与写放大 WAF。

**量化洞察**(§3.2,p.87):定义 CKSim(逐头 K/V 余弦相似度均值),用 MemGPT+ShareGPT 构造 MemHistory,前缀窗口 i∈{100,200,300,…} 步长 100 做部分 prefill 后对比。结果(Figure 4a):位移 0→1000 时 RoPE 的 CKSim 下降 >90%,CoPE 仅降 ~28%。window padding(1K/2K/3K)在多轮推理后 F1 比基线低 >78.1%(Figure 4b)。Figure 5 显示 CoPE 式「相邻 token 共享位置索引」使位置映射斜率更缓、Apox 更小(例:固定段起点 (10,21)→(9,20))。由此得出关键观察:**保住相对顺序 + 低位置敏感编码,固定段内部与固定段之间的注意力可 near-lossless 复用,只有固定段↔更新段的跨注意力需要小规模重算恢复**(§3.3)。

## 三、系统设计

CacheSlide 三组件(Figure 6,p.88;§4.1):

1. **CCPE(Chunked Contextual Position Encoding)**:位置不敏感编码,「提高缓存 KV 与重算 KV 的相似度」(摘要);面向「可变 prompt 分段」的 RPDC 场景。编码器按「学习到的任务特定模式」把输入切成 reused/recomputed chunk,reused chunk 经 hash map 直接取 KV(步骤①)。
2. **WCA(Weighted Correction Attention)**:prefill 期只在 reused chunk 中**选极少数 token 重算**,并用学习到的权重融合新 KV 与缓存 KV、恢复跨注意力(步骤②;Figure 1(c) 标为 Compute local attention + Reuse Attention)。具体选哪些 token、每块算多少:只对复用块中约 12% 的 token 重算注意力(对比 CacheBlend 的 18%、EPIC 的每分段 64 token),用偏差加权融合恢复跨注意力,并每 4 层做一次 CKSim 门控——已收敛的 token 移出重算集合(§4.2–4.3)。
3. **SLIDE**(Spill awareness、Load–write decoupling(Intra-layer)、Dirty-page Evocation,p.84 贡献列表):③ 层内读写解耦提升加载/更新并发;④ 重定位 reused chunk 中被选中的 KV;⑤ 内存压力时优先把**不含选中 token 的干净页**溢写 SSD;⑥ decode 期把 KV 覆写回含选中 token 的页,削减存储足迹——整体指向消除小随机写与脏页驱逐开销。

实现上扩展 vLLM 0.8.5 的 KV cache 管理部分,加入 CCPE 与 WCA(§5)。

## 四、评估

**评估(§5)**:vLLM 0.8.5 扩展,Mistral-7B/MPT-30B/Llama-3-70B 三模型 × Reflexion/MemGPT/SWE-Agent 三 agent,落在精度–TTFT 的 Pareto frontier:对比 ContextCache TTFT 降 2.4–3.3× 且精度损失可忽略;对比 CacheBlend 再降 1.21–2.11×;对比 PromptCache 降 1.97–2.88× 且 accuracy 提升 1.41–3.95×。总口径延迟降 3.11–4.3×、吞吐升 3.5–5.8×、写放大降 3.11–3.62×;吞吐稳定性(标准差)比 CacheBlend/EPIC 低 58.6–77.4%。注意:动机实验(CKSim、window padding、Figures 3–5)用的是 MemGPT+ShareGPT,属 motivation 实验,不能当作端到端评估证据(精读员注)。

## 五、局限与适用边界

论文承认的:reused chunk 内小规模重算「继承了与 PIC 类似的系统级挑战」(§3.3 末段);window padding 单独使用不可行(更新段长度大且不可预测,[18,54,6])。
我发现的(精读员注):(1) CCPE「基于学习到的任务特定模式」切分 chunk——学习开销、跨 agent 泛化未交代;实现上经 LoRA 继续预训练学得(backbone 不动、可切回 RoPE),新任务类别需重做,这是最大适用边界。(2) RPDC 之外的收益不明:纯前缀重复场景 PDC(如 vLLM APC)零重算已近最优;真正乱序插入(RAG 多文档任意排列)不满足「相对顺序固定」前提,可能落到 PIC 原路——论文未验证。(3) 精度损失可忽略已有 Pareto 实验支撑(相对 PromptCache accuracy 反升 1.41–3.95×)。(4) PMKD 量化仅在 MemGPT/ShareGPT/RoPE-vs-CoPE 一个设置上做过。

## 六、关联

KV 复用谱系可排成:位置固定(PDC:ContextCache、PromptCache,以及 vLLM APC/SGLang radix cache 这类前缀缓存——同族,精读员注)→ 位置无关(PIC:CacheBlend、EPIC,靠选择性重算恢复)→ 相对位置保持(CacheSlide:改编码本身把漂移抹平,把重算量从启发式 18%/64-token 压到「极少数」)。论文明确把 vLLM/SGLang/TensorRT-LLM 点名为层内串行读写的问题来源,并直接扩展 vLLM 实现。SLIDE 处理的 SSD 溢写、脏页感知淘汰、读写解耦,与 LMCache 的 KV 分层下沉、Mooncake 的 KVCache 中心化池化属同一问题域(精读员注)——可视为把「KV 分层存储」从容量优化推进到为位置漂移复用服务的并行 I/O 设计。
