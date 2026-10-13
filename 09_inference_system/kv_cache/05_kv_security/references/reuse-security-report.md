# D 线：KV Cache 复用 / 共享 / 安全类论文调研报告（2025–2026）

调研日期：2026-10-12。验证手段：arXiv API（摘要、comments 字段、版本历史）、arXiv HTML 全文（作者机构）、NeurIPS/NDSS/USENIX/SIGOPS 官方页面、ACM DL、GitHub。DBLP 被反爬拦截未用。所有 venue 与关键数字均注明来源；查不到的标「未验证」。**评级标准**：A = 顶会录用且数字扎实；B = 有录用或著名团队；否则淘汰。

**总体结论**：候选 5 篇中仅 1 篇有顶会正式录用（KVLink/NeurIPS 2025）；其余 4 篇均为 arXiv-only。「KV 安全」方向真正正式录用的论文在 NDSS（2025/2026 各一篇），见补充探索。

## 评级汇总

| #   | 论文                    | Venue（来源已核）                              | 团队                                 | 评级             |
| --- | ----------------------- | ---------------------------------------------- | ------------------------------------ | ---------------- |
| 1   | KVLink                  | **NeurIPS 2025 poster**（官方页核实）          | UCSB + Accenture        | **A**            |
| 2   | PrefillShare            | 未录用（arXiv comments 标 Preprint）           | NAVER Cloud + Toss + A2SYS           | 淘汰（观察名单） |
| 3   | PrefixWall              | 未录用（arXiv 2603.10726）                     | IMDEA Software + 马德里理工          | B                |
| 4   | CacheScout              | 未录用（arXiv 2608.14624）                     | UCSC/UW/UChicago（Junchen Jiang 组） | B                |
| 5   | PrfaaS                  | 未录用（arXiv 2604.15039）                     | Moonshot AI + 清华                   | B                |
| 6   | I Know What You Asked   | **NDSS 2025**（官方页 + DBLP）                 | 南方科技大学（SUSTech）+ ByteDance              | **A**            |
| 7   | Shadow in the Cache     | **NDSS 2026**（DOI 已核）                      | 浙大 + 华为                          | **A**            |
| 8   | Auditing Prompt Caching | **ICML 2025**（OpenReview 官方页）             | Stanford                             | **A**            |
| 9   | DroidSpeak（NSDI 论文集版）    | **NSDI 2026**（初版 arXiv:2411.02820）             | UChicago（Junchen Jiang 组）+ 微软   | **A**            |
| 10  | HKVQ                    | **SIGCOMM 2025 short**                         | UVA + Harvard                        | B（short paper） |
| 11  | ARK                     | **NAIC '26 workshop @ SIGCOMM 2026**（ACM DL） | UMass/Caltech 系                     | B（workshop 级） |
| 12  | Strata                  | **OSDI '26**（USENIX 议程页）                  | Stanford/NVIDIA                      | **A**            |
| 13  | RTT- or Bandwidth-Bound | **NAIC '25 workshop @ SIGCOMM 2025**（DOI）    | 中国移动研究院 + 华为                | B（workshop 级） |

1–5 为候选 5 篇（详述见下文 §1–§5），6–13 为补充探索的相关工作（详述见 §6）。

---

## 1. KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse（NeurIPS 2025 poster）

**录用验证**：**NeurIPS 2025 poster**（官方页面 [neurips.cc/virtual/2025/poster/116061](https://neurips.cc/virtual/2025/poster/116061)，含 slides 与 OpenReview 链接）。arXiv 2502.16002，v1 2025-02-21，v4（NeurIPS camera-ready）2025-11-10。

**作者验证**：UCSB + Accenture（Center for Advanced AI）：Jingbo Yang、Bairu Hou（UCSB，共同一作 "Equal contribution"，p.1）、Shiyu Chang（UCSB）、Wei Wei、Yujia Bao（Center for Advanced AI, Accenture；邮箱域 @accenture.com，p.1）。Microsoft 仅出现在致谢（算力经 "Accelerate Foundation Models Research program" 部分支持，p.10）。

**核心贡献与数字**：文档级 KV 预计算复用——每篇文档的 KV cache 独立编码后存储，推理时直接拼接复用。两个保真机制：(1) KV Positional Re-encoding（把缓存 KV 的位置编码调整到拼接后的全局位置）；(2) Link Token 训练（可训练连接 token 恢复不连续文档间的自注意力）。7 个数据集上 QA 准确率平均 +4%，TTFT 相对标准解码降 85–96%（v4 正文数字）。

**口径注意**：「GPU 内存省 98%、吞吐 8.5×」这两个数在 v4 全文与 v1 摘要中均无法溯源（2026-10-13 复核），不建议引用；v1 → v4 的实际口径变化是 TTFT "up to 90%" → "up to 96%"。引用以 v4 为准（[arXiv abs](https://arxiv.org/abs/2502.16002)、[GitHub](https://github.com/UCSB-NLP-Chang/KVLink)，48 star）。定位是 ML 会议的 RAG 效率论文，非 serving 系统论文；后续 CacheClip、C2KV 已跟进。81 引用。

**评级：A**（NeurIPS 2025 正式录用；UCSB + Accenture 团队，文档级 KV 预计算复用路线被 CacheClip、C2KV 等后续工作跟进）。

---

## 2. PrefillShare: A Shared Prefill Module for KV Reuse in Multi-LLM Disaggregated Serving（未录用，观察名单）

**录用验证**：**未录用**。arXiv 2602.12029，v1 2026-02-12，v2 2026-09-26；comments 明确写 **"Preprint. 22 pages, 8 figures"**，无任何录用注记（arXiv API）。DBLP/OpenReview 未检出正式发表记录。

**作者验证**：Sunghyeon Woo、Hoseung Kim 等 12 人，机构为 **NAVER Cloud + Toss + A2SYS**（论文 HTML 首页，含 navercorp.com 邮箱）。通讯作者 Dongsoo Lee 是 NAVER Cloud AI 压缩/推理方向负责人，工业界团队但无顶会系统发表记录（未验证其个人全部履历）。

**核心贡献与数字**：多智能体场景下多个微调模型反复 prefill 同一前缀。PrefillShare 把模型分解为 prefill/decode 两模块，冻结 prefill 模块共享、只微调 decode 模块，使多模型可共享同一份 prompt KV cache；在 vLLM 分离式系统上加路由机制实现共享。多模型 agent 负载下 p95 延迟降 4.5×、吞吐升 2.0×，微调后任务精度与全量微调持平（arXiv 摘要）。

**评级：淘汰**（观察名单；想法与 DroidSpeak（NSDI '26）同向且更激进（权重分解层面共享），工业系统背景真实；但**无录用、作者非著名学者**，按严格标准淘汰。可作为「跨模型 KV 复用」方向的观察对象，与 DroidSpeak 对读）。

---

## 3. PrefixWall: Mitigating Prefix Caching Side Channels in Shared LLM Systems（未录用）

**录用验证**：**未录用**。arXiv 2603.10726，v1 2026-03-11，v2 2026-05-20；**无 comments 录用注记**（arXiv API）。注意实际标题是 "...in Shared LLM **Systems**" 而非候选清单所写 "Serving"。已被 Preserving Provenance in Shared KV Caches（arXiv 2609.38706）等引用（cited by 3，Semantic Scholar 快照）。

**作者验证**：Panagiotis Georgios Pennas、Konstantinos Papaioannou、Marco Guarnieri、Thaleia Dimitra Doudali，机构 **IMDEA Software Institute + Universidad Politécnica de Madrid**（论文 HTML 首页）。Guarnieri（安全/形式化验证，前 AWS 安全研究）与 Doudali（IMDEA 助理教授，EuroSys 2026 workshop co-chair）是安全×系统交叉的正规军；一作 Pennas 应为博士生。

**核心贡献与数字**：面向 APC（Automatic Prefix Caching）的计时侧信道——多租户下缓存命中快、未命中慢，攻击者可逐 token 重建他人请求。现有防御是"关闭缓存"的锤子方案；PrefixWall 监控跨用户缓存复用、标记可疑共享、按需选择性隔离前缀。防御效果：相比用户隔离式防御，**缓存复用率高至 +70%、推理延迟低 30%**（arXiv 摘要）；MAGO 博客转述为中和约 70% 跨用户提取攻击（二手口径，未在摘要中验证）。

**评级：B**（问题真实（vLLM APC 默认开启）、防御视角是当前稀缺的"既安全又保性能"路线，作者团队可信；但数据来自自建测试床、未录用，数字 +70%/−30% 是相对最差防御的比较，含金量需打折）。

---

## 4. Learning Agent Execution for KV-Cache Management in Agentic Serving（系统名 CacheScout；未录用）

**录用验证**：**未录用**。arXiv 2608.14624，v1 2026-07-16；**无 comments 录用注记**（arXiv API）。arXiv 全字段检索无会议标注，OpenReview 未检出（未验证到逐条记录）。

**作者验证**：Rui Zhang、Chaeeun Kim、Shaoting Feng、Kuntai Du、Yuhan Liu、Yi Zhong、Cheng-Wei Ching、**Junchen Jiang**、**Liting Hu**；机构 **UC Santa Cruz + University of Washington + University of Chicago**（论文 HTML 首页）。Junchen Jiang 是 KV cache 复用方向核心人物（LMCache、CacheBlend 获 EuroSys 2025 最佳论文、DroidSpeak），Yuhan Liu/Kuntai Du/Shaoting Feng 均出自该组；Liting Hu（UCSC）为资深系统研究者。团队含金量高。

**核心贡献与数字**：指出多智能体 serving 中 prefix cache 是被动反应式的（recency 驱动），可复用的 agent 固定上下文（系统提示、工具定义、few-shot）总在下一次调用前被逐出。CacheScout 在线学习 agent 执行转移（无需预定义工作流图/离线训练），用学到的执行模型同时指导逐出与主动预取，不占关键路径。vLLM 之上实现：KV 命中率 **+10–18 个百分点（达 81–85%）**、mean TTFT −18–45%、每轮延迟 −29–38%、峰值吞吐最高 +57%；大模型下 TTFT 最高 −54% 且吞吐 +37%（arXiv 摘要）。

**评级：B**（方向（agentic 工作流感知的 KV 管理）正热，作者组是该领域最强团队之一；但未录用、数字为自建负载，观察其后续录用去向（大概率投 OSDI/NSDI '27 周期））。

---

## 5. PrfaaS: Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter（未录用）

**录用验证**：**未录用**。arXiv:2604.15039（v1 2026-04-16，v2 2026-04-22；comments 仅 "16 pages, 5 figures, 6 tables"，无录用注记）。另外，「Breaking The KV Wall for Next Generation LLM Serving」这一标题在 arXiv 与各会议均不存在（arXiv API 标题检索 "Breaking the KV Wall"、全字段 "KV Wall" 均零命中），实为 **Towards AI 博客文章**（2026-04-23，pub.towardsai.net，抓取 403，标题与日期经多次搜索快照交叉确认），内容是对本论文的解读。

**作者验证**：Ruoyu Qin（Moonshot AI×清华）、Weiran He（Moonshot AI，知名 vLLM 核心贡献者）、Yaoyu Wang、Zheming Li、Xinran Xu（Moonshot AI）、Yongwei Wu、Weimin Zheng、**Mingxing Zhang**（清华，通讯）（论文 HTML 首页）。Mooncake（FAST '25 最佳论文）同源团队 + 清华系统组。

**核心贡献与数字**：论证 PD 分离的部署边界由 KVCache 传输决定——稠密注意力下 KV 流量把 prefill/decode 锁死在同一高带宽域；混合注意力（NSA 类）架构把 KV 缩小后，跨数据中心 PD 成为可能，但需要选择性 prefill 卸载 + 带宽感知调度 + 缓存感知请求放置（Hybrid Prefix Cache Pool）。案例研究基于内部 1T 参数混合注意力模型：相比同构 PD 基线**吞吐 +54%、P90 TTFT −64%，等成本下 +15% 吞吐**，跨 DC 带宽占用温和（arXiv v2 摘要）。

**评级：B**（PrfaaS 为架构立场 + 建模案例研究（非完整系统实现），未录用，但出自工业界最懂 KV 传输瓶颈的团队，是「KV 墙」问题的最佳定位文献；引用数字时注意是内部模型案例研究。博客本身不评）。

---

## 6. 补充探索：2026（及 2025）正式录用论文

### 6a. KV cache 安全方向（正式录用）

| 论文                                                                                         | Venue                                                                                           | 机构                                                                                     | 要点                                                                                                                           |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving**   | **NDSS 2025**（官方页 ndss-symposium.org，DBLP 收录；Semantic Scholar 64 引）                   | Guanlong Wu 等，SUSTech + ByteDance                      | 首篇系统论证多租户 serving 的 KV cache 共享构成侧信道、可恢复他人 prompt；是 PrefixWall/Selective Sharing 等防御工作的攻击基线 |
| **Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference** | **NDSS 2026**（DOI 10.14722/ndss.2026.240258；arXiv 2508.09442；代码 github.com/SiO-2/kvcloak） | Zhifan Luo、Shuo Shao 等，**浙江大学 + 华为**（Zhan Qin资深）                        | 三类攻击向量（可从 KV cache 数据本身重建用户输入）+ 防御 KV-Cloak；攻防一体                                                    |
| **Auditing Prompt Caching in Language Model APIs**                                           | **ICML 2025**（OpenReview 官方页面）                                                            | Chenchen Gu、Xiang Lisa Li、Rohith Kuditipudi、Percy Liang、Tatsunori Hashimoto（**Stanford**） | 对真实商业 API 做实测审计（17 家中 8 家检出缓存、7 家跨组织全局共享，含 OpenAI）；缓存时序 oracle 可跨用户探测（探测口径，非逐 token 提取）；攻击面真实性最强      |

三篇均为顶级 venue 正式录用，评级 **A**。方向内未录用论文（供跟踪）：Agent-Assisted Side-Channel Attacks on Non-Prefix KV Cache in RAG（arXiv 2606.21842，2026-06，把攻击扩展到非前缀对齐的 RAG 场景）；Selective KV-Cache Sharing to Mitigate Timing Side-Channels（arXiv 2508.08438，防御，无录用注记）。

### 6b. PD 分离 KV 传输方向（正式录用）

| 论文                                                                                      | Venue                                                                                                               | 机构                                     | 要点                                                                                                                                                          |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DroidSpeak: KV Cache Sharing Across Fine-tuned Model Variants**（NSDI 论文集版）               | **NSDI 2026**（usenix.org 议程页收录；arXiv:2411.02820 最新 v4 仍题为 "…Cross-LLM Communication and Multi-LLM Serving"；NSDI 论文集版改用新题）                                                           | UChicago（Junchen Jiang 组）+ Microsoft  | 跨微调模型 KV 复用，选择性重算关键层（平均约 11%）：预填充延迟最高降 3.1×（TTFT，对照 full prefill）、吞吐最高 +4×（NSDI 版摘要口径）                                                                         |
| **HKVQ: Homomorphic KV Quantization for Disaggregated LLM Serving**                       | **SIGCOMM 2025 short paper**（UVA/Harvard 组页面确认，Coimbra）                                                     | Zeyu Zhang、Haiying Shen（UVA）等        | 传前把 KV 同态量化（隔离离群点 + INT2），以压缩换传输带宽；方向内最正式的传输压缩论文，但 short                                                               |
| **ARK: Avoiding Routing Collisions for KV Cache Transfer in Disaggregated LLM Inference** | **NAIC '26 workshop @ SIGCOMM 2026**（ACM DL 正式收录，作者 Hung-Chun Lin、Ting-Wei Hsu、Chung-En Ho、Ahmed Saeed） | UMass/Caltech 系（Saeed 组）             | KV 传输短促突发与 DCN 路由碰撞冲突，ARK 避让路由碰撞；workshop 级正式发表，评级 B                                                                             |
| **Strata: Hierarchical Context Caching for Long Context Language Model Serving**                      | **OSDI '26**（usenix.org 议程页"KV Cache and Long Context"专场）                                                    | Stanford/NVIDIA（Christos Kozyrakis 等） | 同场还有 ECHO（SJTU/华为，NSA 模型 KV 卸载）与 No Buffer, No Bottleneck（UVA，零拷贝 KV 卸载）——该专场 5 篇都是 OSDI '26 正式录用，可作为 KV 存储分层的坐标系 |

另有 **RTT- or Bandwidth-Bound? Demystifying the KV Cache Transfer in LLM Serving**（NAIC '25 workshop @ SIGCOMM 2025，DOI 10.1145/3748273.3749196，中国移动研究院+华为）：实测证明跨域 KV 传输常为 RTT 而非带宽受限，是理解 PrfaaS/ARK 前提的测量文献，评级 B。

---

## 首篇精读推荐：「KV 安全」方向

**推荐《I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving》（NDSS 2025）作为首篇精读。**

理由（按「攻击面真实性 × 防御实用性」评估）：

1. **攻击面最真实**：直接针对 vLLM APC 等生产默认开启的多租户共享机制，不依赖"攻击者拿到原始 KV 张量"这类过强前提，攻击模型就是同一 serving 集群上的普通租户——与本仓库 KV 复用/共享主题完全对口。
2. **承上启下**：它定义了攻击模型，PrefixWall（防御）、Selective KV-Cache Sharing（防御）、Non-Prefix RAG 攻击（扩展）全部以它为基线；精读一篇即可拿到整个子领域的坐标。
3. **正式录用 + 已受检验**：NDSS 2025 正式发表、64 引用，比 PrefixWall（arXiv-only）和 Shadow in the Cache（攻击需要 KV 数据访问权，前提偏强）更稳。
4. 备选：若想一篇同时覆盖攻防，选 **Shadow in the Cache（NDSS 2026，KV-Cloak）**；若想从 API 层理解真实世界暴露面，选 **Auditing Prompt Caching（ICML 2025，Stanford）**。
