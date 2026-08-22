# 从芯片到 Agent：AI 基础设施的阅读地图 ——「AI 原力注入」好书推荐合集

> **元信息**：整理于 2026-08 | 素材来自公众号「AI 原力注入」好书推荐专辑（2023-11 至今 49 篇）| 章节体系对应 [AI Fundamentals](https://github.com/ForceInjection/AI-fundamentals) 知识仓库

## 引子：为什么需要一份「按层组织」的书单

AI 基础设施是一条从硅片到智能体的长链路：GPU 芯片设计、PCIe/NVLink 互连、集群网络与运维、云原生调度、CUDA 编程、大模型训练、推理引擎、Agent 工程、RAG 检索……任何一个环节的知识缺口，都会在真正的生产实践中暴露。

过去两年多，我们公众号陆续推荐了四十余本好书，覆盖了这条链路上的绝大多数环节。但单篇推荐解决的是「某本书值不值得读」，解决不了「按什么顺序读」。这篇合集把推荐过的书重新按**技术栈层级**组织成一张阅读地图——每一层推荐的书，就是进入该层的第一块敲门砖。章节顺序与 [AI Fundamentals](https://github.com/ForceInjection/AI-fundamentals) 仓库目录一一对应（硬件 → 互连 → 集群 → 云原生 → 编程 → 应用 → 理论 → 训练 → 推理 → 智能体 → 检索 → 学习路径 → 新范式），从底层到上层递进；不按线性阅读的读者可直接跳到文末的「阅读路径建议」按目标裁剪。

选书标准有三条：**中文优先**（无中文版的经典单独注明）；**理论与实战搭配**（每一层至少保留一本「原理书」和一本「动手书」，书目不足的层以开源资源补位）；**经典与新书并重**（基础层用时间检验过的经典，前沿层用 2025-2026 年的新书）。

---

## 一、硬件架构与互连技术

算力从哪来？答案是芯片。这一层决定的是「天花板」：SM 结构、Tensor Core、HBM 带宽、PCIe/NVLink 拓扑，直接决定上层软件能榨出多少性能。该层中文新书稀缺，推荐一本经受住时间检验的经典。

**《计算机体系结构：量化研究方法》（Hennessy & Patterson，机械工业出版社）**〔深度〕——计算机体系结构领域的经典之作，沉淀的量化分析方法论至今仍是评估任何新硬件（GPU、NPU、CXL 设备）的思维框架。适合想真正读懂硬件规格表、而不只是背参数的系统工程师。

> 该层技术迭代极快（NVLink 一年一代、CXL 3.0 织网），纸质书跟不上硬件节奏，体系化课程见第十二章 ZOMI 酱 AISystem，2026 年视角可配合 AI Fundamentals 仓库的[硬件架构章节](https://github.com/ForceInjection/AI-fundamentals)。

## 二、AI 集群运维与高性能通信

单卡算力再高，跑大模型也要组网。这一层解决的是「机器买回来之后怎么稳住」：GPU 状态监控、网络健康、性能观测。中文世界该层好书同样稀少，以下三本分别从网络内核、系统性能和边缘部署三个侧面补位。

**《深入理解 Linux 网络》（张彦飞，电子工业出版社）**〔进阶〕——从「网络包如何被接收和发送、阻塞在内部如何发生、epoll 的底层工作原理、TCP 连接底层如何支持」等开发运维高频问题切入，逐层拆解网络底层实现，带读者「修炼底层内功」、看清问题核心与背后的技术本质；全书是连接建立与收发包流程的概览总结，贴近实战。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247486151&idx=1&sn=efadd6984666d7b5dbe17f849dda12a1&chksm=e995c88adee2419c41d5ddac4b88b65d88285cf04c7ecdcd1ca4698ef9aff343275eb69ada41)〕

**《BPF 之巅：洞悉 Linux 系统和应用性能》（Brendan Gregg，电子工业出版社）**〔深度〕——继《性能之巅》之后，本书为读者「打开 Linux 内核大门」：从 BPF 起源到未来方向，完整覆盖编程模型与 BCC、bpftrace 两大前端框架；另一主线是系统与应用性能调优，讲清 BPF 工具如何与 Linux 传统性能工具互补，工具小巧精致、源码易读，为内核开发之旅铺平道路。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247485127&idx=1&sn=1d9a0604ecd0d01c43b9d2235335dea7&chksm=e995c48adee24d9c8401e310b408bde8e09b8f1e373b51ac75d7d17c269ea6e41721b06985a)〕

**《边缘云部署与运营：系统性实现方法》（Larry Peterson 等，机械工业出版社）**〔进阶〕——博主本人参与翻译的引进书。a16z 合伙人 Martin Casado 在序言中点破「反云」趋势：应用先迁上云、如今又分散部署，大公司把工作负载迁回自有的优化基础设施；正文以边缘云平台 Aether 为例，从架构设计到各子系统构建与运维细致拆解，补齐被深埋进三大云厂商的底层认知。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247486092&idx=1&sn=4eb2aee4e90c5daa4c10fbd1793ff1ce&chksm=e995c8c1dee241d7165f71a8e43f7cd027b5046424a5c5075ff5ba16ff349e74c978e70f551f)〕

## 三、云原生 AI 基础设施

AI 集群跑在 Kubernetes 之上。这一层解决「算力资源怎么编排」：容器、调度、存储、数据库。分布式系统的底层规律，是这一层所有工具的设计根源。

**《数据密集型应用系统设计》（Martin Kleppmann，中国电力出版社）**〔深度〕——全书分「数据系统基础—分布式数据—衍生数据」三部分：从可靠性、存储与检索，到复制、分区、事务、一致性与共识，再到批处理与流处理，从宏观层面讲清各项技术的共性与差异、把底层原理拆解透彻——理解了原理，也就明白每项技术的诞生背景、要解决的问题与适用场景。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247488960&idx=1&sn=92d650291882be60825766f75e3bc264&chksm=e995d78ddee25e9b06ff3fcc3990f6ea231b81019f514add32a978f5b53cf5aaa98a1032ddd9)〕

**《分布式系统应用设计（第二版）》（Brendan Burns，中国电力出版社）**〔入门〕——Kubernetes 共同创始人 Brendan Burns 的可复用设计模式手册：单机层面讲边车、大使、适配器模式（Istio 即边车模式的典型应用），集群层面覆盖负载均衡、分片、分散聚集、FaaS 与所有权选举；并以「算法编程—面向对象—容器编排」三阶段视角串起设计模式发展史。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247488705&idx=1&sn=d832c8feb6b87a31f7e76089fbaf9f9e&chksm=e995d68cdee25f9a522bc2d367ad147b12e3edd70959618625dc6634a2ce28decb8fef728691)〕

**《自己动手写 Docker》**〔进阶〕——用 Golang 从零实现一个精简容器运行时：从 run 命令起步，逐步打通 Namespace 隔离、cgroups 资源限制、rootfs 与 OverlayFS 镜像分层、数据卷与 Bridge 容器网络，每步配教程与对应 GitHub 分支代码；原书第一版年代久远，可结合作者博客的新版教程学习〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247486086&idx=1&sn=f845a71f806d772568e7a262cbac69bd&chksm=e995c8cbdee241ddd755e5018560566b11a3b25120f67ea99e282d59078330942c8572c27cbe)〕。

**《云原生数据库：原理与实践》（李飞飞、周烜等，电子工业出版社）**〔进阶〕——阿里云数据库团队撰写（第一作者李飞飞），以 PolarDB 工程落地贯穿：从 B+ 树、LSM-Tree 到 X-Engine 的存储引擎演进，计算存储分离与 Raft/Parallel Raft 高可用，再到 PolarDB-X 分布式事务与 HTAP 实践，是一本「值得放在手边常读常新」的云原生数据库工具书。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247489770&idx=1&sn=1e9b1f1d10907de0d5c0a0d25e902572&chksm=e995daa7dee253b14feec9ddc0280e5660de1060bbaf1bf317c5bd7499ed564224137c347ecc)〕

## 四、底层计算与异构编程

再往下一层是「亲手写算子」：CUDA 线程模型、共享内存、Tensor Core。这一层决定工程师能否真正榨干 GPU。中文实战书稀缺，从经典中选两本互补的入门路径。

**《CUDA C 编程权威指南》（John Cheng（程润伟）、Max Grossman、Ty McKercher，机械工业出版社）**〔入门〕——英文原书 *Professional CUDA C Programming*，NVIDIA 工程师团队撰写的经典入门，线程/块/网格模型、内存优化、原子操作循序渐进，配套代码可直接编译运行，是 CUDA 从零到一的标准路径。

**《大规模并行处理器程序设计（原书第 4 版）》（Kirk & Hwu，机械工业出版社）**〔进阶〕——美国大学 GPU 编程标准教材（PMPP），从并行思维而非语法出发讲 GPU 编程。第 4 版中译 2025 年上市（第 3 版仅有英文影印版），按「中文优先」原则推荐中译本；英文无障碍的读者亦可直接读原版。

> 动手环节可配合 AI Fundamentals 仓库的 [CUDA-Learn-Notes](https://github.com/xlite-dev/CUDA-Learn-Notes)（200+ 个优化内核示例）与 [nano-vllm](https://github.com/ForceInjection/nano-vllm)（约 1400 行实现核心推理机制的迷你 vLLM）巩固。

## 五、大语言模型应用开发与编排

从「会调 API」到「能交付应用」的中间地带：提示工程、AI 编程工具、应用框架选型。这一层入门门槛最低、市面书最多，也是最容易踩「营销书」坑的领域——以下两本均经过公众号实测推荐。

**《动手构建大模型》（Louis-François Bouchard & Louie Peters，人民邮电出版社）**〔入门〕——面向生产环境的端到端实践指南，目标明确：帮开发者跨过 Demo 阶段走向生产级系统。约 330 页覆盖 Prompting、RAG、微调、智能体、部署与评估全链路，配套 Colab 开箱即练，被一线工程师视作「反复翻、随手查」的实践手册；LlamaIndex CEO 称其为「迄今构建 LLM 应用最全面的教材之一」。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247491484&idx=1&sn=39d905c8b5b4c5f7e6a7d33ee985609e&chksm=e995ddd1dee254c7050b3ffab0b9f66ef032d6fc88b697024a90ca4b919ff24bfdc9f6391be4)〕

**《Cursor 与 MCP 快速入门：零基础开发智能体应用》（黄桂钊，人民邮电出版社）**〔入门〕——30 多个亲手实践的案例贯穿全书，核心公式「一个清晰的创意 + 精准的需求描述 = 智能体生成可运行的应用」：从亲子游戏、教学场景到招牌设计、小红书文案等商业变现，再到 Sealos 云端部署全流程；并手把手讲透 MCP 架构、自定义 Server 开发与阿里云百炼智能体集成。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247490746&idx=1&sn=4eeca1838aad0cd4a8b7a87c19052331&chksm=e995def7dee257e133d0cd9d9471a0995319a66a5d54cc2f131f1c7b7ac1ea4d33ce1348df32)〕

## 六、机器学习基础

AI 的基础学科。这一层不直接产出功能，但决定工程师对模型行为的直觉：损失函数为什么这样设计、正则化在解决什么、生成模型与判别模型的本质区别。

**《深度学习：基础与概念》（Christopher Bishop，人民邮电出版社）**〔进阶〕——大模型时代还需要啃深度学习教科书吗？作者的答案是「理解本质，才能创造未来」。全书以概率论为基石，从单层网络讲到 Transformer、扩散模型等前沿架构，辛顿、杨立昆、本吉奥三巨头联袂推荐，并提供与《模式识别与机器学习》衔接的阅读建议。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247489954&idx=1&sn=b270b5afddb8a59b3d68bd50a0046f9a&chksm=e995dbefdee252f915ecb4541fe402fa57c44d4d28654d35679b14f7f48425794047994447)〕

**《深度学习的数学工程：模型背后的数学原理》（Benoit Liquet 等，王斌译，人民邮电出版社）**〔进阶〕——源自澳洲名校暑期学校课程、美亚评分 5.0 的硬核著作：既非调包手册也非纯理论，以「工程视角」拆解深度学习背后的数学逻辑，从微积分、线性代数、概率论逐步构建到 Transformer 与图神经网络，本科一年级数学基础即可入门，小米 AI 实验室主任王斌翻译。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247492552&idx=1&sn=c57ef7ad02c225ad90e2b3dd33dd7b84&chksm=e9962185dee1a893ea04f8daf6b1036aa9dd74be89b58f516f0130d5a4cf836085cde9faa3c2)〕

**《大模型技术 30 讲》（Sebastian Raschka，叶文滔译，人民邮电出版社）**〔入门〕——《Machine Learning Q and AI》的中译本：30 个高频问答覆盖神经网络与深度学习、计算机视觉、NLP、生产与部署、预测性能与模型评测五个主题，180 余页短小精悍，适合查漏补缺的「问答词典」。

**《统计学习方法》（李航，清华大学出版社）**〔进阶〕——国内最经典的机器学习教材，感知机、SVM、HMM、CRF 的数学推导清晰严谨，是「知其所以然」的标准答案书。

**《机器学习》（周志华，清华大学出版社）**〔入门〕——「西瓜书」，用西瓜数据集讲透机器学习全谱系，兼顾广度与深度，适合与《统计学习方法》配合互为补充。

**《深度学习入门 5：生成模型》（斋藤康毅，人民邮电出版社）**〔入门〕——「鱼书」系列第五部：前四部豆瓣评分均 9.0+、中文版狂卖 19 万册。全书以连贯故事分 10 步从正态分布、EM 算法讲到 VAE 与扩散模型，用男女身高差解释双峰分布、用猜谜游戏类比 EM 算法，最后手把手实现简化版 Stable Diffusion。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247490193&idx=1&sn=60878aa47b9624e9b5d32e0537020b86&chksm=e995d8dcdee251cad3b905d26d767db4a1252ec8ff0c5fc58bf52cfdc2bf7ff3561366c82923)〕

**《精通特征工程》（Alice Zheng & Amanda Casari，人民邮电出版社）**〔进阶〕——「数据决定了模型的上限，算法只是无限逼近这个上限」——本书正是这句话的完整注解：数值、类别、文本、时间序列四类数据的特征构造、转换与选择方法全覆盖，配代码示例与金融风控、电商推荐等实战案例，并介绍 Featuretools 等自动化特征工程工具。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247489464&idx=1&sn=47671131ea315e2d46859ad077ae528d&chksm=e995d5f5dee25ce3ca0003fb097604eabc8970fa5de82d727524cb1e68c6025793c9f7e50bb0)〕

## 七、大语言模型理论与基础

理解 Transformer 为何能「涌现」出语言能力。这一层是当前书市竞争最激烈的赛道，从 200 页小册子到千页巨著都有，按阅读深度排序推荐。

**《How Large Language Models Work》（Stella Biderman 等）**〔入门〕——仅 200 页的 LLM 入门小册子：以清晰逻辑拆解大模型核心机制，覆盖预训练、微调、RAG、评估、对齐与可靠性设计等关键议题，提供可落地的技术路径，内容紧贴前沿，适合想系统掌握 LLM 底层原理、为论文或项目构建坚实方法论的硕博生。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247492451&idx=1&sn=141224caacec6e06f7ec4111dce89dff&chksm=e996212edee1a838e436c1eaeec2dfc67fa902e5f9bc3ec01041f060eb00c8862049686f28e4)〕

**《大语言模型》（Terrence Sejnowski，李梦佳译，中信出版社）**〔入门〕——四院院士、玻尔兹曼机共同发明者谢诺夫斯基新作：以「AI 反向测试人类」现象切入，提出「反向图灵测试」与「厄里斯魔镜」假说，追问 LLM 是否真正理解语言，并给出幻觉是创造力体现、从「我思故我在」到「我演算故我在」等洞见，是技术书之外的思想书。

**《图解大模型：生成式 AI 原理与实战》（Jay Alammar & Maarten Grootendorst，李博杰译，人民邮电出版社）**〔入门〕——图解 Transformer 原作者的作品，300+ 全彩插图覆盖分词、嵌入、Transformer、RAG、微调、多模态全链条，随书附赠「图解 DeepSeek-R1」彩蛋与 155 道面试题。入门体验最好的中文图文书。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247490376&idx=1&sn=084c6a98a9921461d17cb16a943ad8e7&chksm=e995d905dee250132d46570b554ba8b557483d144183b6b07a1584e9a439d050501ff4ba871)〕

**《大模型基础》（浙江大学 ZJU-LLMs 团队，开源书籍）**〔进阶〕——开源书籍的标杆：传统语言模型、架构演化、Prompt 工程、参数高效微调、模型编辑、RAG 六章，月度更新紧跟论文前沿，附完整 PDF 免费下载。学生党友好。

**《百面大模型》（包梦蛟、刘如日、朱俊达，人民邮电出版社）**〔进阶〕——面向大模型求职热点的实战工具书：以「面试题 × 技术点 × 项目实战」三位一体结构组织 100 道核心面试题，覆盖预训练、对齐（PPO/DPO）、RAG、智能体与推理加速等求职五大模块，每章配代码实战与工程图解。不是「背题集」，而是从「知道」到「能做」的成长路径。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247489907&idx=1&sn=7209048dad89e2fb90865ed778f111a8&chksm=e995db3edee25228d3b9233e7935f601622ac47e4d8a1008e80b19303f22e6da3969d586f307)〕

## 八、大模型训练

把模型从随机权重训练到可用，是算力、数据、并行策略的三角工程。这一层书籍最少、门槛最高，两本足以构建完整认知。

**《从零构建大模型》（Sebastian Raschka，人民邮电出版社）**〔进阶〕——从零手写一个 GPT 风格模型，填补「资料或过于抽象、或高度依赖框架封装」的空白：从文本预处理、词嵌入、注意力机制讲到 Transformer 与 GPT 架构，完整走通预训练、指令微调与部署，代码全部开源（含中文注释版），自注意力实现仅 50 行，小数据集在个人电脑即可跑通。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247489728&idx=1&sn=a62683e9c1d7b4f45c40128b15167edb&chksm=e995da8ddee2539b7d52a99491fc2b225fefdcdff987eaf54c9ba7149517aea21cd2ac277ea1)〕

**《大规模语言模型：从理论到实践（第二版）》（张奇、桂韬、郑锐、黄萱菁，电子工业出版社）**〔深度〕——复旦大学四位作者的系统性著作：第二版新增 50% 以上全新内容，涵盖 MOE、多模态、智能体、RAG、效率优化、评估与应用开发等热门方向，构建「理论基础 → 数据处理 → 模型训练 → 应用部署」的完整技术闭环，官方主页提供免费预览版电子书。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247489809&idx=1&sn=799e43b526c75d90ed9297d910395b14&chksm=e995db5cdee2524af696f4414017d64e1f12c784fb71039a3bd582a87cd67146815e94292c8f)〕

## 九、大模型推理

推理是 AI 基础设施的「最后一公里」：KV Cache、批处理、投机解码、PD 分离，直接决定每 token 成本。中文专著极少，这本 2026 年新书是稀缺的实战型补充。

**《vLLM 与 SGLang：大模型高效推理双引擎实战》（李明飞，电子工业出版社）**〔进阶〕——对比两大推理引擎：vLLM 走「通用极致性能」路线，SGLang 走「场景深度优化」路线，拆透 PagedAttention 与 RadixAttention 的缓存哲学；7 个生产级项目覆盖 90% 以上落地需求，附三维度选型决策矩阵。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247492750&idx=1&sn=0dc47c03f65609d03fdd8b15cd10dad9&chksm=e99626c3dee1afd5138aa3d3d9676e2a32edd3dbc7d6f39d66b4d28ad6d6308b4d9e4773f641)〕

> 原理层面可配合 AI Fundamentals 仓库的 KV Cache 技术体系、vLLM 模块分析等文章做源码级补充。

## 十、企业级 AI Agent 开发

从「模型」到「员工」：工具调用、记忆、规划、多智能体协作。2025-2026 年中文 Agent 书集中爆发，按「工程深度」分层推荐。

**《Agent Skills 开发实战：像搭积木一样构建智能体》（代晶，人民邮电出版社）**〔进阶〕——从 OpenClaw 登顶 GitHub Star 榜首讲起：针对单体提示词的痛点，拆解 Skills 文件系统与渐进式披露机制，厘清 Skills（该怎么做）与 MCP（能做什么）的分工；电商经营报表、医疗病历清洗两个企业级案例贯穿始终。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247492729&idx=1&sn=0c766381aec1dff4e9686e80231fcac1&chksm=e9962634dee1af22a679fc7829f4832a5cb1dfeda4d5e2561cef136e371c53ff37f439c6a8b2)〕

**《动手学大模型智能体》（温睦宁、林江浩、张伟楠、俞勇，人民邮电出版社）**〔进阶〕——上海交大团队的 15 章教学实践课程，分基础（提示、评估）、架构（记忆、RAG、工具、规划）、微调（指令微调、LoRA、强化微调）、前沿（多模态、多智能体、安全、协议）四篇，理论与示例代码并重，配套课件与在线课程，亦适合作高校辅助教材。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247491444&idx=1&sn=25c4b9309d943b11993eaeec03467c37&chksm=e995dd39dee2542fd2a9871560b0bd033425a236f0f5db5f3501978349c1ea41efa03b58053c)〕

**《Agent 设计模式：图解可复用智能体架构》（黄佳，人民邮电出版社）**〔深度〕——从 GoF 23 种经典设计模式到「概率性智能」时代的迁移：六大主轴（感知、记忆、推理、行动、反思、协作）× 21 个核心模式，并以 OpenClaw 架构实证；后记「熵的园丁」之喻道出真谛——工程师是培育可演化的生态，而非构建精密机器。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247492016&idx=1&sn=e14d972a0f44fab98ee82f124325b88f&chksm=e99623fddee1aaeb2549a67f0ba7b16ec16e049fec4358e163fadf7e785d79a4c0f141c826c0)〕

**《AI 工程》（Chip Huyen，宝玉译，人民邮电出版社·图灵）**〔进阶〕——O'Reilly 阅读量最高的 AI 工程书；作者 Chip Huyen 曾任 NVIDIA NeMo 核心开发者，坚持「工具会很快过时，但基础原理应该持续更长时间」，不堆代码、原理优先，覆盖评估、幻觉、RAG、Agent 与成本安全。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247491620&idx=1&sn=972420de7d680cc3c4604264a3e1e843&chksm=e9962269dee1ab7f57dacadb8c9358d8b3e2d0cb9c6c4837f6c9387d519cd7c44bde0e68a9b5)〕

**《MCP 开发从入门到实战》（杨威理，人民邮电出版社）**〔入门〕——MCP 少有的中文系统教程，把 MCP 比作「AI 应用领域的 USB-C 接口」：从协议架构与核心组件，到 Claude 桌面应用配置、天气预报服务器开发与 Inspector 调试，再到 Smithery 等生态实践，8 章循序渐进。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247490328&idx=1&sn=952d9f89ea833f4da554530f0ac25418&chksm=e995d955dee25043a88a355aed821e2828591e145b9050788c6cc9b141aa2715a80a8f895dba)〕

**《生成式 AI 开发揭秘：大模型详解》（田雪松，机械工业出版社）**〔进阶〕——全栈技术指南：「白话原理 + 代码复现」拆解 Transformer 与扩散模型，覆盖 NLP、语音、视觉全模态、Agent、RAG；工程化最硬核——Scaling Laws 算力估算、消费级显卡 LoRA 微调、量化蒸馏、RLHF 评测。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247491283&idx=1&sn=f2c25087edef5f0c2abe5b32532b38f3&chksm=e995dc9edee255888667600bb7c5249093222b04990138f47d34141065c66d768b2ad8ebdf33)〕

**《Generative AI in Action》（Amit Bahree，Manning）**〔进阶〕——微软首席项目经理 Amit Bahree 撰写，美亚评分 4.6：三部分 13 章，从认知、提示工程/RAG/微调技术到架构、成本与负责任 AI 治理，以「智能市场研报生成助手」案例贯穿全程，配套 GitHub 代码仓库可运行。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247491289&idx=1&sn=1340dc9f385bee1b42924e9789bb7a83&chksm=e995dc94dee2558247b5fcafc444603aedca361293ab68e267eb1b2aa8ea56a9701b8240ea5a)〕

**《这就是 OpenClaw：半小时玩转小龙虾》（刘江，人民邮电出版社）**〔入门〕——作者刘江是「大模型」术语定名人、智源研究院创始副院长；用 9 个案例（智能体 Felix 月入近 20 万美元）讲透「AI 从认知工具转向执行主体」，以「喂粮食」「出门遛虾」式类比降低门槛，30 分钟零编程基础即可上手，专章给出安全配置。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247492383&idx=2&sn=952b87461c3ba47553d2865a2cb298a6&chksm=e9962152dee1a8440fddfafd2a595cdaaf34921f9a23ad7cd66c8308d30bbd7e4b60f74b20f2)〕

> 工具与协议层可配合 AI Fundamentals 仓库的 MCP 深度解析、上下文工程、记忆系统架构等文章。

## 十一、检索增强生成与文档智能

RAG 解决「知识不进权重」的问题：让模型在回答时引用外部知识库。这一层从 2024 年起成为企业应用的主力形态。

**《大模型应用开发：RAG 实战课》（黄佳，人民邮电出版社）**〔进阶〕——从 0 到 1 构建可落地 RAG 系统的实战指南：从数据分块、向量检索的「从零手搓」到权限分层、溯源审计的工业级设计，覆盖 GraphRAG、Agentic RAG、多模态 RAG 前沿范式，首创「RAG 架构决策树」辅助技术选型。作者咖哥（黄佳）为新加坡科技研究局首席工程师，全书 132 张架构图，获宇树科技创始人王兴兴力荐。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247490027&idx=1&sn=8f3366e5b9d39212a7b1f2d517bc54ef&chksm=e995dba6dee252b029e3dd4902f69ab42b4e7a5b3917ee68afe65480c5e2d229490316d054f2)〕

**《大模型应用开发极简入门：基于 GPT-4 和 ChatGPT（第 2 版）》（Olivier Caelen & Marie-Alice Blete，何文斯译，人民邮电出版社）**〔入门〕——热销 2 万册的经典升级版：以 150 页（第 1 版）至 300 页（第 2 版）的精炼篇幅讲透提示工程、微调、RAG 三大范式，配套新闻稿生成器、YouTube 视频摘要等 6 大场景，代码开源、可直接复用。第 2 版新增 RAG、智能体工作流与 DeepSeek 开发案例，由 Dify 产品经理何文斯翻译，兼具学术严谨性与工业落地视角。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247489547&idx=1&sn=408e0dc2223b5b288731ca570bf69547&chksm=e995da46dee253505bc87bbe7ae0b837de0ff07e9cc23e8515cff8a85c585f3bfa809d80be23)〕

## 十二、课程体系与学习路径

单本书解决单点知识，体系化学习需要课程与路径。这一层收录体系化的免费课程资源。

**ZOMI 酱 AISystem（GitHub 开源课程）**〔进阶〕——AI 系统全栈开源课程（https://github.com/chenzomi12/AISystem）：硬件基础、编译器技术、推理优化、框架设计四大模块，B 站配套视频。AI 基础设施工程师的体系化免费教材，强烈建议通读。

> 更多体系化资源可参考 AI Fundamentals 仓库的[课程体系章节](https://github.com/ForceInjection/AI-fundamentals)：Trae 编程实战课程、多智能体 AI 系统培训材料、微软 AI Agents for Beginners 课程等。

## 十三、AI Native 全栈实践

Software 3.0：AI 编程助手成为「超级编译器」，工程方法论全面重构。2026 年的关键词是 Spec 驱动开发与 Harness 工程。

**《SDD 实战：规范驱动开发之道》（黄佳，人民邮电出版社）**〔进阶〕——国内较早的系统性 Spec-Driven Development 专著。针对 Vibe Coding「需求蒸发、上下文漂移、不可审查、不可复现、不可维护」五大致命问题，提出「规范先行」：工程师的核心产出是 proposal.md、design.md、tasks.md 三份规范文档，代码只是 AI 依据规范生成的衍生物，构成项目的「唯一真相来源」。全书以「智能日报生成器」项目手把手演示六阶段工作流。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247492680&idx=1&sn=1925b1f9e8b1f4471a592723e3607af5&chksm=e9962605dee1af139ebcb569dbdcf0625350cff1d168bc43fe9c4a0064fe967ece91665a3966)〕

**《Codex 快速入门：Harness 工程落地》（袁从德、吴军、陈文浩，人民邮电出版社）**〔进阶〕——国内首批 Codex 系统专著。针对「AI 编程工具已经足够强大，但很多团队仍不知道怎样稳定地用好它」的现实，讲清从「AI 代码生成」到「AI 工程协作」的转变，核心理念：让 AI 看到正确的信息，比写出一条漂亮提示词更重要。全书按「认知迁移→上下文工程→高频实战→团队落地」四阶段递进，示例围绕 emotional_chat 开源仓库展开，兼顾 Cursor、Trae 等主流工具。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247492640&idx=1&sn=638190bae25948789112a12d57193484&chksm=e996266ddee1af7bc2c5fdff42afde70ef6b424b40ecbbf3c788cabe89b4422882e0d0283e70)〕

**《软件设计的哲学（第二版）》（John Ousterhout，人民邮电出版社）**〔进阶〕——源自斯坦福 CS 190 课程的设计原则集：软件设计的本质是对复杂性的管理，问题分解是程序员的首要设计任务。提出「模块应该是深的」、信息隐藏、「通过定义来规避错误」、「先写注释」等原则与识别设计问题的危险信号，均来自作者编写约 25 万行代码、参与创建多个操作系统的一线实践。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247488526&idx=2&sn=122af6bf418154afc4822372d3583171&chksm=e995d643dee25f55b0f7e053d5abb628b703bb577516c54945373bb4426bf529de77e8d9cf6c)〕

**《图解 Skill：AI 提效实战指南》（宝玉，人民邮电出版社）**〔入门〕——一本「用 skill 写出来的 skill 书」：书中配图全部由作者自己开发和迭代的 book-illustrator skill 生成，本身就是「用 skill 写关于 skill 的书」的闭环。从 Skill 心智模型（第 1-2 章）、从零写 Skill 全流程（第 3 章）、提示词工程结合（第 4 章）到写作工作流串联（第 5 章）逐层进阶，并给出渐进式披露三层的 token 预算量化标准（frontmatter 30-100 / 正文至 5,000 / 参考层无上限），是「从入门到精通」的完整路径。〔[公众号原文](https://mp.weixin.qq.com/s?__biz=MzI0OTIzOTMzMA==&mid=2247492518&idx=1&sn=b9bd39b8debeba878909fb8112c33070&chksm=e99621ebdee1a8fdf853da4fb1b9b0df6132550b2a8d3303c533e39c8af4a024e73f64fc907c)〕
## 选书标准的边界说明

本合集的取舍基于三个原则，也意味着三个边界：

1. **中文优先，经典原版注明**——绝大多数条目推荐中译本；确无中文版的经典（如《Generative AI in Action》）才以原版形式推荐。
2. **理论书与实战书搭配**——每一层至少一本「原理书」；纯实战书（如纯工具教程）占比刻意压低，避免书单变成说明书堆砌。
3. **无书章节用开源资源补位**——硬件、集群通信等迭代极快的层级，纸质书天然滞后，以开源课程与仓库文章替代，不硬凑书目。

## 说明

- 本合集素材来自公众号「AI 原力注入」好书推荐专辑（[合集链接](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI0OTIzOTMzMA==&action=getalbum&album_id=3377874180926619650)）2023-11 至今的 49 篇文章，单本详情可回看对应原文。
- 书目信息（书名、作者、出版社、版本）依据公开渠道整理核实，个别条目的版本信息以出版方为准。
- 章节体系与 [AI Fundamentals](https://github.com/ForceInjection/AI-fundamentals) 仓库完全对应，仓库内每章有源码级深度文章可作书目之外的延伸阅读。

---

> **写在最后**：这张地图会随公众号持续更新——每当值得读的新书出现，会按对应层级补充进去。你现在卡在哪一层？欢迎在评论区告诉我，也可以直接去 [AI Fundamentals](https://github.com/ForceInjection/AI-fundamentals) 仓库按章节体系展开阅读。
