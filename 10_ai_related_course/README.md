# AI 相关课程

AI Fundamentals 课程资源库汇集了人工智能领域的系统性课程、培训讲稿、课件与实战案例。内容涵盖了从底层的 AI 基础设施到上层的 AI 编程实战、智能体开发等多个维度，旨在为学习者提供一条完整的人工智能技术进阶路径。

## 1. 课程模块概览

以下六大核心模块梳理了 AI 学习的基础设施与实战应用，为学习者构建了清晰的知识导航体系。

- **[AI 编程入门](ai_coding/README.md)**：包含 AI 辅助编程的基础概念与实战演示课件。
- **[多智能体系统](multi_agent_system/multi_agent_training/README.md)**：深入讲解多智能体架构设计、LangGraph 框架应用及企业级落地实践。
- **[Trae 编程实战](trae/README.md)**：围绕 Trae 智能 IDE 展开的从基础到高级的全流程实战开发教程。
- **[KV Cache 八讲](kv_cache_course/README.md)**：从 token 账本、page/slot 到 nano-vllm / vLLM / SGLang 源码，再到 GLM-5.3-Flash 与 DeepSeek-V4.1-Flash 的容量分析。
- **[哈佛 CS2680 Modern AI Systems](university_courses/harvard-cs2680/README.md)**：哈佛 SEAS 2026 秋季课程课件存档，从 Agent 循环讲到 LLM serving 与底层 GPU 硬件。
- **[大学公开课档案](university_courses/README.md)**：CS336（大模型本身，从 tokenizer 造到 RL）、CMU 11-868（LLM Systems serving 单线）、Berkeley LLM Agents（agent 工作流与安全）三门公开课的定位、资源地图与学习建议。

## 2. 核心课程详情

详细的模块大纲明确了各个技术栈的侧重点与核心知识体系，涵盖从工具使用到架构设计的完整演进路径。

### 2.1 AI 编程入门

AI 编程基础培训通过实战演示与交互式课件，指导开发者如何利用 AI 工具重塑软件开发工作流并大幅提升研发效率。详见 [ai_coding/README.md](ai_coding/README.md)。

- 课件资源：包含多页 HTML 格式的演示幻灯片，可通过浏览器直接查看（附带 `ppt-mode.js` 驱动）。

### 2.2 多智能体系统

多智能体系统开发培训从底层理论延伸至企业级应用，全面覆盖 LangGraph 框架实践与 LangSmith 监控体系。详见 [multi_agent_system/multi_agent_training/README.md](multi_agent_system/multi_agent_training/README.md)。

- **理论基础**：多智能体系统概论，建立基础认知。
- **框架应用**：深入探讨 LangGraph 框架在复杂业务流程中的应用。
- **运维与架构**：涵盖 LangSmith 监控平台集成、企业级系统架构设计以及部署运维实战。

### 2.3 Trae 编程实战

Trae 智能编程实战教程采用循序渐进的五阶结构，系统讲解从环境配置、代码开发到微服务运维的全流程 AI 辅助开发方案。详见 [trae/README.md](trae/README.md)。

- **基础入门**：Trae 简介、环境配置与基础交互模式。
- **场景实战**：现代前端开发、高性能后端 API、数据库设计与安全权限管理。
- **高级应用**：涵盖微服务架构、数据分析可视化与 AI 模型集成。
- **团队协作**：探讨代码质量管理、项目协作与 DevOps 云原生实践。
- **综合项目**：从架构设计到核心功能开发，再到最终的部署运维全流程演示。

### 2.4 KV Cache 八讲

八讲 × 120 分钟的 KV Cache 专题课，主线是“token 账 → page/slot 口径 → request/state 账”。前四讲建立公式、分页和引擎实现，第五讲从 block 级复用走到 radix tree 与 HiCache 三层，第六、七讲分别读 GLM-5.3-Flash 与 DeepSeek-V4.1-Flash 的 KV 相关源码，第八讲做容量规划与引擎选型。详见 [kv_cache_course/README.md](kv_cache_course/README.md)。

- **实践载体**：本地 nano-vllm `b9dd55a`、vLLM `8b365ff949`、SGLang `f4de6abee6`。
- **前置材料**：`ai-infra-course` 模块 5/6；课程不重复 vLLM 入门，只做 KV Cache 纵深。
- **产出**：token/state 容量表、源码映射、Agent 长会话选型备忘；术语见 [术语表](kv_cache_course/glossary.md)。

### 2.5 哈佛 CS2680 Modern AI Systems

哈佛 SEAS 课程 CS2680（Fall 2026，Juncheng Yang / MaDSys 组主讲）的课件存档，课程沿「Agent 循环 → LLM serving → 底层硬件」一条线讲 AI 系统优化。已收录 Efficient LLM serving basics 三讲：I「Intro to LLM」（Transformer / Llama 3 / MoE / KV cache）、II「Intro to GPUs」（roofline、内存层级、nsys/ncu 剖分析、FlashAttention 谱系，上下两份 deck）、III「scheduling, paging and parallelism」（TTFT/TPOT/SLO 口径、batching、PagedAttention、五种并行策略）。详见 [harvard-cs2680/README.md](university_courses/harvard-cs2680/README.md)。

### 2.6 大学公开课档案

三门公开课各补一块能力版图：Stanford CS336 讲大模型本身（从 tokenizer、Transformer 到 scaling laws 与 RLHF 全部从零实现，作业与交互式讲义全公开），补全模型构建基础；CMU 11-868 讲 LLM Systems（serving 单线动手作业，vLLM/SGLang 内核与 CUDA/Triton，slides 未公开仅做资源记录），加强推理系统；Berkeley LLM Agents MOOC 把 agent 线展开成独立体系（12 讲录像与 slides 全公开，agentic 工作流、tool use、评测与安全），加强 Agent 系统。详见 [university_courses/README.md](university_courses/README.md)。

### 2.7 独立学习资料

独立的技术专题文档提供了特定领域的知识纵深，为完整的课程体系补充了具备商业实战价值的延展阅读材料。

- [AI Agents for Beginners 课程之 AI Agent 及使用场景简介](AI%20Agents%20for%20Beginners%20%E8%AF%BE%E7%A8%8B%E4%B9%8B%20AI%20Agent%E5%8F%8A%E4%BD%BF%E7%94%A8%E5%9C%BA%E6%99%AF%E7%AE%80%E4%BB%8B.md)：微软智能体入门课程概览（2026-11 核对，上游已扩至 18 课）：从基础概念、设计模式到智能体协议、上下文工程、记忆管理与 MAF 框架，含第 1 课详解与「何时该用智能体」。
