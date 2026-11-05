# Kubernetes GPU 管理与 AI 工作负载

Kubernetes 最初是为无状态 Web 服务设计的，对 AI 负载并不友好：训练任务要求几十张卡同时就绪、不允许部分调度，推理服务要在 Leader-Worker 形态下跨节点协作，而 GPU 既不能随意超分，也没有成熟的换出机制。这一章围绕一个问题展开：要让 Kubernetes 真正跑好 AI 负载，从容器运行时到上层调度器，需要补上哪几块。

## 1. 核心基础设施组件

设备可见、资源上报、故障诊断，这三件事靠的是 NVIDIA Container Toolkit、Device Plugin 和 containerd 日志组成的一套「暗管道」。

- [NVIDIA Container Toolkit 原理分析与代码深度解析](./01_nvidia_container_toolkit_analysis.md) — OCI hook 注入链路、CDI 模式与各容器运行时的集成，含源码级分析与排障路径。
- [Nvidia K8s Device Plugin 原理解析和源码分析](./02_nvidia_k8s_device_plugin_analysis.md) — 设备发现注册、资源分配与健康检查机制，含 ListAndWatch / health 源码走读与 MIG 策略。
- [容易被忽略的 containerd 运行时日志](./06_containerd_log_analysis.md) — runc 运行时日志位置，容器启动失败的深度定位方法。

## 2. 高级调度与资源管理

GPU 有限、任务排队，谁先跑、一张卡能不能切开给几个人用——需要比 kube-scheduler 更懂 AI 负载的队列系统和 GPU 切分机制。

- [Kueue + HAMi：Kubernetes 原生的 AI 工作负载管理与 GPU 虚拟化解决方案](./03_kueue_hami_integration.md) — ClusterQueue/LocalQueue 配额与公平分享，HAMi 显存与算力的细粒度切分，含生产级部署示例。

## 3. 分布式推理框架

推理端把「一份任务、多卡并行」变成「一份服务、跨节点展开」：Leader 负责调度和请求编排，Worker 真正跑模型，KV Cache 与参数都要跨节点协同。

- [vLLM + LWS：Kubernetes 上的多机多卡推理方案](./04_lws_intro.md) — LWS 控制器的 Leader-Worker 编排与 vLLM 多节点推理部署，含完整 YAML 示例。
- [云原生高性能分布式 LLM 推理框架 llm-d 介绍](./05_llm_d_intro.md) — Gateway API + EPP 前缀缓存感知路由、P/D 分离与 LMCache KV 分层，vLLM/SGLang 双后端。
