# CloudMatrix384 与 CloudMatrix-Infer

华为超节点 + SiliconFlow 推理栈的论文深读方向：384 张昇腾 910 与 192 颗鲲鹏 CPU 经 UB 互联成单层对等超节点（跨节点带宽损失 <3%），配套的 CloudMatrix-Infer 给出 PDC 三池分离、EP320 每 die 一个专家、UB 直连内存池的完整服务栈。

## 内容导航

- **[一座 384 卡的 TokenFactory：CloudMatrix384 论文精读](cloudmatrix384-tokenfactory.md)**：超节点硬件（双 die 910、三级组网、UB/RDMA/VPC 三平面）→ PDC 分离与 KVCache-centric 对照 → FusedDispatch/FusedCombine、MLA 三招、双微批非对称流水线、MTP 去 CPU 化 → SP-TP-SP 与 AIC/AIV/SDMA 分工 → EMS 缓存池与 INT8 五件套 → 产能数字与五条口径声明 + 总评（基于 arXiv 2506.12708 v3，2026-11-11 核对）。

## 相关链接

- [DeepSeek 昇腾开源栈](../deepseek_ascend/README.md)：另一条昇腾推理路线（DeepSeek 六组件，绑定 Ascend 950）
- [Ascend950 微架构速查](../../01_hardware_architecture/ascend/Ascend950_微架构速查.md)：下一代昇腾硬件
- [MLA 的 TP 切分](../vllm/module_analysis/mla_tp_kv_redundancy.md)：TP 组内 KV 全副本与冗余调度（论文 §4.3.3 连接映射的背景）
- [KV 复用总纲](../kv_cache/06_kv_reuse/01_kv-reuse-overview.md)：Context Caching 的「同前缀一致」约束框架
- [TokenFactory 演讲](../../99_misc/token_factory_talk/README.md)：本文「token 工厂」叙事的骨架
