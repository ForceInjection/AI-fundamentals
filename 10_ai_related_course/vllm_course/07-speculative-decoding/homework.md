# 第 7 讲课后作业

## 作业（必做两题）

**题 1（选型报告）**：三个负载，各选方法并写完整 `--speculative-config` 片段 + 三句理由（草稿来源/acceptance 预期/代价）：

1. 代码补全助手（高重复、低延迟敏感、QPS 中）；
2. DeepSeek-V3 生产部署（有官方 MTP checkpoint，高 QPS）；
3. 通用客服机器人（闲聊为主，偶有话术模板），要求零额外模型与零训练。

**题 2（口径对账）**：用实验 2 的三口径数据做一次对账分析：日志均值（含 bonus）、per-request 直方图均值（不含）、Prometheus 计数器算出的 1+accepted/drafts。三个数字两两比较，写出「何时可比、何时不可比」的规则表（至少三条规则，每条附你在实验中看到的数字）。

## 附加题（不计考核）

HEAD 有 `enable_adaptive_verification`（AdaptiveVerificationManager，`adaptive_verification.py:227`，按置信度裁剪要验证的 draft 槽位）。读该类 `__init__` 与核心方法，回答：c 侧（草稿成本）的节省从哪里来？a 侧（接受率）会付出什么？在什么负载下这个开关是净收益？300 字以内。

## 自测题（不提交）

1. 为什么 greedy draft（默认）时 `draft_logits` 是 None？probabilistic 多了什么成本？
2. `can_skip_drafts` 在什么条件下触发「仅 target」步？这与动态 SD(K→0)有何本质区别？
3. spec decode 打开时，第 2 讲的 `draft_slots` 与 `num_output_placeholders` 分别保护什么？（KV 预算 vs 异步输出占位）
