# assets

本目录存放两篇文章的配图。

## dsh-*.png — 《DeepSeek Harness 技术入门与架构原理》书介图解文

- 五张手绘图（`dsh-01` … `dsh-05`），rough.js + 霞鹜文楷程序化生成，1080 逻辑宽 2x 渲染 PNG，可直接用于公众号正文
- 图中涉及的 DSH 概念（一切皆插件、四套 Preset、Skill/Tool/Hook、沙箱边界）以 [deepseek-harness-deep-dive.md](../deepseek-harness-deep-dive.md) 的源码核对结果为准（`deepseek-ai/deepseek-harness` `5dda764e`）
- 图 1 脚注的 8.6 分来自 DeepSeek V4.1 报告 §5.3.4 的 scaffold 消融（厂商口径）：DeepSWE v1.1 Max effort 上 mini-SWE 74.2 至 Codex 65.6

## dsec-*.png — DSec 论文深度解读文配图

- 四张手绘图（`dsec-01` … `dsec-04`），同一管线生成，服务于 [dsec-deep-dive.md](../dsec-deep-dive.md)
- 架构图简化自论文 Fig.1；负载曲线示意对照论文 Fig.3，数字出自 §4.3（Fig.5 / Fig.7）；分层对比对应 Fig.4；rollout 解耦对应 §6.2
- 论文原文 PDF 在 [references/](../references/dsec-arxiv-2609.22978.pdf)
