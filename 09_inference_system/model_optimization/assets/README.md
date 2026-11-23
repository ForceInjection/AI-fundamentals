# assets

本目录存放本目录下文章的配图。

## DLite 系列（[draft 不再重建历史：DLite 的 history-free 投机解码](../dlite-history-free-speculative-decoding.md)）

- 两张手绘图，rough.js + 霞鹜文楷程序化生成，1080 逻辑宽 2x 渲染 PNG
- 图 1：三条 drafter 路线的历史税对比（`dlite-01-history-tax.png`）
- 图 2：DLite 的一轮推理数据流（`dlite-02-inference-flow.png`）
- 图内为示意；数字与机制以论文和官方仓库为准

## Loop Transformer 系列（[Loop Transformer:当「多想一轮」的边际成本趋近零](../loop-transformer-glm54.md)）

- 四张手绘图，rough.js + 霞鹜文楷程序化生成，1080 逻辑宽 2x 渲染 PNG，渲染源文件在 /tmp（不入库）
- 图 1：标准 Transformer 对比 Loop 两轮循环与门控写回（`loop-01-two-pass.png`）
- 图 2：KV Cache 全局共享 + 滑窗独立，对比朴素做法（`loop-02-kv-window.png`）
- 图 3：多想一轮的三笔边际成本账（`loop-03-cost-three-accounts.png`）
- 图 4：堆参数 / MoE 稀疏激活 / Loop 深度三条路线对照（`loop-04-moe-vs-loop.png`）
- 图内数字与机制表述以公众号发布文与 z.ai 官方博客为准，均为厂商口径
