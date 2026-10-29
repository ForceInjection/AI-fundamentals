# 第 7 讲：KV Cache 在 DeepSeek-V4.1-Flash 里怎么压缩 — PPT 提纲

每讲 10–12 页，只列页面主题和讲解落点，不写完整讲稿。

1. 封面：890 B/token 从哪来
   落点：本讲只回答两个问题，global KV 怎么压到 890，SWA 为什么退出持久层。

2. 动机：V4-Flash global 约 3,514 B/token，SWA 占持久层近一半
   落点：先让学生看到“分钟级复用、72 小时保留”的矛盾，再进入 FP4 和层间共享。

3. 5 分钟回顾：混合注意力为什么把缓存拆成多种对象
   落点：第 6 讲按 KDA / full 拆，本讲按 global / SWA 拆；对象由机制决定，不由层数决定。

4. 两条注意力分支：global 与 SWA
   落点：global 分支产生 main KV + indexer K，SWA 分支产生窗口 KV；只有 global 随序列长度增长。

5. 890 的算式：2.5 × (288 + 68)
   落点：2.5 = 3 个 encoder Full × 1/2 + 1 个 decoder Full × 1；逐项落到字节，强调 indexer 不是第三层。

6. CED：20 encoder + 20 decoder
   落点：decoder 的 global KV 从 encoder 输出投影，prefill 的 global 部分只算前一半层。

7. CSA2：4 Full + 4 Reindex + 30 Reuse
   落点：只有 4 个 Full 层产 main/indexer K，这是 2.5 的来源；Reindex 只重选，不重产 KV。

8. 层级 indexer：首层建池，后续层池内 top-k 512
   落点：首层扫全量，后续层只在 16,384 个候选里打分，深层成本从线性变常数。

9. SWA 与总 KV：890 × seq + 2.7 MB
   落点：890 不含 SWA；40 × 128 × 528 ≈ 2.7 MB/request；短序列下 SWA 反而是大头。

10. Prefix caching：只在 bounded replay 开启时只缓存 global KV
    落点：写成条件句，列出非 V2 runner、prefill CP > 1、ROCm 三种会反过来缓存 SWA 的配置。

11. Replay 边界：[H−128, H) 与逐 query 窗口
    落点：给出 replay_start = H − min(128, H)，说明窗口截断和“不保证 bit-exact”。

12. 源码与口径：字段、触发点、缓存类
    落点：`swa_bounded_replay` 是 `CacheConfig` 字段，触发点在 scheduler 的 `_mark_prefix_replay`，缓存类是 `DeepseekV4SWACache`；收尾用 890 / 1,050 / 1,650 三档说明“不是所有 vLLM 部署都叫 890”。
