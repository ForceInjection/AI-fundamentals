# 第 6 讲课后作业

## 作业（必做两题）

**题 1（部署决策报告）**：三个部署需求，各写「并行组合 + 完整部署命令 + 一句理由」:

1. 单机 8 卡 H200，跑 DeepSeek-V3（MoE，参考 `expert_parallel_deployment.md:85-96` 的示例形态）;
2. 2 机 16 卡（每机 8 卡）跑同一模型，decode 延迟优先；
3. 单机 8 卡跑 Qwen3-235B（H=4 个 kv-head），想省 KV 重复——DCP 能开多大?写出约束推导（`context_parallel_deployment.md` + `parallel.py:756-770`）。

**题 2（失效说法考古）**：从网上找一篇 2025 年前的 vLLM 并行教程（或 AI 生成的旧知识回答），逐条对照本讲八条失效表，标注：哪些说法在 HEAD 上仍成立、哪些已失效（附 `file:line` 证据）、哪些是文档与代码不一致（如「多机默认 Ray」）。三条以上，宁缺毋滥。

## 附加题（不计考核）

实验 1 组合 1（超卡数）的报错告诉你「需要 N 张卡但只有 M 张」。读 `parallel.py:1284-1296` 与 `docs/serving/parallelism_scaling.md:59-137`（Ray 集群节），回答：若真有 2 机 16 卡但不想用 mp 跨机，该怎么配（ray placement group 的写法）?给出与 mp+nnodes 两条路径的对比。

## 自测题（不提交）

1. `world_size` 为什么不含 DP?（提示：DP 是「复制」不是「切分」——每份 replica 的 world_size 内部还是要凑齐 TP×PP×PCP）
2. 为什么 PP 不参与 EP 组?（提示：PP 的各段是流水线前后关系，同一只专家只存在于某一个 PP 段里）
3. 纯 TP-EP（单机 TP=8 开 EP）时，token 的路由结果在各卡之间靠什么汇总?为什么不需要专家间通信?
