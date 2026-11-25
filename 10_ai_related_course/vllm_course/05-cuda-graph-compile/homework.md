# 第 5 讲课后作业

## 作业（必做两题）

**题 1（概念对照）**：nano 的 `capture_cudagraph` 有三个硬编码折中（整段录像、prefill 不进图、有限尺寸集）。对每一项，写出 vLLM V2 的对应解法、机制落点（标注来源文件即可）、以及「解掉了什么、又引入了什么新代价」。示例格式：

| nano 折中      | V2 解法                                | 落点          | 解掉                  | 新代价                      |
| -------------- | -------------------------------------- | ------------- | --------------------- | --------------------------- |
| prefill 不进图 | `FULL_AND_PIECEWISE`（prefill 走分段图） | `vllm.py:359` | prefill 也能消 launch | 图更碎，dispatch 逻辑更复杂 |

（第二、三行自己完成。）

**题 2（排障决策树）**：为三个故障场景各写一条二分定位路径（用讲义的二分表，写出每步旗标与预期现象）：

1. 用户报告「-O2 起服务比 --enforce-eager 慢，怀疑 capture 出问题」；
2. 用户报告「某固定 batch size 触发 CUDA graph 断言失败」；
3. 用户报告「Inductor 编译报 graph break，但不知道断在哪」。

每条路径最后给一句「若三种二分都无效，你的下一步」。

## 附加题（不计考核）

实验 1 里态 3(-O2)与态 2(-O0)的启动耗时差主要来自编译与 capture。设计一个量化实验把「编译耗时」与「capture 耗时」分开(提示：`-cc.mode=0` 关编译不关图是不可能的组合——想清楚为什么，再从 `-O1`（编译+PIECEWISE）与默认(`FULL_AND_PIECEWISE`)的启动差入手，结合日志「Graph capturing finished in X secs」)。写出实验方案与预期结果。

## 自测题（不提交）

1. 为什么 piecewise 图不需要 request padding，而 FULL 图需要？（提示：`cudagraph_utils.py:367-375` 与图内地址固定）
2. `--enforce-eager` 与 `-O0` 的区别是什么？各关掉哪一层？
3. 两族 capture 网格（decode 请求族 / prefill token 族）为什么不合并成一张表？
