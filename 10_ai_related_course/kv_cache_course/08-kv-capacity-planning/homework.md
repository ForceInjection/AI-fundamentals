# 第 8 讲：课后作业

本讲作业即结课作业（作业 3），占总评 40%。结课项目是一页决策备忘，属于作业 3 的必做部分，不单独计分。

## 固定假设

| 项目 | 值 |
| --- | --- |
| 模型 | GLM-5.3-Flash |
| token 账 | scaled FP8，7,260 B/token 容量预留（含 11 层 KV + 11 层 indexer） |
| KDA 账 | fp32，147.6 MB/state slot，5 slots/request（预留上限），约 738 MB/request |
| 并行 | TP8 / DP8 / attn_tp = 1 / CP = 1 |
| 主机 DRAM | 1 TiB/节点 |
| 上下文 | 128K 含输出；输出上限 2K |
| prefix 复用率 | 60% 的输入 token 命中前缀 |
| page | 64 token |

每 rank 的 HBM 只算 KV/state 占用，不含模型权重和激活。

## 两个场景

1. **平均驻留**：全局并发 64，请求分布为 10% × 128K、40% × 32K、50% × 8K。
2. **峰值**：64 个请求同时按 128K 驻留。

两个场景都要算 token 池和 KDA 池，指出各自先爆的账本。

## 参考算例

平均 token 数：

```text
0.1 × 131,072 + 0.4 × 32,768 + 0.5 × 8,192 = 30,310.4 token/请求
```

平均驻留：

```text
token 池 = 64 × 30,310.4 × 7,260 B ≈ 14.1 GB
KDA 池  = 64 × 738 MB ≈ 47.2 GB
先爆：KDA
```

峰值：

```text
token 池 = 64 × 131,072 × 7,260 B ≈ 60.9 GB
KDA 池  = 64 × 738 MB ≈ 47.2 GB
先爆：token
```

注意这是单 rank、attn_tp = 1 的全局账。DP8 时每 rank 按 8 个请求分摊：平均驻留下 token ≈ 1.76 GB/rank、KDA ≈ 5.9 GB/rank；峰值下 token ≈ 7.6 GB/rank、KDA ≈ 5.9 GB/rank。每 rank 合计约 13.5 GB，不含权重。

## 要交的内容

### 1. 两张场景表

下表由学员填写，第一行是示例：

| 场景 | token 池（全局） | KDA 池（全局） | 每 rank token | 每 rank KDA | 先爆的账 |
| --- | --- | --- | --- | --- | --- |
| 示例：平均驻留 | 14.1 GB | 47.2 GB | 1.76 GB | 5.90 GB | KDA |
| 平均驻留 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 峰值 64×128K | 待填 | 待填 | 待填 | 待填 | 待填 |

表下回答：推荐配置按哪个场景定？为什么？另一个场景用来做什么？

### 2. page 开销

按 `ceil(len/page) × page − len` 算 len = 1,000 在 page = 16、64、256 下的碎片，再估算本负载的平均碎片开销。说明大 page 和小 page 各省什么。

### 3. DRAM 账

按对象列，不按 HBM 全量镜像：

- 11 层 KV：按 60% 复用率估；
- 34 层 KDA state：按每请求 1 个和 2 个 prefix checkpoint 两种情况估；
- indexer 和 compress tail：说明为什么不计入 HiCache；
- 给出总 DRAM 需求，对照 1 TiB 判断是否够用。

这里每个 prefix checkpoint 按 1 个 state slot（147.6 MB）估，不是把活跃请求的 5 slots 全部复制到 DRAM；5 slots 是调度预留，prefix checkpoint 数取决于实际保留的前缀节点。若你的实现按 5 slots 备份，DRAM 账要相应乘 5。

### 4. 一页决策备忘（结课项目）

- 选 SGLang 或 vLLM，说明理由；
- 给出 PD 分离 / offload / prefix cache 的取舍，每条理由带一个你测到或假设的数；
- 用一条真实启动日志对照容量表，并注明日志来源；没有真实日志时，可以用 `code/` 下的推演样例检查字段格式，但不能把它当作验证依据。

## 常见错误

- 把 KDA state 折算成每 token 字节，和 7,260 B/token 相加；
- 用平均驻留的结论去配峰值并发；
- 把峰值的 KV/state 合计（约 108 GB）当成 DRAM 需求；
- 忽略 page 对齐，直接用 `len × bytes_per_token`；
- 把 GLM-5 / GLM-5.3 / GLM-5.3-Flash 的数字混用，或把 V4-Pro / V4-Flash / V4.1-Flash 的数字混用。

## 提交格式

- Markdown 报告，不超过 4 页；
- 代码或脚本放在独立目录；
- 所有数字附公式或命令；
- 启动日志必须注明是真实测量还是推演样例。
