# 协作协议：任务卡、回报模板与裁决

## 1. 任务卡（judge → worker）

跨会话消息是纯文本，模板由调用方约定。下发时包含三件事：

```text
@<worker> 任务：<目标一句话>。
输入：<分支 / 文件路径 / 规格文档路径 / diff 命令>
输出格式（只输出 JSON，不修改任何文件）：
{"verdict":"pass|fail","findings":[{"line":42,"severity":"high","issue":"...","evidence":"file:line"}],"unknowns":["..."]}
时限：<N> 分钟内给结论；拿不准的写进 unknowns，不要猜。
```

要点：

- 文件必须写路径（`@` 引用在接收侧只是字面文本，不会附带文件）
- 「不修改任何文件」要显式写，防止 worker 越权
- 拿不准的进 `unknowns`，禁止用猜测填空

## 2. 回报模板（worker → judge）

```text
结论：<通过 / 不通过>
证据：<file:line 列表，逐条对应>
整改单：<文件行 + 具体改法 + 验收标准>
未知：<无法静态确认的部分>
```

judge 收到的消息带 `from` 字段，回信直接按发送方地址回发。等 worker 完成用 `notify_when_idle` 订阅一次空闲通知，不要轮询。

## 3. 结构化 findings

```json
{
  "findings": [
    {
      "id": "F-001",
      "file": "src/rbac/guard.ts",
      "line": 42,
      "severity": "high",
      "issue": "缓存命中的权限检查跳过了租户归属校验",
      "evidence": "src/rbac/guard.ts:42-58",
      "confidence": "observed",
      "verdict": "confirmed"
    }
  ],
  "unknowns": ["无法静态确认的运行时分支：..."]
}
```

- `confidence`：observed / inferred / unknown（证据分档，禁止把推断写成 observed）
- `verdict`：confirmed / plausible / rejected
- 每条 finding 必须带 `evidence`（file:line），否则视为未完成

## 4. Rubric（judge 用）

| 维度 | 问题 | 通过阈值 |
| --- | --- | --- |
| 正确性 | 结论能否被证据复现？ | 每条 finding 带 file:line |
| 完整性 | 声明的范围都检查了吗？ | 覆盖清单无缺口，缺口写进 unknowns |
| 风险 | 修改面是否超出裁决单？ | 只动授权文件 |
| 成本 | 重跑一次的 token 与时间？ | 在预算内（人定） |

## 5. 裁决三档

1. **通过**：修改进入下一环节
2. **打回**：附整改单（文件行 + 改法 + 验收标准），交给 modifier
3. **升级人工**：confidence 低于阈值、findings 互相矛盾、或涉及授权之外的改动

反模式：judge 自己改代码；没有 rubric 的「感觉评审」；把 teams 的 plan approval 当评审（它会被 lead 自动批准）。
