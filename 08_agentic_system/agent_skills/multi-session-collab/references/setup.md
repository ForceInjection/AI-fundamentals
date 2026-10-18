# 环境与配置：版本、命名、入站策略、排错

## 版本要求

- 跨会话消息：Claude Code **v2.1.224+**（macOS / Linux / WSL 2）或 **v2.1.234+**（原生 Windows）
- `/list-agents` 首行显示本会话名：**v2.1.239+**
- `@` 提及会话：**v2.1.232+**
- Agent Teams：实验特性，设置 `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` 后启用

## 后台会话当 worker（实测）

```bash
# 在已受信任的工作区启动；用 --add-dir 授权目标目录
claude --bg --name reviewer --add-dir /path/to/target --settings /path/to/settings.json
# 管理
claude agents            # 列出交互与后台会话
claude logs <id>         # 查看输出（消息被扣时也可从这里读回内容）
claude stop <id>         # 停止
```

- 后台会话会出现在 `/list-agents` 里（kind 为 `bg`），可以被 `SendMessage` 寻址
- `--bg` 要求工作区已受信任：未受信任的临时目录会被拒绝，先用受信任目录启动并用 `--add-dir` 授权
- `--settings` 推荐加 `{"crossSessionInbound": "accept"}`，确保无人值守时能收信

## 命名与发现

```bash
claude --name reviewer     # 启动时命名
/rename reviewer           # 会话内改名
```

- 人类查看：`/list-agents`（别名 `/peers`），第一行是本会话自己的名字
- Claude 侧：`ListAgents` / `SendMessage` 内置工具；也可用 `@名字` 提及（名字含空格加引号：`@"release notes"`）
- 只有绑定了收件 socket 的会话可寻址；bare 模式、容器/宿主、WSL 2/原生 Windows 之间互不可达
- 同名会话由系统附加短标识区分；`/status` 的 `Peer address` 可查本会话地址

## 入站策略（crossSessionInbound）

接收方设置：`accept` / `hold` / `refuse`（`/config` 的「Messages from your other sessions」或 `--settings`）。未显式设置时按权限模式推导：

| 接收方 | 发送方 | 结果 |
| --- | --- | --- |
| 会弹权限（默认、`acceptEdits`、`auto`、`dontAsk`） | 非 bypass | 直接送达 |
| 会弹权限 | bypass（交互式终端里的 plan 也算 bypass） | 扣留待批准 |
| bypass | bypass | 直接送达 |
| bypass | 非 bypass | 扣留待批准 |

相关设置：

- `dialogExpiry`：审批框有效期，默认 5 分钟，可设 `"never"`
- `isolatePeerMachines: true`：跨机发送需人工批准（同机不受影响）
- `-p` 会话无审批框：默认扣留的消息到期即丢；无人值守 worker 必须设 `crossSessionInbound: "accept"`

## 消息语义速查

- 接收方忙：在两次工具调用之间读取，不打断正在跑的工具
- 接收方空闲：开新回合处理
- 纯文本，不带对方历史与文件；超过约 100 万字符拒发；接收侧限流、去重、最多排队 50 条
- `notify_when_idle`：订阅一次空闲/退出通知，12 小时过期，仅主会话、仅同机

## 故障排查

| 现象 | 原因 | 处置 |
| --- | --- | --- |
| 对方不在 `/list-agents` | bare 模式无收件 socket；容器与宿主互不可达；WSL 2 与原生 Windows 互不可达；版本过旧 | 普通交互模式重启；跨边界改用同容器会话 |
| 消息一直不到 | 权限模式导致扣留 | 到接收会话处理审批框，或调整 `crossSessionInbound` |
| 消息 5 分钟后消失 | 审批框到期 | 调大 `dialogExpiry` 或设 `"never"`；`-p` worker 改设 `accept` |
| 名字冲突 | 同名会话 | 用列表里的短标识寻址，或 `/rename` 改名 |
| 云会话没有回信 | 云会话当前可收不可回 | 让云会话把结果写进文件，或经 Remote Control 接手 |

## 安全边界

- 消息不能代批权限、不能改配置或 CLAUDE.md、斜杠命令不执行；接收方权限规则仍然生效
- `SendMessage` / `ListAgents` 的 deny 规则会连带禁用 subagent 与 teammate 消息，团队环境慎用
- 跨机消息经 Anthropic 服务器中转，需要 claude.ai 登录与 Remote Control 条件；API key / Bedrock / Vertex / Foundry 下不可用
