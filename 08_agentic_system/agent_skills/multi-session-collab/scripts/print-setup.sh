#!/usr/bin/env bash
# 打印多会话协作的启动命令与设置清单（只读：不修改任何配置、不启动会话）
set -euo pipefail

echo "== Claude Code 多会话协作 · 启动清单 =="
echo "版本要求：跨会话消息 v2.1.224+（macOS/Linux/WSL2）或 v2.1.234+（原生 Windows）"
if command -v claude >/dev/null 2>&1; then
  echo "本机 claude：$(claude --version 2>/dev/null | head -1)"
else
  echo "本机 claude：未检测到 claude 命令"
fi
echo

echo "--- 1) 四个窗口（普通终端）---"
cat <<'EOF'
claude --name reviewer   # 只读评审
claude --name modifier   # 可写修改
claude --name tester     # 跑测试
claude --name judge      # 汇总与裁决
EOF
echo

echo "--- 2) tmux 一键四窗（可选）---"
cat <<'EOF'
tmux new-session -d -s collab -n judge 'claude --name judge'
tmux new-window -t collab -n reviewer 'claude --name reviewer'
tmux new-window -t collab -n modifier 'claude --name modifier'
tmux new-window -t collab -n tester   'claude --name tester'
tmux attach -t collab
EOF
echo

echo "--- 3) 入站策略（按需写进各会话 settings 或 --settings）---"
cat <<'EOF'
交互式盯盘（默认）：不设置，按权限模式推导
无人值守 worker（-p）：crossSessionInbound = "accept"
审批框有效期：dialogExpiry = "never"（默认 5 分钟）
跨机发送要人工确认：isolatePeerMachines = true
EOF
echo

echo "--- 4) 自查 ---"
cat <<'EOF'
/list-agents   # 首行是本会话名，其余是可寻址会话
/status        # 查看本会话 Peer address
EOF
echo

echo "--- 5) 下一步 ---"
cat <<'EOF'
用任务卡下发（references/protocol.md），judge 按 rubric 出三档裁决；
排错查 references/setup.md 的故障表。
EOF
