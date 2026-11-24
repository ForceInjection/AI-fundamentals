#!/usr/bin/env python3
"""第 1 讲实验 1:观察并归类 vLLM 的进程拓扑。

用法:
    vllm serve Qwen/Qwen3-0.6B --enforce-eager   # 终端 A
    python3 process_topology.py                  # 终端 B

只依赖标准库。按进程命令行/title 里的特征词归类,对照课程公式:
    进程数 = A(API server) + DP(EngineCore) + N(Worker) [+ coordinator, DP>1 时]

注意:进程 title 由 vllm/v1/engine/core.py 的 run_engine_core 设置
(EngineCore / EngineCore_DP{rank});Worker 由 MultiprocExecutor 拉起。
平台上 setproctitle 的可见性不同(macOS 对 ps 的 title 暴露有限),
归类不出来时会打印原始 cmdline 供人工判断——这本身就是观察的一部分。
"""
from __future__ import annotations

import re
import subprocess
import sys

# 特征词 → 角色。顺序即优先级(先匹配更具体的)。
PATTERNS: list[tuple[str, str, str]] = [
    (r"EngineCore_DP\d+", "EngineCore (DP rank)", "进程 title 带 DP rank"),
    (r"EngineCore", "EngineCore", "run_engine_core 设置的进程 title"),
    (r"coordinator", "DP Coordinator", "DP>1 时的协调进程"),
    (r"WorkerProc|worker_main|from multiprocessing", "Worker", "MultiprocExecutor 拉起的执行进程"),
    (r"vllm serve|entry\.py|api_server|launchers", "API server", "vllm serve 主进程"),
    (r"EngineCoreProc|spawn_main", "EngineCore/Worker(待分辨)", "mp spawn 子进程,需看父进程"),
]


def collect_processes() -> list[tuple[int, str]]:
    """返回 (pid, cmdline) 列表;ps 输出格式在 macOS/Linux 间有差异,按列数自适应。"""
    out = subprocess.run(["ps", "-eo", "pid,command"], capture_output=True, text=True).stdout
    procs: list[tuple[int, str]] = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        pid, _, cmd = line.partition(" ")
        if pid.isdigit():
            procs.append((int(pid), cmd.strip()))
    return procs


def build_parent_map(procs: list[tuple[int, str]]) -> dict[int, int]:
    """ps -eo pid,ppid,command 在部分平台才有 ppid;这里退化用命令特征判断父子不可靠,
    改为二次调用 ps 取 ppid,失败则返回空表。"""
    out = subprocess.run(["ps", "-eo", "pid,ppid"], capture_output=True, text=True).stdout
    mapping: dict[int, int] = {}
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            mapping[int(parts[0])] = int(parts[1])
    return mapping


def classify(cmd: str) -> tuple[str, str] | None:
    lowered = cmd.lower()
    for pattern, role, why in PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE) or re.search(pattern, lowered):
            return role, why
    return None


def main() -> int:
    procs = collect_processes()
    parents = build_parent_map(procs)
    # 噪音过滤:grep/分页器等一次性命令的 cmdline 也可能含 vllm 字样
    noise = re.compile(r"\b(grep|less|more|tail|head|cat|tee|vi|vim|nvim)\b")
    hits: list[tuple[int, str, str, str]] = []
    for pid, cmd in procs:
        if noise.search(cmd):
            continue
        if "vllm" not in cmd.lower() and "EngineCore" not in cmd and "nanovllm" not in cmd.lower():
            continue
        result = classify(cmd)
        if result:
            role, why = result
            parent = parents.get(pid)
            pinfo = f" ← ppid={parent}" if parent else ""
            hits.append((pid, role, why, pinfo))
        else:
            hits.append((pid, "UNKNOWN", "含 vllm 字样但无特征匹配,人工判断", ""))

    if not hits:
        print("没找到 vLLM 相关进程。先在另一个终端运行: vllm serve <model>")
        return 1

    width = max(len(h[1]) for h in hits)
    print(f"{'PID':>7}  {'角色':<{width}}  判据 / 备注")
    print("-" * 72)
    for pid, role, why, pinfo in sorted(hits):
        print(f"{pid:>7}  {role:<{width}}  {why}{pinfo}")

    counts: dict[str, int] = {}
    for _, role, _, _ in hits:
        counts[role] = counts.get(role, 0) + 1
    print("-" * 72)
    print("按角色计数:", ", ".join(f"{k}×{v}" for k, v in sorted(counts.items())))
    print("对照公式: A(API) + DP(EngineCore) + N(Worker) [+ coordinator, DP>1]")
    print("注意: 单卡 uni executor 时 Worker 与 EngineCore 同进程,进程数会少于直觉。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
