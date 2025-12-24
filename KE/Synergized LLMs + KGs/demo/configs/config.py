"""本地环境变量加载工具。

目标：
- 允许通过本地配置文件替代 `export`，便于复现实验与演示；
- 使用 `.gitignore` 过滤敏感文件，避免密钥入库；
- 仅在环境变量未设置时写入，保留外部注入（CI/容器）优先级。

支持格式：
- KV：`KEY=VALUE`（`.env` / `.env.local`）
- JSON：`{"KEY": "VALUE"}`
"""

import os
import json


def _load_kv_file(path: str):
    """加载 KEY=VALUE 配置文件到环境变量。"""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _load_json_file(path: str):
    """加载 JSON 配置文件到环境变量。"""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for k, v in data.items():
            os.environ.setdefault(str(k), str(v))


def _parent_dirs(path: str):
    cur = os.path.abspath(path)
    while True:
        yield cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent


def load_env():
    """按约定路径加载本地配置文件。

优先级说明：
- 仅在环境变量未设置时写入（`os.environ.setdefault`），外部注入优先；
- 候选路径覆盖“项目根”与“demo/configs”两类常见运行方式。
    """
    cwd = os.getcwd()
    dirs = list(_parent_dirs(cwd))
    dirs.reverse()
    for d in dirs:
        _load_kv_file(os.path.join(d, ".env"))
        _load_kv_file(os.path.join(d, ".env.local"))
        _load_kv_file(os.path.join(d, "demo", "configs", ".env"))
        _load_kv_file(os.path.join(d, "demo", "configs", ".env.local"))
        _load_json_file(os.path.join(d, "demo", "configs", "env.json"))
        _load_json_file(os.path.join(d, "demo", "configs", "env.local.json"))
