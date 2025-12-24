"""Moonshot LLM（Kimi）调用封装。

该模块提供两层能力：
1. `call_llm`：对 Moonshot Chat Completions 的最小封装；
2. `llm_verdict`：将图谱侧的结构化证据转为可审计的自然语言研判（JSON 输出）。

安全约束：
- 不在代码中硬编码密钥；
- 仅通过环境变量或本地 `.env.local`/JSON 配置文件加载密钥；
- 不打印或记录密钥内容。
"""

import os
import json

import httpx

from demo.configs.config import load_env

load_env()


MOONSHOT_API_URL = os.getenv("MOONSHOT_API_URL", "https://api.moonshot.cn/v1/chat/completions")
MOONSHOT_MODEL = os.getenv("MOONSHOT_MODEL", "kimi-k2-turbo-preview")


def call_llm(system_prompt: str, user_content: str, timeout: float = 15.0) -> str:
    """调用 Moonshot Chat Completions 并返回模型文本输出。

    参数：
    - system_prompt：系统指令，用于固定角色与输出规范；
    - user_content：用户输入内容（建议为结构化 JSON 字符串）；
    - timeout：HTTP 超时（秒）。

    返回：
    - 模型返回的 `choices[0].message.content` 字段。
    """
    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        raise RuntimeError("MOONSHOT_API_KEY not set")
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": MOONSHOT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(MOONSHOT_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def llm_verdict(evidence: dict, base_action: str) -> dict:
    """生成结构化研判结论（JSON）。

    该函数用于将图谱侧的确定性证据（`evidence`）与策略侧的初步建议（`base_action`）
    组织成结构化上下文，交给 LLM 生成“可审计”研判输出。

    约束：
    - 输出必须为 JSON，便于后续审计落库与回写图谱。
    """
    sys_prompt = "你是银行反电诈专家。仅返回JSON，字段：risk_type,evidence_points,verdict,suggested_action。"
    user_content = json.dumps(
        {
            "evidence": evidence,
            "base_action": base_action,
            "requirement": "结合证据生成可审计结论与处置建议，中文输出。",
        },
        ensure_ascii=False,
    )
    txt = call_llm(sys_prompt, user_content)
    return json.loads(txt)
