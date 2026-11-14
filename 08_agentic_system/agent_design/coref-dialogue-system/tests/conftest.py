"""共享 fixtures。

关键约定：所有测试在「无 spaCy zh 模型」假设下运行（CI / 新环境不保证
模型已下载），demo._NER 强制置 None。NER 本身的正确性是 spaCy 的事，
本项目测试的是消解逻辑、状态记账与降级路径。
"""

import pytest

import demo


@pytest.fixture(autouse=True)
def _no_spacy(monkeypatch):
    """强制规则路径走纯代词表，保证测试与模型安装状态无关。"""
    monkeypatch.setattr(demo, "_NER", None)
    monkeypatch.setattr(demo, "_NER_TRIED", True)


@pytest.fixture()
def store():
    from entity_store import EntityStore

    return EntityStore()
