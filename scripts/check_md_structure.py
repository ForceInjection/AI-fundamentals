#!/usr/bin/env python3
"""Markdown 结构检查。

查的不是语法，是「渲染结果和写作意图不一致」。

纯语法 lint 拦不住这类问题：attention_kv_cache_formats.md 修复前的版本在
markdownlint 0.49.1 下退出码 0、零告警，因为 4 个反引号开、4 个反引号闭
完全符合 CommonMark。它只是把 §五 整节当成了代码。

所以这里改查结果——两条会让内容凭空消失的情况：

  1. 代码围栏没有闭合，其后所有内容被吞进代码块
  2. 中文序号章节断号，说明中间有内容没能渲染成标题

用法：
    python3 scripts/check_md_structure.py             # 扫描仓库全部 .md
    python3 scripts/check_md_structure.py a.md b.md   # 只查指定文件（pre-commit 用法）

退出码：有错误为 1，仅有警告为 0。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {'.git', 'node_modules', '.venv', 'venv', '__pycache__'}

CN_DIGITS = '一二三四五六七八九十'
SECTION_RE = re.compile(r'^([一二三四五六七八九十]+)、')
FENCE_OPEN_RE = re.compile(r'^ {0,3}(`{3,}|~{3,})')  # 开启围栏，后面可以带 info string
FENCE_CLOSE_RE = re.compile(r'^ {0,3}(`{3,}|~{3,})[ \t]*$')  # 闭合围栏，不能带 info string

PARSER = MarkdownIt('commonmark')


def is_closer(line: str, opener: str) -> bool:
    """line 能否闭合由 opener 开启的围栏：同种字符、不短于开启围栏、无 info string。"""
    matched = FENCE_CLOSE_RE.match(line)
    return bool(matched) and matched.group(1)[0] == opener[0] and len(matched.group(1)) >= len(opener)


def cn_to_int(text: str) -> int:
    """把「一」「十」「二十三」这类中文数字转成整数。"""
    if text == '十':
        return 10
    if '十' in text:
        tens, _, ones = text.partition('十')
        return (CN_DIGITS.index(tens) + 1 if tens else 1) * 10 + (
            CN_DIGITS.index(ones) + 1 if ones else 0
        )
    return CN_DIGITS.index(text) + 1


def check_text(text: str) -> tuple[list[str], list[str]]:
    lines = text.splitlines()
    tokens = PARSER.parse(text)
    errors: list[str] = []
    warnings: list[str] = []

    headings: list[tuple[int, int, str]] = []
    for i, tok in enumerate(tokens):
        if tok.type == 'fence':
            line_no = tok.map[0] + 1
            opened = FENCE_OPEN_RE.match(lines[tok.map[0]])
            # 围栏跑到文件末尾，且最后一行不是合法的闭合围栏，才是没闭合。
            # 闭合围栏正好落在最后一行时 map[1] 也等于总行数，不能只看 map。
            if tok.map[1] >= len(lines) and opened and not is_closer(lines[-1], opened.group(1)):
                errors.append(f'{line_no}: 代码围栏没有闭合，第 {line_no} 行之后的内容全部被当作代码')

            # 长围栏只有在正文里出现同种围栏字符时才需要；否则和普通三反引号等价，
            # 却让下一个编辑的人要多想一步（PR #33 留下的那行就是这样：
            # 开启是 3 个、闭合是 4 个，渲染没问题但读起来容易看错）。
            closed = FENCE_CLOSE_RE.match(lines[tok.map[1] - 1])
            if opened and closed:
                fence, closer = opened.group(1), closed.group(1)
                if len(fence) != len(closer):
                    warnings.append(
                        f'{line_no}: 围栏开启用了 {len(fence)} 个 {fence[0]}、'
                        f'闭合用了 {len(closer)} 个，两端应当一致'
                    )
                elif len(fence) > 3 and not re.search(re.escape(fence[0]) + r'{3,}', tok.content):
                    warnings.append(
                        f'{line_no}: 围栏用了 {len(fence)} 个 {fence[0]}，'
                        f'内部没有同种围栏字符，用 3 个即可'
                    )
        elif tok.type == 'heading_open' and i + 1 < len(tokens):
            headings.append((int(tok.tag[1]), tok.map[0] + 1, tokens[i + 1].content.strip()))

    for level in (2, 3):
        expected = 1
        for heading_level, line_no, title in headings:
            if heading_level != level:
                continue
            matched = SECTION_RE.match(title)
            if not matched:
                continue
            number = cn_to_int(matched.group(1))
            if number != expected:
                errors.append(
                    f'{line_no}: 章节序号断了：当前是「{title[:30]}」，期望第 {expected} 节。'
                    f'中间某一节多半被代码围栏或 HTML 注释吞掉了'
                )
            expected = number + 1

    return errors, warnings


def main(argv: list[str]) -> int:
    if argv:
        targets = [Path(a) for a in argv]
    else:
        targets = sorted(p for p in REPO_ROOT.rglob('*.md') if not SKIP_DIRS.intersection(p.parts))

    checked = errors_found = warnings_found = 0
    for path in targets:
        if not path.is_file() or path.suffix != '.md':
            continue
        checked += 1
        try:
            label = path.resolve().relative_to(REPO_ROOT)
        except ValueError:
            label = path

        errors, warnings = check_text(path.read_text(encoding='utf-8'))
        for message in errors:
            print(f'{label}:{message}')
            errors_found += 1
        for message in warnings:
            print(f'{label}:{message}（警告）')
            warnings_found += 1

    if errors_found or warnings_found:
        print()
    print(f'检查 {checked} 个文件：{errors_found} 个错误，{warnings_found} 个警告')
    return 1 if errors_found else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
