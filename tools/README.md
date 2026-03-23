# 工具使用指南

本文档旨在详细说明 `tools` 目录下各自动化工具的功能特性与使用方法。

## 1. Markdown 链接检查工具

`check_links.py` 是一个基于 Python 开发的自动化脚本，专门用于验证 Markdown 文档中超链接的有效性。该工具不仅支持标准的 Markdown 链接语法，还兼容 HTML 图片标签（`<img>`）的解析。在设计上充分考虑了文件名中包含空格、URL 编码处理以及 Git 子模块未初始化等复杂场景。对于外部网络地址，工具内置了针对反爬虫机制的智能重试逻辑（如处理 403、404 等状态码）。每次执行后，系统将在项目根目录自动生成一份详细的检测报告（默认输出为 `link_check_report_v2.txt`）。

### 1.1 命令行参数说明

该脚本通过 `argparse` 模块提供灵活的命令行交互能力，主要支持以下参数：

| 参数名称 | 简写 | 选项范围 | 默认值 | 功能描述 |
| --- | --- | --- | --- | --- |
| `--file` | `-f` | 文件路径 | 项目根目录的 `README.md` | 指定要检查的单个 Markdown 文件路径。如果不指定任何目标参数，则默认检查项目根目录下的 `README.md`。 |
| `--dir` | `-d` | 目录路径 | 无 | 指定要检查的目录，将递归检查该目录及其子目录下的所有 Markdown 文件。 |
| `--all` | `-a` | 无 | 否 | 全量检查项目内的所有 Markdown 文件（相当于对项目根目录使用 `--dir`，执行时将自动跳过隐藏目录及环境依赖目录）。 |
| `--type` | `-t` | `local`, `external`, `all` | `local` | 指定需要检查的链接类型。`local` 表示仅检查本地文件路径，`external` 表示仅检查外部 URL 地址，`all` 表示两者都检查。 |
| `--help` | `-h` | 无 | 无 | 显示帮助信息并退出。 |

### 1.2 典型使用示例

以下代码块展示了如何通过终端运行该检查工具：

```bash
# 示例 1：默认行为，仅检查项目根目录下的 README.md 文件中的本地链接
python tools/check_links.py

# 示例 2：指定检查特定的 Markdown 文件中的所有链接
python tools/check_links.py -f path/to/your/file.md -t all

# 示例 3：递归检查某个目录及其子目录下的所有 Markdown 文件的本地链接
python tools/check_links.py -d 08_agentic_system

# 示例 4：全量检查项目中所有的 Markdown 文件，排查所有本地与外部链接
python tools/check_links.py -a -t all

# 示例 5：全量检查项目中所有的 Markdown 文件，但仅校验外部网络链接
python tools/check_links.py -a -t external
```
