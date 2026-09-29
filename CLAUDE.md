# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

AI Fundamentals is a Chinese-language knowledge repository covering the full AI infrastructure stack: GPU architecture, CUDA programming, LLM theory, inference systems, cloud-native AI platforms, agentic systems, RAG, and more. All content is Markdown.

- **License**: Apache 2.0
- **Structure**: semantically numbered top-level directories (`01_hardware_architecture/` … `11_ai_native_everything/`, plus `98_llm_programming/` and `99_misc/`), each with its own `README.md` portal. `02_dpu_programming/`, `02_gpu_programming/`, `02_npu_programming/` share the `02_` prefix — all three are sub-modules under "底层计算与异构编程".
- **`99_misc/`** hosts standalone project folders (e.g., `token_factory_talk/`: outline + illustrated article + PPTX + `img/` + `references/`). This "project folder" pattern is reusable for any new talk or long-form deliverable.
- **Directory map**: `01_hardware_architecture` 硬件架构与互联(GPUDirect/PCIe/NVLink) · `02_dpu_programming` DPU/DOCA 编程 · `02_gpu_programming` GPU 编程基础(CUDA 范式与调优) · `03_ai_cluster_ops` 集群运维与通信(IB 网络/NCCL/GPU 运维) · `04_cloud_native_ai_platform` 云原生 AI(K8s/GPU 池化 HAMi/调度) · `05_model_training_and_fine_tuning` 训练与微调(SFT 实践) · `06_llm_theory_and_fundamentals` LLM 理论(量化/MoE/Embedding) · `07_rag_and_tools` RAG 与工具(KG/GraphRAG/PDF 解析/分块) · `08_agentic_system` 智能体(多 Agent/记忆/MCP/上下文工程) · `09_inference_system` 推理系统(KV Cache/LMCache/vLLM) · `10_ai_related_course` 课程课件与讲稿 · `11_ai_native_everything` AI Native 工程实践(FDE 案例) · `98_llm_programming` LLM 编程(LangGraph/Spring AI/Harness) · `99_misc` 独立项目与杂项
- **`AGENTS.md`** is a stable one-line pointer to this file — Copilot/Trae/Qoder auto-read it, so don't grow it; edit here instead.

## Commit conventions

**Conventional Commits** with Chinese descriptions: `docs(scope):`, `chore(scope):`, `refactor(scope):`, `feat(scope):`.

Scopes come from the topic directory or subject area (`inference`, `kv_cache`, `vllm`, `sglang`, `cuda`, `npu`, `agent_infra`, `readme`, …). Naming varies (`kv_cache` vs `kv-cache`) — run `git log --oneline` and match the dominant form for the area you're touching.

**No AI attribution trailers** — commit messages must not include `Co-Authored-By` or similar generated-by lines.

## File conventions

- Top-level topic directories use zero-padded numeric prefixes (`01_`, `02_`…); files within a topic may too (`01_concepts.md`, `02_practice.md`). Numbered series use prefix + Chinese descriptive filename (`01-背景与目标.md`).
- Translated content appends a language suffix (`file.zh-CN.md`).
- Images live in `img/` at the repo root or alongside the files that reference them.
- Interactive HTML visualizations sit beside the markdown they complement; include a `.gif` preview in the same directory when possible.
- **Every directory root has a `README.md` portal with a link tree. When adding, removing, or renaming an article, update the parent `README.md` and check the top-level `README.md` for stale links** — this is the primary navigation mechanism for readers.
- Local links use **relative paths**; validate link-heavy files with `md-link-checker`.
- When restructuring or moving files, update all cross-references.

## Skills (user-level: `~/.claude/skills/`, not repo content)

### 写文章线 —— 全流程打包为 `tech-article-pipeline`,按序调用这些;单项任务也可单独使用

| 阶段     | Skill                  | 用途                                                                                       |
| -------- | ---------------------- | ------------------------------------------------------------------------------------------ |
| 事实核对 | （pipeline 内建协议）  | 写前逐条核对源码/一手文档,协议在 `tech-article-pipeline/references/fact-check-protocol.md` |
| 大纲     | `tech-outline-planner` | C-I-S-T 框架,标题候选 + 章节结构                                                           |
| 评审     | `doc-reviewer`         | 四种独立评审:大纲 / 内容 / 资产与链接 / 格式                                               |
| 去 AI 味 | `humanizer-zh`         | 去除中文 AI 写作痕迹——对外发布的文档必过                                                   |
| 链接校验 | `md-link-checker`      | 本地 + 外链连通性(`-t all`)                                                                |
| 结构闸门 | `md-structure-checker` | 围栏闭合 / 中文序号断号——本仓库 pre-commit 调它的脚本                                      |

### 素材与媒体线 —— 按任务单独调用

| Skill                             | 用途                                            |
| --------------------------------- | ----------------------------------------------- |
| `wechat-article-downloader`       | 公众号文章下载为 Markdown/HTML,存 `references/` |
| `web-content-downloader`          | 网页转 Markdown                                 |
| `pptx-reader` / `pptx-editor`     | 提取 PPTX 文本 / 逐 shape 编辑 + 渲染验证       |
| `md-translator` / `md-summarizer` | 翻译(文件名加语言后缀) / 结构化中文摘要         |
| `reference-organizer`             | 参考链接整理成结构化引用                        |
| `update-submitter`                | 从 git 变更生成 Conventional Commit             |

## Content creation workflow

The full sequence (素材 → 事实核对 → 大纲 → 写作 → 门户 → 去 AI 味 → 校验 → 提交) is packaged as the **`tech-article-pipeline`** skill — invoke it for "produce a submittable article from a topic, link, or repo". It orchestrates the 写文章线 skills above and stops at gates: 大纲与标题定稿前、引用无法核实的断言前,必须停下等用户确认。

**Article lifecycle**: when a new source-verified article _supersedes_ an older estimation-based article on the same topic, **delete the old article** and update all references (directory README, top-level README). Do not keep both — conflicting information misleads readers.

## Writing conventions

- **All content is Chinese** (Simplified), including code comments, commit descriptions, and README portals.
- Long-form articles often use **Chinese numerals** for major headings (一、二、三…). Follow the existing heading style of the document you're editing.
- **Time-sensitive data** (prices, benchmarks, model releases, market stats): record the as-of date, mark vendor-claimed vs independently measured figures (e.g. 「厂商口径」), and add a 复核 reminder when data moves fast (see `99_misc/token_factory_talk/README.md`).
- **Math formulas (`$$`)**: never put underscores inside `\text{}` — GitHub restores `\_` to a bare `_` before handing TeX to its math renderer, which then fails with `'_' allowed only in math mode` (local MathJax tolerates it, so it passes local checks and only breaks on GitHub). Use spaces or short words in subscripts (`n_{\text{kv heads}}`, `\text{bytes}`); wrap bare notation in prose (`c^KV`, `d_c`) in code spans. Self-check with `grep -n '\$\$.*\\text{[^}]*_'`.

## Source-code-based deep-dive articles

- **Verify every claim against source code** — read the actual file and confirm line numbers, method signatures, and behavior. If the codebase isn't available locally, say so explicitly and fall back to public documentation.
- **Use `file_path:line_number`** for source references (e.g., `vllm/distributed/eplb/eplb_state.py:526-658`); point at methods or logic blocks, not whole files.
- **Include a source file index** at the end, listing every referenced file with its key classes/functions.
- **Prefer code excerpts over prose** for critical mechanisms; simplified pseudocode is acceptable if the behavior matches the source.
- **Be honest about gaps** — mark unsupported features as "not available" rather than inventing a workaround.
- **Structure**: Context → per-technique source analysis (mechanism + code + config) → maturity assessment → practical configuration → source file index.

Commonly referenced codebases and their local paths:

| Codebase | Local path                                      |
| -------- | ----------------------------------------------- |
| vLLM     | `/Users/wangtianqing/Project/ai-infra/vLLM/`    |
| SGLang   | `/Users/wangtianqing/Project/ai-infra/sglang/`  |
| LMCache  | `/Users/wangtianqing/Project/ai-infra/LMCache/` |

## Companion media files

- **`.pptx` decks** sit beside the `.md`. For page-by-page illustrated articles, render with `soffice --headless --convert-to pdf`, then `pdftoppm -jpeg -r 110`; store images in a sibling `img/` (`cover.jpg`, `01.jpg`…). **Decks are often hand-edited by the user in PowerPoint — re-read from disk before any scripted edit.**
- **`references/` source notes** — one numbered note per source (`01-xxx.md`, `02-xxx.md`…); 公众号 articles via `wechat-article-downloader`.
- **`.pdf` references** (papers, whitepapers, exported decks) and **`.ipynb` notebooks** (executable demos) sit in the topic directories.

## Python demos and notebooks

Self-contained educational Python projects and notebooks, each possibly with its own `.venv/` (gitignored): `04_cloud_native_ai_platform/gpu_manager/code/`, `07_rag_and_tools/synergized_llms_kgs/demo/`, `08_agentic_system/memory/langchain/code/`, `09_inference_system/memory_calc/`, plus scattered `*.ipynb` in `05_`, `07_`, `98_`. Not a cohesive application — no top-level build system, linter, or test runner.

## Multi-IDE support

`.trae/` and `.qoder/` are gitignored per-user IDE configs. `.claude/` holds Claude Code settings; **only `settings.local.json` is gitignored**, so anything else added under `.claude/` will be tracked by git.

## CI/CD

No build step, no test suite, no repo-owned workflow files — what runs (CodeQL, Pages build, Dependabot, Dependency Graph) is GitHub default setup, invisible in the tree. "This repo has no CI" is wrong — check `gh workflow list --all` first.

**Structure check retired from CI (2026-09-29)**: the script moved to the author's skill `md-structure-checker` (GitHub runners can't reach `~/.claude/skills/`), so the workflow was deleted. Remaining coverage: the pre-commit hook (`.pre-commit-config.yaml`) calls the skill's script and skips gracefully when the skill is absent — **external contributors' PRs are no longer structure-checked**. It flags the two content-vanishing modes markdownlint cannot see: unclosed code fence, and broken `## 一、` → `## 二、` sequence (the original case, `attention_kv_cache_formats.md` fixed in `20127cb`, passes markdownlint clean because 4-backtick-open + 4-backtick-close is valid CommonMark — only the render shows it). Long-fence inconsistencies are warnings.

A local `.markdownlint.yaml` (gitignored — personal preference) relaxes rules that clash with Chinese technical writing: MD013, MD033, MD041, MD014, MD024 `siblings_only`, MD045, MD049, MD060. Don't reformat existing prose to satisfy markdownlint defaults.
