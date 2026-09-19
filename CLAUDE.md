# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

AI Fundamentals is a Chinese-language knowledge repository covering the full AI infrastructure stack: GPU architecture, CUDA programming, LLM theory, inference systems, cloud-native AI platforms, agentic systems, RAG, and more. All content is Markdown.

- **License**: Apache 2.0
- **Structure**: semantically numbered top-level directories (`01_hardware_architecture/` … `11_ai_native_everything/`, plus `98_llm_programming/` and `99_misc/`), each with its own `README.md` portal. `02_dpu_programming/`、`02_gpu_programming/`、`02_npu_programming/` share the `02_` prefix — all three are sub-modules under "底层计算与异构编程".
- **`99_misc/`** hosts standalone project folders (e.g., `token_factory_talk/`: outline + illustrated article + PPTX + `img/` + `references/`). This "project folder" pattern is reusable for any new talk or long-form deliverable.
- **`AGENTS.md`** covers module-level architecture details for GitHub Copilot; this file focuses on project-level conventions.

## Commit conventions

**Conventional Commits** with Chinese descriptions: `docs(scope):`, `chore(scope):`, `refactor(scope):`, `feat(scope):`.

Scopes come from the topic directory or subject area (`inference`, `kv_cache`, `vllm`, `sglang`, `cuda`, `npu`, `agent_infra`, `readme`, …). Naming varies (`kv_cache` vs `kv-cache`, `agentic` vs `agentic_system`) — run `git log --oneline` and match the dominant form for the area you're touching.

**No AI attribution trailers** — commit messages must not include `Co-Authored-By` or similar generated-by lines.

## File conventions

- Top-level topic directories use zero-padded numeric prefixes (`01_`, `02_`…); files within a topic may too (`01_concepts.md`, `02_practice.md`).
- Translated content appends a language suffix (`file.zh-CN.md`).
- Images live in `img/` at the repo root or alongside the files that reference them.
- Interactive HTML visualizations sit beside the markdown they complement; include a `.gif` preview in the same directory when possible.
- **Every directory root has a `README.md` portal with a link tree. When adding, removing, or renaming an article, update the parent `README.md` and check the top-level `README.md` for stale links** — this is the primary navigation mechanism for readers.
- Local links between documents use **relative paths**. External links must stay accessible; validate with `md-link-checker` when touching link-heavy files.
- When restructuring or moving files, update all cross-references.

## Content creation workflow

The full sequence is packaged as the **`tech-article-pipeline`** skill — invoke it when the task is "produce a submittable article from a topic, link, or repo". The steps, and the repo-specific tools each uses:

1. **Verify before writing** — fact-check every claim against source code or primary documents, never from memory or secondary sources. Protocol: `tech-article-pipeline/references/fact-check-protocol.md`.
2. **Plan** — `tech-outline-planner` (C-I-S-T framework).
3. **Write** — `.md` in the right topic directory, numeric prefix + Chinese descriptive filename.
4. **Link** — update the parent `README.md` portal.
5. **Review** — `doc-reviewer` (outline + content + format).
6. **Polish** (external-facing docs) — `humanizer-zh` to strip AI-writing tells; recent commits have applied this before publishing.
7. **Validate** — `md-link-checker` for local and external links.
8. **Commit** — `update-submitter` for the Conventional Commit message.

**Article lifecycle**: when a new source-verified article *supersedes* an older estimation-based article on the same topic, **delete the old article** and update all references (directory README, top-level README). Do not keep both — conflicting information misleads readers.

## Writing conventions

- **All content is Chinese** (Simplified), including code comments, commit descriptions, and README portals.
- Long-form articles often use **Chinese numerals** for major headings (一、二、三…). Follow the existing heading style of the document you're editing.
- Numbered article series use zero-padded prefixes with Chinese descriptive filenames (`01-背景与目标.md`, `02-集群规模分类与特征分析.md`).
- **Time-sensitive data** (prices, benchmarks, model releases, market stats): record the as-of date, mark vendor-claimed vs independently measured figures (e.g. 「厂商口径」), and add a 复核 reminder when data moves fast (see `99_misc/token_factory_talk/README.md`).

## Source-code-based deep-dive articles

- **Verify every claim against source code** — read the actual file and confirm line numbers, method signatures, and behavior. If the codebase isn't available locally, say so explicitly and fall back to public documentation.
- **Use `file_path:line_number`** for source references (e.g., `vllm/distributed/eplb/eplb_state.py:526-658`); point at methods or logic blocks, not whole files.
- **Include a source file index** at the end, listing every referenced file with its key classes/functions.
- **Prefer code excerpts over prose** for critical mechanisms; simplified pseudocode is acceptable if the behavior matches the source.
- **Be honest about gaps** — mark unsupported features as "not available" rather than inventing a workaround.
- **Structure**: Context → per-technique source analysis (mechanism + code + config) → maturity assessment → practical configuration → source file index.

Commonly referenced codebases and their local paths:

| Codebase | Local path |
| --- | --- |
| vLLM | `/Users/wangtianqing/Project/ai-infra/vLLM/` |
| SGLang | `/Users/wangtianqing/Project/ai-infra/sglang/` |
| LMCache | `/Users/wangtianqing/Project/ai-infra/LMCache/` |

## Companion media files

- **`.pptx` decks** sit beside the `.md`. For page-by-page illustrated articles, render with `soffice --headless --convert-to pdf`, then `pdftoppm -jpeg -r 110`; store images in a sibling `img/` (`cover.jpg`, `01.jpg`…). **Decks are often hand-edited by the user in PowerPoint — re-read from disk before any scripted edit.**
- **`references/` source notes** — one numbered note per source (`01-xxx.md`, `02-xxx.md`…); WeChat 公众号 articles are a common source (use `wechat-article-downloader`).
- **`.pdf` references** — papers, whitepapers, or exported decks, typically in `references/`.
- **`.gif` previews** — animated previews of interactive HTML visualizations, alongside the `.html`.
- **`.ipynb` notebooks** — Jupyter notebooks with executable demonstrations.

## Python demos and notebooks

Self-contained educational Python projects and notebooks, each possibly with its own `.venv/` (gitignored): `04_cloud_native_ai_platform/gpu_manager/code/`, `07_rag_and_tools/synergized_llms_kgs/demo/`, `08_agentic_system/memory/langchain/code/`, `09_inference_system/memory_calc/`, plus scattered `*.ipynb` in `05_`, `07_`, `98_`. Not a cohesive application — there is no top-level build system, linter, or test runner.

## Project-specific skills

Use these when the task matches:

| Skill | When to use |
| --- | --- |
| `tech-article-pipeline` | End-to-end article pipeline: source → fact check → outline → write → portal → de-AI → links → commit |
| `doc-reviewer` | Review markdown docs (outline, content, asset, format) |
| `md-link-checker` | Validate local and external links |
| `md-translator` | Translate markdown (adds language suffix to filename) |
| `md-summarizer` | Structured Chinese summaries of markdown documents |
| `tech-outline-planner` | Plan article structure (context-first + process narrative) |
| `update-submitter` | Conventional Commit messages from git changes |
| `reference-organizer` | Format reference links into structured citations |
| `humanizer-zh` | Strip AI-writing tells from Chinese prose — apply to external-facing docs |
| `pptx-reader` | Extract text and render slides from `.pptx` decks |
| `pptx-editor` | Shape-targeted `.pptx` text edits with render verification |
| `wechat-article-downloader` | Download 公众号 articles as Markdown/HTML for `references/` notes |
| `web-content-downloader` | Download web pages to Markdown, preserving original language |

## Multi-IDE support

`.trae/` and `.qoder/` are gitignored per-user IDE configs — local development environments, not repo content. `.claude/` holds Claude Code settings; **only `settings.local.json` is gitignored**, so anything else you add under `.claude/` (e.g. `skills/`) will be tracked by git.

## CI/CD

No build step, no test suite. The one automated gate is a markdown **structure** check — `scripts/check_md_structure.py` (needs `markdown-it-py`) — wired into two places:

- **Pre-commit hook** (`.pre-commit-config.yaml`): catches your own commits. Needs `pre-commit install` once per clone; `pre-commit run --all-files` checks the whole repo.
- **GitHub Actions** (`.github/workflows/md-structure.yml`): catches pull requests. External contributors won't have the hook installed, so CI is the only thing covering them.

It flags two failure modes that silently drop content: an unclosed code fence, and a broken Chinese section sequence (`## 一、` → `## 二、` → …). Syntax linting catches neither — the original regression case (`attention_kv_cache_formats.md`, fixed in `20127cb`) passes markdownlint with zero warnings, because a 4-backtick fence closed by a 4-backtick fence *is* valid CommonMark. Only the rendered result gives it away. Long-fence inconsistencies are warnings and don't fail the build.

If you add hooks to `.pre-commit-config.yaml`, add them to the workflow too — CI runs the script directly, not through pre-commit.

Content quality is otherwise maintained by manual review (`doc-reviewer`).

A local `.markdownlint.yaml` (gitignored — personal preference, not enforced) relaxes the rules that clash with Chinese technical writing: line length (MD013), inline HTML (MD033), first-line heading (MD041), table pipe style (MD060), emphasis marker style (MD049), `$` in shell blocks (MD014), image alt text (MD045), and duplicate headings across sections (MD024 `siblings_only`). Don't reformat existing prose to satisfy markdownlint defaults.
