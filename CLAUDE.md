# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

AI Fundamentals is a Chinese-language knowledge repository covering the full AI infrastructure stack: GPU architecture, CUDA programming, LLM theory, inference systems, cloud-native AI platforms, agentic systems, RAG, and more. All content is authored in Markdown.

- **License**: Apache 2.0
- **Content** is organized in semantically numbered top-level directories (`01_hardware_architecture/` through `11_ai_native_everything/`, plus `98_llm_programming/` and `99_misc/`). Each directory corresponds to a major topic area with its own `README.md` portal.
- **`02_dpu_programming/`、`02_gpu_programming/` 和 `02_npu_programming/`** share the `02_` prefix — all three are sub-modules under "底层计算与异构编程."
- **`99_misc/`** hosts standalone project folders that don't fit a topic directory (e.g., `token_factory_talk/` — a complete talk project: outline + illustrated article + companion PPTX + page images in `img/` + numbered source notes in `references/`). This "project folder" pattern (article + PPTX + `img/` + `references/`) is reusable for any new talk or long-form deliverable.
- **`AGENTS.md`** exists alongside this file and covers module-level architecture details for GitHub Copilot. This file focuses on project-level conventions that apply to all work in the repo.

## Commit conventions

This repo uses **Conventional Commits** with Chinese descriptions:

```text
docs(scope): description
chore(scope): description
refactor(scope): description
feat(scope): description
```

Scopes are derived from the topic directory or subject area. Frequent scopes in the history: `readme`, `kv_cache`, `cuda`, `npu`, `inference`, `misc`, `vllm`, `memory`, `gpu`, `pcie`, `agentic`, `agentic_system`, `agent_infra`, `deployment`, `data-agent`, `sglang`, `theory`, `profiling`, `gpudirect`, `trae`, `rag`, `graph_rag`, `reference_design`, `storage`, `gpu_manager`, `k8s`, `course`. Scope naming has natural variation (e.g., `kv_cache` vs `kv-cache`, `agentic` vs `agentic_system`) — check `git log --oneline` for recent examples and match the dominant form for the area you're touching.

**No AI attribution trailers** — commit messages must not include `Co-Authored-By` or similar generated-by lines.

## File conventions

- All top-level topic directories use zero-padded numeric prefixes (e.g., `01_`, `02_`, `03_`) to maintain ordering.
- Within a topic directory, files may use numeric prefixes for ordering (e.g., `01_concepts.md`, `02_practice.md`).
- Translated content appends a language suffix to the filename (e.g., `file.zh-CN.md`).
- Image assets live in `img/` at the repo root, or alongside the files that reference them within topic subdirectories.
- Interactive HTML visualizations (e.g., inference pipeline demos) are placed alongside the markdown documents they complement, in the same topic subdirectory.
- `README.md` files at directory roots serve as navigation portals and contain link trees to content within that directory. **When adding a new article, you must update the corresponding directory's `README.md` portal** to include a link to the new file — this is the primary navigation mechanism for readers.

## Content creation workflow

When creating a new technical article, follow this sequence:

1. **Plan** — use `tech-outline-planner` to design the article structure with the C-I-S-T (Context → Issue → Solution → Trade-off) framework.
2. **Write** — create the `.md` file in the appropriate topic directory with a numeric prefix and Chinese descriptive filename.
3. **Link** — add the new article to the parent directory's `README.md` link tree.
4. **Review** — use `doc-reviewer` (outline + content + format) to catch structural, accuracy, and formatting issues.
5. **Polish** (external-facing docs) — use `humanizer-zh` to strip AI-writing tells before publishing; recent commits have applied this to outward-facing documents.
6. **Validate** — use `md-link-checker` to ensure all local and external links are accessible.
7. **Commit** — use `update-submitter` to generate a Conventional Commit message and submit.

## Python demos and notebooks

Some directories contain small Python projects and Jupyter notebooks for demonstration purposes. Each is self-contained and may include its own `.venv/` (gitignored). Notable locations:

- `04_cloud_native_ai_platform/gpu_manager/code/` — GPU scheduler and virtualization examples
- `07_rag_and_tools/synergized_llms_kgs/demo/` — Anti-fraud system demo (LLM + KG)
- `08_agentic_system/memory/langchain/code/` — LangChain memory demos
- `09_inference_system/memory_calc/` — Memory calculation scripts
- Scattered `*.ipynb` notebooks in `05_model_training_and_fine_tuning/`, `07_rag_and_tools/`, `98_llm_programming/`

These are primarily educational references, not a cohesive application. There is no top-level build system, linter, or test runner.

## Markdown links

- Local links between documents use **relative paths**.
- External links must remain accessible; validate with the `md-link-checker` Skill when modifying link-heavy files.
- When restructuring documents or moving files, update all cross-references.

## Writing conventions

- **All content is in Chinese** (Simplified). Code comments, commit descriptions, and directory README portals are also in Chinese.
- Major section headings in long-form articles often use **Chinese numerals** (一、二、三…) rather than Arabic numbers. Follow the existing heading style of the document you are editing.
- Article series that follow a numbered sequence (e.g., `09_inference_system/reference_design/`) use zero-padded numeric prefixes with Chinese descriptive filenames: `01-背景与目标.md`, `02-集群规模分类与特征分析.md`. Maintain this convention when adding new entries to an existing series.
- Interactive HTML visualizations (e.g., inference pipeline demos) placed alongside the markdown documents they complement should include a `.gif` preview in the same directory when possible.
- **Time-sensitive data** (prices, benchmarks, model releases, market stats): record the as-of date in the document, and mark vendor-claimed figures vs independently measured ones (e.g., 「厂商口径」). Add a 复核 reminder when data is fast-moving (see `99_misc/token_factory_talk/README.md` for the pattern).

## Source-code-based deep-dive articles

When writing technical deep-dives that reference external codebases (vLLM, SGLang, etc.), follow these conventions:

- **Verify every claim against source code.** Never describe a mechanism from memory or secondary sources alone — read the actual source file and confirm line numbers, method signatures, and behavior. If the referenced codebase is not available locally, state this explicitly and fall back to public documentation.
- **Use `file_path:line_number` format** for all source references (e.g., `vllm/distributed/eplb/eplb_state.py:526-658`). Line ranges should reference specific methods or logic blocks, not entire files.
- **Include a source file index** at the end of the article, listing every referenced file with its key classes/functions.
- **Prefer code excerpts over prose descriptions** for critical mechanisms. Show the actual code (simplified pseudocode is acceptable for readability, but the behavior must match the source).
- **Be honest about gaps.** If the target framework (e.g., vLLM) doesn't support a feature described in a reference article (e.g., w4a8c8 quantization), mark it clearly as "not available" rather than inventing a workaround.
- **Article structure**: Context → per-technique source analysis (mechanism + code + config) → maturity assessment → practical configuration → source file index.

Commonly referenced codebases and their local paths:

| Codebase | Local path |
|---|---|
| vLLM | `/Users/wangtianqing/Project/ai-infra/vLLM/` |
| SGLang | `/Users/wangtianqing/Project/ai-infra/sglang/` |
| LMCache | `/Users/wangtianqing/Project/ai-infra/LMCache/` |

## Article lifecycle

- When a new source-code-verified article **supersedes** an older theoretical/estimation-based article on the same topic, **delete the old article** and update all references (directory README, top-level README). Do not keep both — conflicting information from outdated articles misleads readers.
- Update the parent directory's `README.md` portal whenever adding, removing, or renaming an article. Also check the top-level `README.md` for stale links to the changed file.

## Companion media files

Markdown documents are frequently accompanied by:

- **`.pptx` slide decks** — PowerPoint presentations that mirror or expand on the markdown content. Placed in the same directory as the `.md` file. When creating new technical deep-dives, consider whether a companion slide deck would be helpful. To produce a page-by-page illustrated article ("PPT 逐页配图"), render the deck with `soffice --headless --convert-to pdf`, then `pdftoppm -jpeg -r 110` — output filenames are zero-padded (`slide-01.jpg`). Store page images in a sibling `img/` directory named by page (`01.jpg`…, `cover.jpg` for the title slide).
- **`references/` source notes** — when building a talk or deep-dive from external material, capture each source as a numbered note (`01-xxx.md`, `02-xxx.md`…) in a `references/` subdirectory; WeChat 公众号 articles are a common source (use `wechat-article-downloader`).
- **`.pdf` references** — Reference papers, whitepapers, or exported slide decks, typically in a `references/` subdirectory.
- **`.gif` previews** — Animated previews of interactive HTML visualizations, placed alongside the `.html` file.
- **`.ipynb` notebooks** — Jupyter notebooks with executable code demonstrations.

`.pptx` decks in this repo are frequently hand-edited by the user in PowerPoint — before any scripted edit, re-read the file from disk rather than relying on an earlier read.

## Project-specific skills

This repo has a rich set of Skills available for content authoring and review. Use them when the task matches:

| Skill | When to use |
|---|---|
| `doc-reviewer` | Review markdown docs — supports outline, content, asset, and format review types |
| `md-link-checker` | Validate local and external links in markdown files |
| `md-translator` | Translate markdown files to another language (adds language suffix to filename) |
| `md-summarizer` | Generate structured Chinese summaries of markdown documents |
| `tech-outline-planner` | Plan and structure new technical articles using context-first + process-narrative approach |
| `update-submitter` | Analyze git changes and generate Conventional Commit messages |
| `reference-organizer` | Format and organize reference links into structured citations |
| `humanizer-zh` | Rewrite AI-flavored Chinese prose into natural human writing — apply to external-facing docs before publishing |
| `pptx-reader` | Extract text and render slides from companion `.pptx` decks |
| `pptx-editor` | Shape-targeted text edits in existing `.pptx` files, with render verification |
| `wechat-article-downloader` | Download WeChat 公众号 articles as Markdown/HTML source material for `references/` notes |
| `web-content-downloader` | Download arbitrary web pages to Markdown, preserving original language |

## Multi-IDE support

The repo supports multiple AI-assisted IDEs beyond Claude Code:

- **`.trae/`** — Trae IDE configuration (gitignored, created per-user)
- **`.qoder/`** — Qoder IDE configuration with `agents/` and `skills/` subdirectories (gitignored, created per-user)
- **`.claude/`** — Claude Code settings (gitignored, `settings.local.json` contains per-user permissions)

These directories are all in `.gitignore` — they are local development environments, not repo content.

## CI/CD

This repo has **no GitHub Actions workflows or CI pipelines**. There is no build step, no enforced linting, and no automated testing. Content quality is maintained through manual review (using the `doc-reviewer` skill).

A local `.markdownlint.yaml` (gitignored — personal preference, not enforced) relaxes the markdownlint rules that clash with Chinese technical writing: line length (MD013), inline HTML (MD033), first-line heading (MD041), table pipe style (MD060), emphasis marker style (MD049), `$` in shell blocks (MD014), image alt text (MD045), and duplicate headings across sections (MD024 `siblings_only`). Don't reformat existing prose to satisfy markdownlint defaults.
