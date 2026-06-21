# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

AI Fundamentals is a Chinese-language knowledge repository covering the full AI infrastructure stack: GPU architecture, CUDA programming, LLM theory, inference systems, cloud-native AI platforms, agentic systems, RAG, and more. All content is authored in Markdown.

- **License**: Apache 2.0
- **Content** is organized in semantically numbered top-level directories (`01_hardware_architecture/` through `11_ai_native_everything/`, plus `98_llm_programming/` and `99_misc/`). Each directory corresponds to a major topic area with its own `README.md` portal.
- **`02_dpu_programming/`、`02_gpu_programming/` 和 `02_npu_programming/`** share the `02_` prefix — all three are sub-modules under "底层计算与异构编程."
- **`AGENTS.md`** exists alongside this file and covers module-level architecture details for GitHub Copilot. This file focuses on project-level conventions that apply to all work in the repo.

## Commit conventions

This repo uses **Conventional Commits** with Chinese descriptions:

```
docs(scope): description
chore(scope): description
refactor(scope): description
feat(scope): description
```

Scopes are derived from directory/topic areas. Common scopes seen in the history: `readme`, `dpu`, `gpu`, `npu`, `training`, `llm-theory`, `rag`, `graph_rag`, `agentic`, `agent_infra`, `inference`, `kv_cache`, `vllm`, `reference_design`, `storage`, `gpu_manager`, `k8s`, `course`, `trae`, `multi_agent`, `ai-native`, `smart_customer_service`, `kvbm`, `submodule`. Look at `git log --oneline` for recent examples before committing.


## File conventions

- All top-level topic directories use zero-padded numeric prefixes (e.g., `01_`, `02_`, `03_`) to maintain ordering.
- Within a topic directory, files may use numeric prefixes for ordering (e.g., `01_concepts.md`, `02_practice.md`).
- Translated content appends a language suffix to the filename (e.g., `file.zh-CN.md`).
- Image assets live in `img/` at the repo root, or alongside the files that reference them within topic subdirectories.
- Interactive HTML visualizations (e.g., inference pipeline demos) are placed alongside the markdown documents they complement, in the same topic subdirectory.
- `README.md` files at directory roots serve as navigation portals and contain link trees to content within that directory.

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

## Companion media files

Markdown documents are frequently accompanied by:

- **`.pptx` slide decks** — PowerPoint presentations that mirror or expand on the markdown content. Placed in the same directory as the `.md` file. When creating new technical deep-dives, consider whether a companion slide deck would be helpful.
- **`.pdf` references** — Reference papers, whitepapers, or exported slide decks, typically in a `references/` subdirectory.
- **`.gif` previews** — Animated previews of interactive HTML visualizations, placed alongside the `.html` file.
- **`.ipynb` notebooks** — Jupyter notebooks with executable code demonstrations.

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

## Multi-IDE support

The repo supports multiple AI-assisted IDEs beyond Claude Code:

- **`.trae/`** — Trae IDE configuration (gitignored, created per-user)
- **`.qoder/`** — Qoder IDE configuration with `agents/` and `skills/` subdirectories (gitignored, created per-user)
- **`.claude/`** — Claude Code settings (gitignored, `settings.local.json` contains per-user permissions)

These directories are all in `.gitignore` — they are local development environments, not repo content.

## CI/CD

This repo has **no GitHub Actions workflows or CI pipelines**. There is no build step, no linting, and no automated testing. Content quality is maintained through manual review (using the `doc-reviewer` skill).
