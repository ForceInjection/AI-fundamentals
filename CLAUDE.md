# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

AI Fundamentals is a Chinese-language knowledge repository covering the full AI infrastructure stack: GPU architecture, CUDA programming, LLM theory, inference systems, cloud-native AI platforms, agentic systems, RAG, and more. All content is authored in Markdown.

- **License**: Apache 2.0
- **Content** is organized in semantically numbered top-level directories (`01_hardware_architecture/` through `10_ai_related_course/`, plus `98_llm_programming/` and `99_misc/`). Each directory corresponds to a major topic area with its own `README.md` portal.
- **`AGENTS.md`** exists alongside this file and covers module-level architecture details for GitHub Copilot. This file focuses on project-level conventions that apply to all work in the repo.

## Commit conventions

This repo uses **Conventional Commits** with Chinese descriptions:

```
docs(scope): description
chore(scope): description
refactor(scope): description
feat(scope): description
```

Scopes are derived from directory/topic areas (e.g., `dpu`, `readme`, `inference`, `kv_cache`, `vllm`, `agentic`, `agent_infra`). Look at `git log --oneline` for recent examples before committing.

## Git submodule

`09_inference_system/nano-vllm` is a git submodule pointing to `https://github.com/ForceInjection/nano-vllm`. After cloning, run:

```bash
git submodule update --init --recursive
```

When the submodule reference is updated, commit it as `chore(submodule): <description>`.

## File conventions

- All top-level topic directories use zero-padded numeric prefixes (e.g., `01_`, `02_`, `03_`) to maintain ordering.
- Within a topic directory, files may use numeric prefixes for ordering (e.g., `01_concepts.md`, `02_practice.md`).
- Translated content appends a language suffix to the filename (e.g., `file.zh-CN.md`).
- Image assets live in `img/` at the repo root, or alongside the files that reference them within topic subdirectories.
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
