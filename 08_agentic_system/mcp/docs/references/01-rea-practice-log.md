# REA 实践日志（文章事实底稿）

《[402MB 的证据、21KB 的答案：REA 拆解 Codex.app 实战](../02_reverse_engineer_anything_in_practice.md)》的原始记录。2026-10-16 于 macOS arm64（Darwin 25.6.0，16GB 内存，Node 26.11.1，npm 11.20.0）实测，工具为 `rea-agents` 6.5.0（当日经 npx 安装）。原始输出（303MB / 402MB 证据文件、3.9MB 清单等）留存在本机工作目录，未入库。

## 0. 环境与就绪检查

- `rea doctor`：221 行；`healthy: false`；13 项检查按三类分类：healthy / missing_analysis_engine / config_drift
- healthy：node 26.11.1、host 26.6.2
- missing_analysis_engine ×4（各带 remediation）：hopper（`HOPPER_LAUNCHER_PATH` 或 setup 安装）、ghidra（`GHIDRA_INSTALL_DIR`，要求 12.1.x 解压目录）、ida-registration（`REA_IDA_MCP_CONFIG`）、jadx（`REA_JADX_MCP_JAR`，注明「REA does not download or install it」）
- config_drift ×7：skill:identity（Installed REA skill identity is missing）+ 六个客户端注册（claude_code → `~/.claude.json`、codex → `~/.codex/config.toml`、cursor → `~/.cursor/mcp.json`、dsh → `~/.dsh/cordis.patch.yml`、opencode → `~/.config/opencode/opencode.json`、qoder → `~/.qoder/settings.json`）
- 本机 JDK 17（Ghidra 12.1.x 不满足）；结论：三层目标全部落在零引擎路径

## 1. ollama 热身（inspect-go-binary）

- 首跑（符号链接）被拒：`invalid_request` / `invalid_input` / retryable true
  - issues 原文：`Expected a regular file without a symlink: /opt/homebrew/bin/ollama`
- `realpath` → `/opt/homebrew/Cellar/ollama/0.40.3/libexec/ollama` 后成功；退出码 0
- 结果：go1.27.2；`github.com/ollama/ollama`；`v0.40.3+dirty`；25,320,034 字节；sha256 `b2a15c47ba2dfbbb74535fb7ae019e91962e8b782d9a9d0f91653513799b9792`；module_text 67 行
- evidence 骨架：`confidence: observed`；`authority: shipped-artifact`；顶层 16 字段（subject / provider / limitations / evidence_id / normalized_result / raw_result=null / analysis_profile / relationship_coverage=null 等）
- limitations ×3（原文）：
  1. Embedded build metadata is observed file content, not proof of compiler identity, dependency authenticity or runtime behavior.
  2. Only Go build information is inspected; function names, source mappings, type metadata and decompiled code are not recovered.
  3. 资源护栏：稳定 artifact 读取与 header 搜索各 256 MiB、嵌入字符串合计 1 MiB、结构表 16 MiB、元数据 8 MiB；原文注明「these are not operating-system memory or CPU limits」

## 2. Codex.app：三道闸门与对照实验

目标：`/Applications/Codex.app` 1.2GB；`Resources/app.asar` 147MB；`Resources/codex` 236,141,904 字节（Rust）；`Resources/codex_chronicle` 4.3MB（Rust）；Codex Framework = 重命名的 Electron。版本 26.611.62324（build 4028）。asar 根 sha256 `5bda30b039746d731a0398403ae4628ef18385e4c057b58d86f1ecfd3c7ca166`。

### 2.1 完整性闸门

- 默认失败：`artifact_integrity_mismatch`（analyze 与 inspect-artifact 行为一致）
- 首例 details：logical_path `node_modules/@worklouder/device-kit-oai/…/node-hid/prebuilds/HID-darwin-arm64/node-napi-v4.node`；declared `ba4eee95e5b6226f19758addc1f03113d47698a5b0104804e94e66420f05496c` vs calculated `f38067bfd8db650fc22df90f2a860fb3e64ab31807b2ecccf56c40626cea80a3`；unpacked true
- 工具解释原文：packaging tools commonly sign or strip unpacked native binaries after writing the archive
- `--integrity-policy record-and-continue`：retain observed bytes as untrusted
- 全量 8 处（全部为 macOS 签名后的原生二进制）：
  1. node-hid prebuilds/HID-darwin-arm64/node-napi-v4.node
  2. node-hid prebuilds/HID-darwin-x64/node-napi-v4.node
  3. serialport bindings-cpp prebuilds/darwin-x64+arm64/node.napi.node
  4. better-sqlite3/build/Release/better_sqlite3.node
  5. node-pty/build/Release/pty.node
  6. node-pty/build/Release/spawn-helper
  7. objc-js/prebuilds/darwin-arm64/node.napi.armv8.node
  8. objc-js/prebuilds/darwin-x64/node.napi.node
- 形式化留档：contradictions[]（示例 contradiction_id `aic_081bfc96…`；statement「Declared integrity for … contradicts observed bytes.」）；limitation 原文「8 integrity contradiction(s) were recorded; mismatched content is observed-untrusted.」
- 公开旁证：`@worklouder/device-kit-oai` 对应 OpenAI × Work Louder「Codex Micro」键盘（约 $230，2026 年中发售）

### 2.2 inspect-artifact 清单（3.9MB）

- root-manifest：node_count 3,139 / occurrence_count 3,368 / edge_count 3,367；root_format asar
- 调查模型：observations(9) / hypotheses / contradictions(8) / unexplored_branches(1) / next_probes(1) / coverage{complete-within-substeps, retained 9, omitted 0}
- next_probes 原文：`{"operation":"analyze_javascript_application","rationale":"Recover bounded JavaScript modules, source maps, Electron boundaries, and application relationships.","authority":"ast-static-analysis"}`

### 2.3 资源闸门与对照实验

| 运行 | 差异                                     | parent_result_budget | 结果                                       | 输出                                            | 起止                         |
| ---- | ---------------------------------------- | -------------------- | ------------------------------------------ | ----------------------------------------------- | ---------------------------- |
| A    | 默认                                     | 1,940,736,048 B      | 2143/2372 处 `resource_constraint`，exit 1 | 302,876,942 B（错误封套 + partial_observation） | 16:10:13 → 16:14:12（3m59s） |
| B    | `NODE_OPTIONS=--max-old-space-size=6144` | 3,014,066,932 B      | 2372/2372，exit 0                          | 402,351,260 B                                   | 16:15:27 → 16:19:11（3m44s） |

- 断点原文：`The parent's result capacity was exhausted after retaining static facts for webview/assets/thread-side-panel-tabs-CvFaBaXm.js; remaining sources were not analyzed`
- 限额：worker_old_space_mb 1024、worker_heap_limit_bytes 1,098,907,648、worker_timeout_ms 300000；semantic_result_budget_bytes 0；semantic 上限 file 节点 20,000 / 全局 100,000
- 比值：Node 默认 heap_size_limit 4.09375 GiB（4,395,972,608 B）→ 44.2%；6144 MB 档 → 46.8%
- 失败运行 `details.partial_observation` 是完整 evidence 对象（含 graph / semantic_graph / statistics / summary / integrity_contradictions）
- 全量 summary：relevant_files 2372；text_bytes_read 113,459,082；parsed_javascript_files 2148；visited_ast_nodes 16,844,266；findings 47,798；modules 520；parse_failures 0；invalid_utf8 0；limitation_count 3,209；application/semantic coverage 均 `partial`
- Electron 边界（部分与全量完全一致）：browser_windows 1；explicit_web_preferences 5；preload_entrypoints 0；context_bridge_apis 2；ipc operations 26（literal 1 / dynamic 25）、renderer_transmissions 18、renderer_listeners 8、main_handlers 0、paired 0、unpaired_literal 1；sender_validation 13；native_addon_bindings 0
- unknown：application 3,191；semantic 261,341（部分运行分别为 3,875 / 170,134）
- 部分 → 全量：findings 42,807→47,798；parsed 1,921→2,148；视图覆盖 examined/total 2883/9671→4988/11170
- 语义图按 12 个关系族分账（boundary / call-flow / child-process / closure / configuration / data-flow / event / object-flow / promise-ownership / request / resource-lifecycle / timer），每族同时记 retained_relations 与 unknown_count（例：object-flow 一族 retained 1,452 / unknown 76,454，为部分运行口径）
- limitations 投影示例：「Dynamic IPC channel expressions remain unresolved and are never paired by textual similarity.」「Graph coverage is incomplete; unavailable facts remain explicit and must not be read as absence.」「3192 additional limitations omitted from this view; consult source parent Evidence ev_8676aeb9…」

## 3. 视图投影（inspect-analysis-view）

- 官方配方（cli.md）：保存全量证据 → `jq -c '{source: {kind: "inline", evidence: .}, view: {kind: "summary"}}'` → `rea inspect-analysis-view app-view.json`；官方例「334MB Obsidian → 摘要视图约 10KB」
- 本机产出：summary 21,342 B（全量）/ 24,696 B（部分）；page 22,717 B；item 18,828 B
- page 覆盖账目：examined 32 / total 4,988 / next*offset 32；行样例含 `jag_node*…`/`javascript-module`/`javascript-asset`
- item 字段：exports / hashes / identity / observations / path / source_ranges（3Dmol 模块为 1 range / 1 observation）
- 请求 schema 三次报错引导：合法 kind（18 个）→ collection（modules / sections / symbols）→ selector 对象
- 发布版差异：main 文档中的 `javascript-facts` 视图不在 6.5.0 合法 kind 列表

## 4. Rust 二进制（inspect-rust-binary）

- `codex`（236,141,904 B；sha256 `d88ade7335a752c81645a9dc610112ab46856c332d707e6c2a29dcd894b7624f`）
- `auditable_metadata: null`（无 `.dep-v0` 段）；limitations 原文：「No .dep-v0 section was found. Absence of .dep-v0 does not mean the binary has no dependencies.」
- `rustc_commit`：`evidence_kind: inferred`；`59807616e1fa2540724bfbac14d7976d7e4a3860`（file_offset 166,865,522 处的 `/rustc/<40-hex>/` 路径串）；「does not by itself prove which compiler produced the binary」

## 5. 环境备忘

- 安装：`npx -y rea-agents@latest`（6.5.0）；npm engines：`^22.19.0 || ^24.11.0 || >=26.0.0`
- 命令编目 `--help` 176 行；含 `--llms`（LLM 可读清单）与 `--token-count / --token-limit / --token-offset`
- 文档克隆基准：`morluto/rea`，2026-10-16，最新提交 `2c75835`
- 星标快照：276,649 star / 69,716 fork / 954 watcher / 1,039 open issues（GitHub API，2026-10-16）
