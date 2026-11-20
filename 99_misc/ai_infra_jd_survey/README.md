# AI Infra / Agent 岗位 JD 数据集

国内 12 家 AI 公司「AI Infra / Agent / 大模型训练 / 大模型推理」相关在招 JD 的结构化数据集，用于多维度分析（技能词频、公司对比、方向差异、城市分布）。采集日期 2026-11-19，**6,819 条**（全部为 JD 全文；按招聘渠道分为社招 3,407 条、校招 3,412 条）。

**导读文章**：[01-岗位技能版图.md](01-岗位技能版图.md) — JD 的采集方法、社招与校招两张版图、技能三视角谱系与按岗位线的学习路线；[02-校招岗位与技能.md](02-校招岗位与技能.md) — 校招专册：方向可及性排名、公司与城市结构、校招技能谱系与实习入口。**技能分类体系的设计与判定规则**：[docs/技能分类体系设计.md](docs/技能分类体系设计.md)（kind / type / scope / work_type 四维定义）。**交互可视化**（按渠道分四个页面）：社招 [skill-map-society.html](skill-map-society.html)（技能地图）/ [jobs-society.html](jobs-society.html)（岗位版图）；校招 [skill-map-campus.html](skill-map-campus.html)（技能地图）/ [jobs-campus.html](jobs-campus.html)（岗位版图）。

## 数据文件

- `data/jd_2026-11-19.jsonl` — 主数据集，6,819 条，每行一条 JSON 记录（16 字段 schema 见下）。来源：`~/Downloads/AI_Companies_JD/` 的合并文件 `all_companies_ai_jobs.json`（6,978 条），经 `scripts/import_ai_companies.py` 转换为统一 schema（6 条 status=pause 跳过、137 条应用算法岗位剔除、16 条关键词误中的非 AI 岗按 `data/manual_exclusions.json` 人工剔除，见「分析口径注意」）。转换幂等可重跑。
- `data/skill_catalog.json` — 分类体系唯一定义源（岗位类别与细分的规则、技能 taxonomy 与家族、学习路线、双 matrix），脚本只引用不定义。
- `data/`、`analysis/`、`scripts/` 均为本地工具资产（.gitignore），不随仓库发布；仓库发布的明细数据见 [jobs-society.html](jobs-society.html) 与 [jobs-campus.html](jobs-campus.html) 内嵌的记录（社招 3,407 / 校招 3,412，合计 6,819 条）。

## 数据来源

12 家国内 AI 公司的官方招聘渠道（官网 / 官方 ATS），2026-11-19 抓取：

| 公司          | AI 相关条数 | 渠道                      | 公司     | AI 相关条数 | 渠道                |
| ------------- | ----------- | ------------------------- | -------- | ----------- | ------------------- |
| 字节跳动      | 2,914       | 飞书 ATS（社招+校招）     | 腾讯     | 719         | careers.tencent.com |
| 阿里巴巴      | 1,275       | talent.alibaba.com        | 美团     | 336         | zhaopin.meituan.com |
| 蚂蚁集团      | 474         | hrcareersweb.antgroup.com | 快手     | 314         | zhaopin.kuaishou.cn |
| 小红书        | 289         | job.xiaohongshu.com       | 阶跃星辰 | 152         | Moka（社招+校招）   |
| MiniMax       | 143         | 飞书 ATS                  | 智谱 AI  | 103         | 飞书 ATS            |
| 月之暗面 Kimi | 85          | Moka                      | DeepSeek | 15          | Moka                |

合计：抓取到 34,369 条（字节社招接口受 1 万条展示上限约束）→ AI 相关 6,978 条 → 转换入库 6,819 条（6 条 status=pause 跳过、137 条应用算法岗位剔除、16 条人工剔除）。百度和华为无公开数据，不在覆盖范围。

## Schema（每行一条）

| 字段               | 类型     | 说明                                                                                                                                                                                                                                                                       |
| ------------------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `company`          | string   | 公司名（12 家）                                                                                                                                                                                                                                                            |
| `title`            | string   | 职位名（照抄原文）                                                                                                                                                                                                                                                         |
| `category`         | enum     | `研发` / `算法` / `测试` / `SRE` / `运维` / `产品与运营`（由 catalog 的 sub_category.rules 派生：category = items[sub].categories[0]）                                                                                                                                     |
| `sub_category`     | enum     | 16 个标准方向（行业通用岗位名，定义见 `data/skill_catalog.json`，见下）                                                                                                                                                                                                    |
| `location`         | string   | 工作城市（`/` 分隔多城市）                                                                                                                                                                                                                                                 |
| `employment_type`  | enum     | `全职` / `实习`（源字段已归一为两值，直接采信）                                                                                                                                                                                                                            |
| `recruit_channel`  | enum     | `社招` / `校招`（最终社招 3,407 / 校招 3,412。渠道归因依据源文件 `recruit_channel_source` 字段，为剔除前分布：campus_file 3,109、source_portal 1,606、society_file 1,406、meituan_jobType 363、site 259、recruit_type_parent 234、matched_campus 1，合计 6,978，无未知值） |
| `level`            | string   | 职级（来源无可靠字段，留空）                                                                                                                                                                                                                                               |
| `salary`           | string   | 薪资（来源无字段，留空）                                                                                                                                                                                                                                                   |
| `responsibilities` | string[] | 岗位职责（description 按编号条目拆分；来源职责与要求合并在一段，约 15% 记录的拆分条目混有任职要求/加分项内容）                                                                                                                                                             |
| `requirements`     | string[] | 任职要求（来源职责与要求合并在 description，本批留空）                                                                                                                                                                                                                     |
| `skills`           | string[] | 技能标签（description+title 词表匹配，全部命中保留、不设上限；方法见 `scripts/extract_skills.py`）                                                                                                                                                                         |
| `source_url`       | string   | JD 页面 URL，可追溯（全部唯一）                                                                                                                                                                                                                                            |
| `source_platform`  | string   | `官网`（12 家有数据的公司均为官方渠道）                                                                                                                                                                                                                                    |
| `captured_at`      | string   | 采集日期（2026-11-19）                                                                                                                                                                                                                                                     |
| `completeness`     | enum     | `full`（本批全部为 JD 全文）                                                                                                                                                                                                                                               |

### sub_category：16 个标准方向

唯一定义源是 `data/skill_catalog.json`：`sub_category.rules` 按 title 关键词顺序匹配（首个命中生效，无命中落 `fallback`「通用研发」）。16 个方向全部用行业通用岗位名：大模型算法 / Agent 应用 / 推荐算法 / 推理系统 / 训练系统 / 算子与高性能计算 / Infra 工程 / 平台与调度 / 网络系统 / 存储系统 / 通用研发 / 产品与运营 / 测试/评测 / SRE / GPU/IDC 运维 / 模型运维。`scripts/import_ai_companies.py` 只引用 catalog，不定义分类常量。

### skill：109 词 taxonomy（三视角）

技能词全集 109 个，数据中出现 101 个；73 条记录（1.1%）未命中任何技能词。每个词在 `skill.taxonomy`（109 行机器可读数组，唯一定义源）中恰有一个主归属：

- **kind（主视角，三值）**：`topic`（11 个，市场主题，如 大模型/LLM、Agent框架、多模态）/ `technique`（85 个，可学习的具名技术对象，如 CUDA、vLLM、模型量化）/ `work`（13 个，工作内容，如 评测、微调 / SFT、性能优化）。三视角各自成榜、不混排比较；
- **type（technique 六类）**：language / framework / platform-tool / hardware-interface / data-system / method，回答「学习单元是什么」；
- **scope（technique 三域）**：ai-infra / agent-infra / general，回答「主要用在哪」；跨域记 `scope_facets`；
- **work_type（work 四类）**：action（岗位动作）/ role（角色）/ metric（指标）/ capability（能力保障）；
- **facets**：跨 kind 的次要视角（如 Agent框架 主归 topic、facets 记 technique/work），仅交叉查询用，不进其他视角主榜；
- **base**：行业默认词标记（当前仅 大模型/LLM 一处，人工认定），统计时置顶单列、不参与区分度排名。

判定规则与多面词表见 [docs/技能分类体系设计.md](docs/技能分类体系设计.md)。`refresh_catalog_stats.py` 从 taxonomy 回写 `items` 并断言一致性（行数、键集、字段合法），违反即失败。

### 双 matrix

`refresh_catalog_stats.py` 在 catalog 中产出两张矩阵：

| 名称               | 维度                  | 口径                                               | 用途                         |
| ------------------ | --------------------- | -------------------------------------------------- | ---------------------------- |
| `matrix_content`   | 细分 × 技能家族       | 家族各技能提及率合计，混合 kind 口径，可超 100%    | 内容提及分析，不用于技能结论 |
| `matrix_technique` | 细分 × technique_type | 提及该类型任一技能的 JD 占比（jd_coverage），≤100% | 技能结论专用                 |

## 分析口径注意

- **应用算法剔除**：仅 title 明写「算法」且分类落算法类的岗位做相关性判定——title 含模型词（大模型/LLM/多模态等）保留；title 含应用信号（推荐/搜索/调度/增长/风控等）**直接剔除（不看正文）**；两类信号皆无的看正文是否有训练/推理证据。共剔除 137 条。另按 `data/manual_exclusions.json` 人工剔除 16 条关键词误中的非 AI 岗（美团本地生活调度/运营岗 14、腾讯游戏 QA 1、阿里渠道销售 1）。非算法类岗位不受算法闸门影响（如「测试开发工程师-算法评测」按规则入测试/评测）。剔除后算法类 2,152 条。
- **算法岗口径**：算法类 2,152 条（占 32%）由 title 关键词派生——1,691 条 title 明写「算法」，461 条由 研究 / Research / RLHF / 预训练 等词进入（后者不经过应用算法闸门），其中约百条是 工程 / 系统 title（如预训练数据工程师、推荐大模型后台开发工程师），算法岗与工程岗混合。细分 大模型算法 2,001、推荐算法 151；Agent 应用 1,764 条归研发类，约六成 title 是工程岗。研发类合计 3,952 条。
- **非技术岗单列**：产品经理/运营/商业化类岗位（84 条）单列为「产品与运营」类别，不计入研发类——以非技术岗为主，仍含少量技术向岗位；title 同时含 研发/工程师 称谓词的岗（如「AIGC 研发工程师-商业化方向」）按规则落通用研发。
- **sub_category 由 title 规则派生**（规则见 catalog）：title 无命中落「通用研发」（1,006 条）；按 title 关键词派生，同方向内算法岗与工程岗可能混合。
- **技能词覆盖缺口**：技能按全文匹配（词表 109 个、数据中出现 101 个；英文词加非字母数字边界；量化一词排除 轻量化/向量化/可量化 与量化策略/量化投资等非模型量化语境），73 条（1.1%）未命中任何技能词；技能占比的分母是全部 JD。
- **字段限制**：`salary` / `level` 来源无字段，全空——薪资与职级维度不可分析；`requirements` 留空（来源职责与要求合并在 description，拆分后存于 `responsibilities`）。
- **公司集中度**：字节 2,914 条（43%）为最大来源，公司与「市场整体」的叙事需注明口径。
- **校招批公司覆盖**：校招 3,412 条来自 11 家公司（腾讯 386、蚂蚁 276 等为 11-19 补抓渠道）；阿里校招与快手校招的 `source_url` 为岗位详情链接（首批抓取曾塌缩到列表页地址，已修复）。

## 可视化

按招聘渠道分成四个页面，社招与校招面向不同读者、各自独立成站：

- [skill-map-society.html](skill-map-society.html) — 社招 · 技能地图：三视角总览（主题覆盖率 / 技术栈 / 工作内容分布）、技能谱系树（kind 徽标与性质×应用域过滤）、按岗位的学习路线、技术栈×岗位类别热力（仅 technique 词）
- [jobs-society.html](jobs-society.html) — 社招 · 岗位版图：类别分布与公司构成、城市×类别热力矩阵、JD 明细检索（类别/城市/类型/性质/应用域/技能/公司筛选，技能 chip 按 kind 标色）
- [skill-map-campus.html](skill-map-campus.html) — 校招 · 技能地图：同结构，基于 3,412 条校招岗
- [jobs-campus.html](jobs-campus.html) — 校招 · 岗位版图：同结构，基于 3,412 条校招岗

四个文件为零外部 JS/CSS 依赖的单文件 HTML（页首 logo 引用同仓库 `img/` 图片，整仓分发时正常显示）；数据更新后重跑 `scripts/build_visualization.py` 即刷新，`scripts/verify_pages.py` 做页面渲染与计数核对。其中 Playwright 动态验收（控制台错误、过滤交互）依赖 [Playwright](https://playwright.dev/python/)：`pip install playwright && playwright install chromium`；未安装时脚本自动跳过动态部分，保留单文件检查与三视角计数核对（仍以退出码 0 结束并打印 SKIP 提示）。

## 脚本（本地工具资产）

- `scripts/import_ai_companies.py` — 来源文件 → 统一 schema 转换 + 应用算法剔除 + 人工剔除（幂等；分类规则引用 `data/skill_catalog.json`）
- `scripts/extract_skills.py` — 技能词提取（词表与语境排除规则引用 catalog；`--eval` 做提取一致性评估）
- `scripts/refresh_catalog_stats.py` — 从数据重算 catalog 统计字段（技能频次、taxonomy 回写与校验断言、家族计数、双 matrix、方向计数），import 之后运行
- `scripts/analyze.py` — 分类统计 + 技能树三视角报告（输出 `analysis/classification.md` 与 `analysis/skill-tree.md` 全库、`skill-tree-society.md` / `skill-tree-campus.md` 分渠道三份）
- `scripts/build_visualization.py` — 生成四个可视化 HTML（kind/type/scope 归属读 catalog taxonomy，不在 build 内硬编码）
- `scripts/verify_pages.py` — 页面验收：四页 Playwright 控制台检查 + 三视角计数与 skill-tree md 核对 + 单文件检查
