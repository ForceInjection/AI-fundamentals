# Skill 生态扫描：2026 年论文与项目盘点

> 探查日期：2026-11-18｜两个方向并行检索（学术论文 + 开源项目），作为本目录后续文章的素材库。
> 口径说明：论文均为 arXiv 预印本（无正式 venue）；恶意/脆弱 skill 数量（341/900+/1467 等）均为安全厂商研究口径；GitHub 星标为 2026-11-18 经 GitHub API 逐一核实。该领域月产数十篇新论文（2026-10 至 11 两周 arXiv "agent skills" 命中已 476 篇），引用数字需带复核日期。
> 已知基线（另见 [01-agent-skills-in-the-wild.md](01-agent-skills-in-the-wild.md) 与[正文文章](../04_skillspector-security-scanner.md)）：《Agent Skills in the Wild》（arXiv:2601.10338）及其姊妹篇《Do Not Mention This to the User》（USENIX Security 2026 = arXiv:2602.06547）、NVIDIA SkillSpector。注意：搜索引擎常把这两篇已知论文的标题互相混淆，二手摘要有张冠李戴，以 arXiv 页面为准。

## 一、论文（按价值排序，8 条均已核到 arXiv 原文）

### 安全线

1. **Scan Before You Install: Detecting and Mitigating Local and Composite Vulnerabilities in Skills** — arXiv:2610.37901（2026-10-27），Ben-Gurion University（Gandhi、Aviv、Wudali、Elovici、Shabtai）。六模块 skill 扫描器，主打「良性 skill 也有漏洞」与「多 skill 交互漏洞」两个现有扫描器的盲区；直接以 SkillSpector、Cisco Skill Scanner、SkillSieve、AgentSkillsScanner 为 baseline，MalSkillBench（1,000 skill）F1 0.947，组合漏洞 SCR-Bench F1 0.991。SkillSpector 文章的天然续篇素材。
2. **When Skills Tell Agents Not to Ask: A Large-Scale Study of Oversight Directives in Agent Skills** — arXiv:2610.37734（2026-10-27），Queen's University（Sanko、Tian）。150 万 skill 语料：约 1/17（约 8.8 万）含「别问用户、跳过检查、--no-verify」类削弱人工监督的放权指令，70% 无条件生效；跑 shell/装包/sudo 的 skill 命中率高 1.7–1.8 倍；开源扫描器最多抓出 46% 且精度 <0.4。核心洞察：最危险的可能不带恶意，只是让你别问。
3. **Detecting Insecure Compositions of Clean Agent Skills via Capability-Flow Graph Analysis**（系统名 KnotScryer）— arXiv:2610.32155（2026-10-23），NTU + 中关村实验室 + 清华（通讯 Ke Xu）。单个干净、组合有害：把 skill 抽象成能力流图，沿跨 skill 数据交接边搜「受污染数据流入危险操作」路径；13,598 个 ClawHub 单扫全过的干净 skill 中找出 612 条不安全组合，13.7% 的链上所有 skill 单独看都安全。与第 1 条攻防互证，共同论证单 skill 扫描器的结构性上限。
4. **Under the Hood of SKILL.md: Semantic Supply-chain Attacks on AI Agent Skill Registry** — arXiv:2605.11418（2026-05-12，31 页），University of Maryland（Saha、Faghih、Feizi）。纯文本语义供应链攻击（无代码 payload）：检索投毒 Top-10 命中 80%、配对实验 77.6% 选恶意变体、36.5%–100% 逃过治理审查。补上「扫代码」类扫描器覆盖不到的语义层。
5. **SkillLifeBench: Benchmarking Security Across the Lifecycle of Agent Skills** — arXiv:2610.35215（2026-10-26），Imperial College London + 北京大学（通讯 Wentao Zhang）。388 组配对场景：运行时任务并列第一的两个模型（82.0%），生命周期任务上 96.7% vs 56.7% 分化；103 次运行「任务成功 + 攻击目标也达成」双双命中。「只测安装时」不够的时间轴证据。

### 工程与评测线

1. **How Agent Skills Evolve: A Large-Scale Empirical Study of Skills on GitHub** — arXiv:2611.04456（2026-11-03），HKUST + 浙江大学（通讯 Zhongxin Liu）。8.42 万 skill、75 万 commit：组件/变更模式/动因三套人工 taxonomy；五条含义中最关键的一条是「安全动机的改动实际降险有限」，冗余与可靠性问题被低估。
2. **SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks** — arXiv:2602.12670（2026-02-13，v4 至 2026-06-14），77 位作者社区共建（BenchFlow、OSU、Amazon、UC Berkeley、UC Santa Cruz，含 Dawn Song、Terry Yue Zhuo）。87 任务 × 8 领域配对评测：skill 平均把 pass rate 从 33.9% 提到 50.5%（+16.6pp）；反直觉：3 模块以内的紧凑 skill 胜过大而全，小模型挂 skill 可匹敌大模型。
3. **A First Measurement Study on Authentication Security in Real-World Remote MCP Servers** — arXiv:2605.22333（2026-05-21），复旦大学 + 中南大学。7,973 个在线 remote MCP 服务器实测：40.55% 无鉴权直接暴露；119 个可测 OAuth 服务器全部至少 1 个缺陷（共 325 个），动态客户端注册缺陷影响 96.6%；9 个 CVE。归纳 4 类 9 种缺陷可复用为清单。

### 备选池（已核实存在与主题，仅核到摘要页）

- **Skill-Inject**（arXiv:2602.20156，2026-02）：202 组注入任务对，前沿模型最高 80% 攻击成功率；结论「扩模型/过滤输入救不了」。
- **SkillJect**（arXiv:2602.14211）：自动化 skill 投毒框架，payload 藏辅助脚本 + SKILL.md 前置诱导。
- **BadSkill**（arXiv:2604.09378）：「skill 内嵌模型」载体的后门攻击，最高 99.5% ASR；4 页短文。
- **CoSA**（arXiv:2610.35082）：skill 间冲突攻击，加 5% 恶意 skill 使违规率 35%→58%。
- **The Devil Wears Markdown**（arXiv:2611.10830，2026-11-07）：静态 + 动态 + agentic 三层混合检测恶意指令。
- **SkillBuffBench**（arXiv:2611.22017，2026-11-17）：跨 harness、模型、访问模式、记忆的贡献度基准。
- **SkillSieve**（arXiv:2604.06550）：分层分诊（廉价静态 + 定向 LLM + 多模型投票）。此条仅经 alphaXiv 与引用交叉确认，引用前需再核原页面。
- **Synapse**（arXiv:2611.20856）：把 skill 优化当「生成-验证」程序修复问题，工程方向候选。

### 标准动向（非论文）

2026-10-22，Anthropic 把 Agent Skills 开放标准（名称、logo、GitHub 组织、agentskills.io 域名）捐给 Linux Foundation 旗下 Agentic AI Foundation（AAIF，Anthropic 与 Block、OpenAI 共同发起），同日宣布 Agent Skills、Open Responses、OpenEnv、OpenSandbox 四个项目。格式未变，现有 skill 兼容。一手来源：Anthropic 官方新闻页 + Linux Foundation 新闻稿。

## 二、项目（10 条，星标经 GitHub API 2026-11-18 核实）

1. **OpenClaw 运行时 + ClawHub 注册表的恶意 skill 危机** — github.com/openclaw/openclaw（39.5 万★）｜github.com/openclaw/clawhub（9.6k★）。2026 上半年 skill 供应链安全的标志性事件：Koi Security 2026-02 审计 ClawHub 2,857 个 skill 报出 341 个恶意（厂商口径），投放 AMOS/Atomic Stealer 与键盘记录器，后续分析称总数超 900；Unit 42 证实存在绕过自动扫描的规避型恶意 skill（`security-check`/`nanopdf` 等被点名）。OpenClaw 官方文档已提供 skill 门控、allowlist、环境注入配置。
2. **anthropics/sandbox-runtime**（7.2k★，Apache-2.0，2025-10 创建，活跃）— Anthropic 官方轻量 OS 层沙箱：无容器强制文件系统与网络限制。「把 skill 当不可信代码」路线的运行时基座。
3. **tech-leads-club/agent-skills**（7.2k★，2026-01 创建）— 面向专业 coding agent 的 skill 注册表，内建安全扫描与 `security-scan-allowlist.yaml`（标记申请越权沙箱权限的 skill）；宣称市场上 13% 的 skill 存在问题（厂商口径）。
4. **iflytek/skillhub**（5.4k★，科大讯飞，2026-03 创建）— 自托管企业级 skill 注册表：namespace 发布、版本化、RBAC、审计。组织内部 skill 治理的开源参照实现。
5. **obra/superpowers**（31.9 万★，Jesse Vincent）— 最早的社区级 agentic skills 框架与开发方法论（brainstorm→计划→失败测试→编码→自审），配套 obra/superpowers-skills。方法论库形态的鼻祖。
6. **mattpocock/skills**（32.7 万★，当前全 GitHub skill 类最高星）— 「Skills for Real Engineers」，作者直接从自己 .agents 目录发布。个人 skill 集合半年冲 30 万星，是 skill 分发格式规模化的证据。与第 5 条构成「方法论库 vs 个人实践库」两种生态形态。
7. **cloudflare/security-audit-skill**（7.5 万★）— 多云 agent 的多阶段安全审计 skill，产出机器可读 findings。大厂把「安全审计」本身做成 skill 的样板。
8. **OWASP Agentic Skills Top 10**（github.com/owasp/www-project-agentic-skills-top-10，431★）— skill 威胁分类学基准；其 Scanner Integration 页（2026-03）已把 SkillSpector 列为参考集成扫描器，可作扫描器对比的共同坐标系。
9. **rohitg00/skillkit**（1.6k★，2026-01 创建，近半年放缓）— 跨 agent skill 兼容层：install/translate/share，同一 skill 翻译到 Claude Code、Cursor、Codex、Copilot。「一次编写、多端运行」的样本；同名小工具洗牌严重。
10. **Tego「Agent Skills 开放安全数据库」**（index.tego.security/skills/，非开源）— 首个专做 skill 安全评估的数据库：自动扫描 + AI 模型 + 人工复审，对注入、提权、外泄打分。HN 讨论（item 47402118）中「npm 类比 vs agent 不像人一样权衡风险」的争论可引用。同周期可搭配 Snyk ToxicSkills 报告（2026-02，36% skill 含安全缺陷、约 1,467 个脆弱 skill，厂商口径）与 Mitiga「静默外泄整个代码库」研究（2026-10）。

**落选备用**：addyosmani/agent-skills（11.9 万★）、K-Dense-AI/scientific-agent-skills（5.8 万★）、Karanjot786/agent-skills-cli（194★，45 个 agent 统一管理）、SkillGuard 权限框架论文（2026-06，未见活跃开源实现）。

## 三、运行时演进备忘

- GitHub CLI 新增 `gh skill` 命令（官方 changelog 2026-04-16），GitHub 成为 skill 一方分发渠道。
- Claude Code 2026-11 安全版：跳过问题插件、`claude plugin validate` 上报、沙箱网络权限提示；2026-05（2.1.157）起 `.claude/skills` 内插件自动加载。社区 PSA：旧版本插件缓存不清理导致 skill 双载。
- Gemini CLI 约 2026-04 起原生支持 SKILL.md（`.gemini/skills/`）；Codex CLI、Cursor、Antigravity 均已支持。SKILL.md 是跨 agent 事实标准。
- HKUDS/nanobot issue #1448（2026-03）：框架层把「skill 执行安全沙箱」列入路线图的样本。

## 四、交叉观察（续篇的三条候选主线）

1. **组合是下一个战场**：论文 1（BGU 扫描器）与 3（KnotScryer）攻防互证，数据都来自 ClawHub——正好接上项目 1 的恶意 skill 事件。SkillSpector 文章的天然续篇。
2. **最危险的指令不带恶意**：论文 2（放权指令）与 4（纯文本语义攻击）指向「扫代码类扫描器覆盖不到语义层」的结构性上限；SkillSpector 文章可补一节「边界之后」。
3. **标准已成，治理刚起步**：标准进 Linux Foundation（2026-10）+ SKILL.md 跨 agent 事实标准 + OWASP 分类学就位，但市场治理（ClawHub 事件、iflytek/skillhub 的 RBAC、tech-leads-club 的 allowlist）还在追。工程治理篇的主线。

## 探查方法备忘

论文方向：arXiv 摘要页 + HTML 全文逐一核实（作者机构取自页眉），已排除两篇已知论文；备选池仅核摘要。项目方向：GitHub API 核实星标与活跃日期，厂商口径数据均标注。两个方向均以「宁可少而准」为筛选取向，读不到原文的条目已明确标注。
