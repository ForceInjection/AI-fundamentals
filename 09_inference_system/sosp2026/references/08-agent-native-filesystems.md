# 08 · Agent 原生文件系统：别让 AI Agent 随意动你的文件

> SOSP 2026 论文精读笔记。来源：`../papers/08_agent-native-filesystems.pdf`（arXiv:2604.13536v5，2026-11-14 下载；SOSP '26, Prague）。作者来自 UW-Madison（Arpaci-Dusseau 团队）、Microsoft Research、Iowa State；系统 YoloFS 开源于 github.com/YoloFS/YoloFS。文中数字均为论文口径。

## 一句话

对 290 份真实事故报告做首个系统性研究，指出当前 Agent 文件访问的「信息缺口」（谁都看不清一条命令的文件系统效果）与「控制缺口」（事后拦不住、恢复不了），提出把信息与控制从 Agent 侧移进文件系统本身，并落地为 Linux 内核文件系统 YoloFS。

## 问题与背景

编码 Agent（Claude Code、Codex、Cursor 一类）以用户权限直接操作本地文件系统，harness 现有的三道护栏都有结构性盲区（§2，Table 1）：built-in 工具策略管不到 shell 命令；shell 过滤器匹配的是命令字符串而非文件系统效果（`rm` 与 `rm -f`、Python `os.remove`、`;` 与 `&&` 链、子 shell 都能换形态绕过）；沙箱要么太紧挡住合法工作（测试套件、GPU 访问），要么太松挡不住沙箱半径内的破坏。

研究方法（§3.1）：收集 2024 年 2 月至 2026 年 3 月的 290 份公开报告（GitHub issues 205、社媒 31、产品论坛 25、博客 18、NVD 11），覆盖 13 个 harness（Claude Code 97 份最多）。人工标注 100 份定标签体系，再用 agent 标注全部并交叉验证。

影响的量级（§3.2）：写操作占 44%（最常见），包括用占位符覆写源码、把文件截断为零字节、用生成数据替换真实内容（39% 的写事故如此）；另有整盘擦除、home 目录删除。泄露占 17%，API key、.env、SSH 凭据都在名单上。42% 的伤害落在项目之外（系统文件 15%、用户文件 14%、机密 13%）。恢复侧：40% 不可恢复（其中 23% 完全丢失），31% 可以 `git checkout` 简单恢复。察觉侧：83% 用户立即发现，10% 延后发现（代码稍后才报错），8% 完全无感（静默凭据外传）。

六个 Finding（§3.6）归成两个缺口：信息缺口（F1：用户与 Agent 都认不出命令的真实效果）与控制缺口（F2：现有机制不能可靠阻止或恢复）。原因拆到三个角色上——模型是不可靠执行者（168/290：错误目标、错误范围、忽略指令、上下文压缩后丢指令、提示注入）；harness 是有限的守卫（218/290 涉及：shell 漏洞 77、效果盲过滤器 52、沙箱错配 41、策略僵硬 130）；用户是不堪重负的审批者（105/290：80 份 YOLO 全放行，31 份审批提示根本看不出危险，Figure 5 里一条 352 字符的 sed 命令用户无从判断）。

## 核心设计

论文的主张（§4）：把信息与控制下沉到文件系统，提供三个原语，并给出四个约束（完整性：拦截包括项目外在内的一切访问；适应性：策略可在执行中调整；兼容性：不依赖 reflink/快照等非 POSIX 特性；性能：关键路径低开销）。先行方案各缺一角（§4.3）：VCS 只管仓库内、`.gitignore` 常忽略敏感文件、历史存在 agent 可删除的 `.git`；ZFS/Btrfs 粒度错配（快照不按 session 分组、整 FS 回滚丢无关改动、自有磁盘格式）；OverlayFS 的 rename 要复制整文件进上层、单可写层没有命令边界；Docker/Landlock 类策略改一条要重启进程。

**YoloFS**（§5）是 Linux 栈式文件系统（2.7K 行 C）加用户态 CLI（8K 行 Rust），三个机制对应三个原语：

1. **Staging（§5.2）**：所有变更停在 staged 状态，用户 review 后才 commit 或 abort。关键工程挑战是 union 文件系统的 rename 放大问题，解法是把内容与路径解耦成三个结构：flat file store（变更内容按单调递增的 inode 号平铺存放）、override tree（路径到 backing 的内存映射）、session journal（append-only 日志，只记 S(stage)/D(delete)/R(rename) 三类动作，Figure 8-9）。正常文件操作只走 override tree，不被日志拖慢；review 时 CLI 在用户态回放日志重建净变更视图。
2. **Snapshots and travel（§5.3）**：默认每条命令后打 P 标记分段，journal 记录配 generation 号，文件写新版本时 copy-on-write 到平铺层而非原地覆盖。travel 靠 T 标记回放「存活段」重建任意历史状态，且不截断日志：回退后发现回错了还能「回到未来」（Figure 10）。preimage 字段让每个段独立可回放，不必从头重放整个日志。
3. **Progressive permission（§5.4）**：不做前置完整策略，路径配五档规则（ask/write-ask/read-only/deny/allow），沿层级 rule tree 继承。Agent 访问触发时 kernel 阻塞请求线程、CLI 弹问用户、决定写回 kernel 并唤醒线程（超时拒绝）；「拒绝且不再问」这类决策持久化到 per-project 配置，下次直接生效。被拦访问以 G/C note 记入日志，review 时与文件变更并排呈现，让 Agent 与用户都能看到「这条命令试图访问什么、被什么规则挡了」。

安全隔离（§5.5）：agent 命令在以 YoloFS 挂载点为根的环境里运行（pivot_root），内核按调用进程的根文件系统识别 agent 侧，拒绝其执行规则修改、权限决定与 commit。崩溃恢复靠 journal 与 staged 数据都在 base 文件系统上：重挂载后回放日志即可重建状态。集成方式是 harness 钩子：Claude Code 用 PreToolUse、Copilot 用 preToolUse、Gemini 用 BeforeTool，每条 agent 命令经 `yolo` 包裹执行。

## 效果与代价

评估（§6-7）分三块（agent 用 Claude Code 2.1.45/claude-sonnet-4-6、Codex 0.10.1/gpt-5.3-codex、Copilot CLI、Gemini CLI/gemini-3-flash-preview）：

- **自纠正（§6.2）**：11 个隐藏破坏性副作用的不透明任务（10-41 行代码的小项目，不透明度分四级，最难的藏在预编译二进制里）。无 YoloFS 时没有基线 harness 可靠防住：Claude 防住 4/11（其中 optimizer 任务靠先反编译才判断出危险）、Codex 全靠撞沙箱后问用户、Copilot 基本全漏。YoloFS + Claude Code 11/11 保住了 base 文件系统：8 起 Agent 自己检视效果、travel 回滚（formatter 任务里 agent 看 diff 后断言「CRITICAL: 这是破坏性变更」并撤销）；3 起效果看起来像合理目标（部署任务删 `src/` 是部署语义的一部分），Agent 保持 staged 可见，由用户拒绝。
- **例行任务交互（§6.3）**：112 个单操作任务（18 种操作 × 5-7 种路径）。任务成功率 YoloFS 99%（Claude 98%、Codex 100%、Copilot 96%、Gemini 75%），每任务用户交互次数 YoloFS 平均 0.4（Claude 0.9、Copilot 1.3、Gemini 2.2），与 Codex 的 0.4 持平但 Codex 依赖可写沙箱加回退设计。项目外路径是交互主要来源（平均 0.9）。
- **性能（§7）**：CloudLab EPYC + Ext4 对比 OverlayFS 与 BranchFS。元数据操作大多持平或更快（staged 命中时 readdir/rename/unlink 反而快于 Ext4）；顺序写相对 base 约 -2%；快照扩展性上 OverlayFS 约 50 个后挂掉（mount 选项长度超内核限制），YoloFS 不受影响；真实内核补丁工作流里 YoloFS 贴平 Ext4，仅 commit 十万级文件多 3.5 秒，OverlayFS 则慢 18% 且首次构建后要重建挂载。

代价与边界：评估任务刻意做小（10-41 行项目），作者明说是为了让 Agent 有机会检视代码发现异常，真实大型仓库的检出效果未测；三原语只覆盖文件系统这一种持久状态，Agent 的 API 写入、数据库操作、云服务调用都门外（§8 承认）；YoloFS 目前是 Linux 专属内核模块（支持 Ubuntu 24.04/26.04 默认内核），macOS 与远程开发场景无覆盖。

## 总评

这是很正统的 Arpaci-Dusseau 式论文：先用手工业者式的耐心把 290 份事故报告拆成可复用的标签体系与因果分类，再让系统设计严格对应研究发现的缺口，六个 Finding 与三个原语一一挂钩。290 份报告这个样本量在系统会议里算扎实，其中最有传播力的一组数字是恢复与察觉的分布：40% 不可恢复、8% 用户全程无感。这两个数字比任何 argument 都更能说明「事后审批 + git 兜底」这套默认实践的覆盖面缺口。

YoloFS 的设计判断也有可圈点处：把控制点从「命令字符串」挪到「路径访问」，一步绕开了过滤器军备竞赛（F4）；把审批从「前置完整策略」改成「执行中按访问逐步细化」，直接回应 approval fatigue（F6/U1）。staging 默认全量暂存这个决定相当激进，等于把文件系统改成了默认事务语义，性能评估支撑了这个决定在常规负载下可行，但十亿级文件或网络文件系统上的表现是未测领域。

对本仓库读者，这篇与 11_claude-code-sandbox（Bubblewrap 进程沙箱）、agent_infra 的沙箱系列（DSec、Substrate 的 VM 边界）构成一条完整的谱系：进程/VM 沙箱解决「代码能不能跑」，YoloFS 解决「跑了的代码动没动不该动的东西」，两者互补而非替代。Table 1 对六大 harness 护栏的横评（Claude Code 的 Bubblewrap opt-in、Copilot 的 122 规则 5.6K 行正则过滤器等）本身就是很好的现状快照。YoloFS 式的「路径级渐进权限 + 会话级 staging」对任何自研 Agent 执行环境都是一份值得对照的设计清单，即使不采用内核方案，journal/staging 的思路也可以在用户态 fuse 或虚拟文件层（参考本仓库 14 号虚拟文件系统一篇）近似实现。

## 疑点与待核

- 290 份报告来自公开渠道（GitHub/社媒/论坛），幸存者偏差方向值得想：能写出报告的多半是「发现了」的用户，8% 无感比例的估计基础较弱
- §6.2 的 11 任务每个跑几次、随机性如何处理，正文未见说明（三次取均值仅出现在 §6.1 方法描述，且针对的是 112 例行任务）
- symlink 场景交互次数偏高（1.1-5.7 次/任务），论文未深入分析原因
- 「kernel 按 calling process rootfs 识别 agent」在容器嵌套或共享 rootfs 的部署下是否可靠，论文未讨论
