# Agent 的每一步，都需要大模型吗？——七张图看懂刚爆火的决策模型 Jev

> 2026-09-21 | 图解版：七张手绘图讲清 TypeSafe AI 的决策模型 Jev（System One Models）。数字口径分三档标注——厂商口径、Browser Use 社区实测、腾讯技术工程 demo 转引；完整核对来源见文末。
>
> 严谨说明：「一个字」指一个 token；延迟与价格均为厂商口径，实测数字单独标注。

最近一个叫 Jev 的模型在社区里刷屏：有人拿它玩 DOOM，有人拿它通关 Mario、刷网页。它不写文章、不聊天，只干一件事——做判断。腾讯技术工程的工程师拿它做了个真 demo，然后问了一个好问题：我们现在搭 Agent，让大模型反复决定下一步——这里面有多少调用，真的需要大模型吗？

![图1：一次 7 秒的浏览器任务，AI 在忙什么](assets/jev-01-hook.png)

**图 1**｜Browser Use 的实测把账摆了出来：一次 7 秒的浏览器查票任务，AI 被调用了 19 次——17 次是「判断」（点哪个、选哪个），只有 2 次是真正的「写文字」（填城市名）。Agent 的绝大多数调用，其实都是在做判断。

![图2：官方给 Jev 的定位：System One Model](assets/jev-02-systems.png)

**图 2**｜判断和写文字是两种活。官方给 Jev 的定位就叫 System One Model，灵感来自卡尼曼《思考，快与慢》：System 1 快而省，管反射式的判断；System 2 慢而深，管推理和生成。Jev 一次判断 70–500 毫秒；官方对标的前沿大模型端到端区间是 3–329 秒（均为厂商口径）。

![图3：三种接口，答案永远在你给的标签里](assets/jev-03-api.png)

**图 3**｜快是有代价的——它只会做选择题。三种接口：Choice 从你给的选项里选一个；Score 按你定的等级打分；Noul 判断一句话真假。返回选项、概率和置信度，答案永远在预设标签里，没有自由发挥——所以也没有幻觉。（选项基数上限 255。）

![图4：Jev 是拿手柄的玩家，代码是游戏本身](assets/jev-04-roles.png)

**图 4**｜所以它不替代大模型，是分工。demo 作者的原话：「Jev 是拿手柄的玩家，普通代码是游戏本身」——状态整理好交给 Jev，它选动作，代码执行；地图、伤害、碰撞、渲染全归游戏引擎，只有要填「城市名」这种文字时，才叫一次生成模型。社区实测里，Jev 单次判断的中位延迟只有 178 毫秒。

![图5：同一个选A，三个数三件事](assets/jev-05-confidence.png)

**图 5**｜但有一个概念最容易用错：置信度不是准确率。Jev 返回的 0.85，量的是「答案分布有多集中」——官方公式是 (3×最大概率−1)/2；它不是「这个答案有多大概率正确」。想拿 0.9 当自动执行的门槛？官方原话：阈值没有通用值，用你自己的数据标定。

![图6：confidence 怎么用：三档门禁](assets/jev-06-gating.png)

**图 6**｜正确用法是三档门禁：高置信直接自动执行；中置信升级给强模型再想一遍；低置信转人工或追问。阈值画在你自己的数据上，随风险调整——不可逆操作的门槛更高。你的代码，编码了你的风险容忍度。

![图7：判断类任务的账](assets/jev-07-cost.png)

**图 7**｜最后算账。输入 $0.042/百万 token，输出免费——判断类任务的账只有输入这一项，十亿 token 才 $42。DOOM 那个每秒 10 次判断的玩法，厂商口径约 $7/小时。省下的不是算力，是每一次判断的延迟和钱。

判断与生成的分工，可能是 Agent 架构的下一层。Jev 能不能成为这个方向的主要选择还不确定——大模型厂商的结构化输出在推进，传统分类器在另一侧挤压，它的位置在两者之间：任务需要语义理解、标签变化频繁、又不值得为每个判断单独训模型。

## 来源与核对

- 官方博客：[Introducing System One Models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)——定位、RLCD、并行采样、价格与延迟、DOOM 口径、限制声明
- 官方文档：[docs.typesafe.ai](https://docs.typesafe.ai/introduction)——三接口定义、[confidence 的计算与标定](https://docs.typesafe.ai/confidence)、[智能家居 demo](https://docs.typesafe.ai/demos/smart-home)
- Browser Use 实测记录：[jev-ultrafast performance.md](https://github.com/browser-use/jev-ultrafast/blob/main/docs/performance.md)——17 次 Jev 请求、178 ms 中位延迟、7.073 秒任务
- 体验向原作（本文多篇引述）：[腾讯技术工程《聊聊最近爆火的 Jev 模型到底是个啥》](https://mp.weixin.qq.com/s/w4kDVSF_beVA7CVhL_Xohw)
- 配图由 rough.js + 霞鹜文楷程序化生成，源文件见本目录 `assets/`
