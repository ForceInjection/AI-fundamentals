#  分享：Token工厂财务运营测算工具（附下载和学习笔记拆解）

原创  两年砍柴  两年砍柴  两年砍柴

_2026年08月17日 07:44_ __ _ _ _ _ _ 浙江  _

在小说阅读器读本章

去阅读

![image](https://mmbiz.qpic.cn/mmbiz_png/IPq0428B56hCCG5YkqBY8iaia4NeBSAuDEwjUKR0mSda5lSYUWZQKb8wCUicuzK77jYQKia1gAQicmz4IN4ia3j9b39LMf5icZL7U1KuBuIddWqic8o/640?wx_fmt=png&from=appmsg)
本文总共  3948  字，读完总共约需  10  分钟  作者 | 编辑 | 审校：两年砍柴

** 大家好，我是两年砍柴。  **

** 这是一份学习笔记（不仅无偿分享这个测算工具，还分享一下我的学习逻辑），这不是商业评估的工具，不带有有何商业性质哈。  
**

** 最近我在研究  AI  基础设施（模型聚合平台、  AI  中转站和  Token
工厂）的投资逻辑，想搞清楚一个问题：如果我想建一个Token工厂，到底能不能赚钱？我发现市面上关于  “Token  工厂  ”
商业模型相关资料少得可怜（几乎没有）。  **

** 求人不如求己，索性自己动手，基于我查询总结的各种相关资料，用  Excel  搭了一套完整的财务测算工具  ——  从硬件选型到  Unit Econ
，从折旧摊销到  ROI  ，从敏感性分析到运营仪表盘，反正我是把能想到的维度都塞考虑进去了（  文末附有工具的下载方法  ）。  **

![](https://mmbiz.qpic.cn/mmbiz_png/IPq0428B56joVEKk1GYfEbxHXB86hnqTcVlN3CEhfQYHbEqB5Tq7MHSfb1wiaZ8xMY9xSicpyiaAzQDV2zwfwRycwqnSVo9sGk6eER5kkhkElQ/640?wx_fmt=png&from=appmsg)

** 这个工具用的也是典型的商业模型分析方法，内容其实很简陋（甚至不能排除有错误），抛砖引玉而已。  **

今天，我就把这套工具的逻辑完整拆一遍，既是记录和复盘一下自己的思考过程，也希望能帮到同样在研究这个方向的朋友。

我还有个公众号的技术小号“  砍柴者说  ”，请大家点个关注哈，后续我会在小号上做技术分享。

⚠️  ** 郑重声明：
这套模型纯粹是个人业余的学习工具，里面的参数假设、测算结果均不构成任何投资建议或生产评估依据。如果你要用它来做商业决策，后果自负哈。  **

**
![](https://mmbiz.qpic.cn/mmbiz_png/IPq0428B56h8YlFT0WGFaOlCpheKunsopwbm0vDPeA1rUhbODLvl12iaO8yNJOFgvzsTB56GgUU2gVh4fib0XvXhNyqiaEqlvNmD5jx6RF14xY/640?wx_fmt=png&from=appmsg)
为什么我要做这个工具？  **

坦白说，我不是财务出身，也没有真的要去建Token工厂。纯粹是想通过建模的方式，逼自己去理解，比如：

  * 一张GPU卡一年能产出多少token？ 

  * 电费到底占成本多大比例？ 

  * 什么样的利用率和定价才能回本？ 

  * 如果市场降价了，还能扛得住吗？ 

我查阅了大量的信息，市面上关于  Token  工厂的讨论大多停留在  “  算力很贵  ”“  推理成本高  ”
这种定性层面，很少有人把账算透（想必算透的人也不会分享出测算的方法，哈哈）。

作为这个行业的从业者，尤其是产品经理，这些问题如果不自己亲手完整算一遍，就很难形成体感。

所以我就按照  “  先假设  →  算投入  →  算成本  →  算收益  →  做敏感分析  ”
的这个思路，搭了这个模型。毕竟我也不是财务行业的科班出身，工具肯定还不够系统化，甚至都有可能不太准确，能力有限，希望大家海涵。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/IPq0428B56gIicqSatwtQQ78jibbKbAG2vEibSu1tYJ6FOENibGKX7cJfot0GpTqn4ozmC0j3lxRj5KIIGUcF1Micn723esVy8PleAewouVnLHrA/640?wx_fmt=png&from=appmsg)

** 测算模型工具有什么（8张sheet）？  **

整个工作簿分了  8  个  Sheet  ，逻辑链条是这样的：

** 1️  ** ** ⃣  ** ** 硬件投资明细：  钱花在哪了  **

建一个  Token  工厂，第一笔钱是买硬件。我把所有一次性资本支出拆成了  10  项。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/IPq0428B56hiaPyD1XmbbMvED8udPWyEbQcL7zmLl7l5T5eib9kytC2iaOn29dlEfrx7Zjgnxn8PHKHgia3qXsMd81iaSEQ4o9hbXhib4Wx8gKdNQ/640?wx_fmt=png&from=appmsg)

这里有个细节：机房租金的年度运营成本和机房建设的一次性投入不要重复计算，工具里已经做了区分。

** 2️  ** ** ⃣  ** ** 参数假设：  唯一需要你动手的地方  **

所有黄色单元格都可以修改，改完整个表格自动联动。

![](https://mmbiz.qpic.cn/mmbiz_png/IPq0428B56iaYcBGxxMzr7YXgm2ibGK3Bat40QL5f4POicvMmn1LXan9oDDMp8hYYjQxcAFVppF9icQaZ7KCLbDoq74Pg0D9IUMwSfg6zfMXYcc/640?wx_fmt=png&from=appmsg)

核心参数  包括：

·  ** 硬件参数  ** ：  GPU  数量、单卡价格、功耗、推理吞吐量

·  ** 运营参数  ** ：  利用率、电价、  token  售价、售价年降幅

·  ** 产能爬坡  ** ：  可以设置第一年只达到  60%  利用率，第三年才满产

·  ** 融资选项  ** ：  可以模拟贷款，默认是全自筹

·  ** 运营成本  ** ：  人工、租金、带宽、维护费等

举个例子，  默认参数  下：

·  64  张  GPU  ，单卡  2000 tok/s

·  利用率  70%  ，电价  0.75  元  /kWh

·  token  售价  5  元  /  百万  tokens

·  折旧  4  年，残值率  10%

** 3️  ** ** ⃣  ** ** 财务测算：  核心算式一览  **

这一页是所有公式的集中体现，也是我最花心思的地方。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/IPq0428B56gX0NJGZ7lFLYvPP6vl4lYKJia1zfM2icbiaaiamoVSeIOQ3tFnricRzZAFSR994BU3ExXHuc67IAbDcYiagxGohqyKprF4mzfvCsz4o/640?wx_fmt=png&from=appmsg)

** 年产出怎么算？算法是：单卡吞吐  ** × 3600  秒  × (8760  小时  ×  利用率  ) ×  卡数。

默认参数下：  2000 × 3600 × (8760 × 70%) × 64 =  ** 28.26  ** ** 亿  tokens/  年  **

> ⚠️ 校正注（评审复算）：此式为**单位换算笔误**。2000 × 3600 × 6132 × 64 = 2.8256 × 10¹² = **2825.6 亿（2.83 万亿）tokens/年**，而非 28.26 亿（少 1000 倍）。作者下文"282.56 万百万 tokens"与年收入 1412.8 万元均自洽，可反证 28.26 亿为笔误。

换算成计价单位：  ** 282.56  万百万  tokens。预计的年销售收入：  ** 282.56  万  × 5  元  =  ** 1412.8
** ** 万元  **

** 电力成本怎么算？算法是：(GPU  ** 裸卡功耗  \+  服务器系统开销  ) × PUE × 8760 ×  利用率  ×  卡数  ×
电价。

默认参数下：  (1 + 0.2) × 1.4 × 8760 × 70% × 64 × 0.75 =  ** 49.45  ** ** 万元  /  年
**

你没看错，  64  张卡的集群一年电费不到  50  万，相比近  2000  万的硬件投资，电费其实不是大头。

** 真正的大头是折旧：1918.4  ** 万  × (1 - 10%) ÷ 4  年  = 431.64  万元  /
年。加上人工、租金、带宽、维护等其他固定成本，  ** 年固定成本合计  787.56  万元  ** 。

** 最终结果：  **

·  年净利润：  ** 431.85  万元  **

·  年经营净现金流：  ** 863.49  万元  ** （净利润  \+  折旧）

·  静态回收期：  ** 2.22  年  **

·  ROI  （现金流口径）：  ** 45%  **

·  盈亏平衡利用率：  ** 40.4%  **

也就是说，只要利用率超过  40%  ，这个项目可能就不亏钱。

** 4️  ** ** ⃣  ** ** 仪表盘：  一页看清全局  **

所有核心  KPI  汇总在一页：

·  收入、净利润、现金流

·  初始投资、残值

·  回收期、  ROI

·  盈亏平衡利用率

·  成本饼图、累计现金流曲线

改完参数，直接看这页就行。

** 5️  ** ** ⃣  ** ** 单卡单位经济  ——  从微观视角看  **

每张  GPU  单独算账：

·  年收入：  22.08  万元

·  年总成本：  13.08  万元

·  年净收益：  ** 9  万元  **

·  每百万  token  成本：  ** 2.96  元  **

·  每百万  token  毛利：  ** 2.04  元  **

如果售价降到  3  元以下，单卡就开始亏钱了。

** 6️  ** ** ⃣  ** ** 5  ** ** 年现金流：  看什么时候真正回本  **

考虑了产能爬坡和期末残值回收（第  5  年末收回  191.84  万元）。

默认不开爬坡的情况下，第  0  年投出  1918.4  万，第  5  年末收回残值，  5  年累计净现金流为  -1726.56  万元  ——
等等，这不是亏了吗？

> ⚠️ 校正注（评审复算）：-1726.56 万 = 1918.4 − 191.84（投资减期末残值），是**净投资敞口**而非"5 年累计现金流"；按模型公式（年净现金流 863.49 万 × 5 年 + 残值 191.84 万 − 投资 1918.4 万），5 年累计现金流应为 **+2590.9 万元**（含残值）。作者下一段的解释（"第 3 年已回本，后面是纯赚"）与此一致，此处表述用词不准确。

别急，这是因为第  5  年末的残值回收还没覆盖初始投资。实际上静态回收期只有  2.22  年，意味着在第  3  年左右就已经回本了，后面几年都是纯赚。

** 7️  ** ** ⃣  ** ** 情景对比：  保守  /  中性  /  乐观  **

三套假设并排对比，快速评估抗风险能力：

** 指标  **

|

** 保守  **

|

** 中性  **

|

** 乐观  **  
  
---|---|---|---  
  
利用率

|

55%

|

70%

|

85%  
  
token  售价

|

3.5  元

|

5  元

|

7  元  
  
单卡吞吐

|

1500

|

2000

|

2800  
  
年净利润

|

-248.8  万 

|

431.9  万

|

1892.2  万  

> ⚠️ 校正注（评审复算）：保守情景按公开公式与参数复算约为 −182.7 万（税后）/ −243.6 万（税前），与文中 −248.8 万差约 5 万（回收期 10.49 年与文中数字自洽）。推测 Excel 保守情景含未披露细节假设（如电价或固定成本微调）。引用时建议标注"含未披露细节假设"。
  
回收期

|

10.49  年

|

2.22  年

|

0.83  年  
  
保守场景下，项目基本不赚钱甚至亏损。这说明  ** 利用率、售价、吞吐量三个变量对结果影响极大  ** 。

** 8️  ** ** ⃣  ** ** 敏感性分析：  二维热力图  **

横轴是  token  售价（  3~7  元），纵轴是利用率（  50%~90%  ），交叉点的数字是回收期和  ROI  。

绿色区域代表回收快、回报高；红色区域代表慢、可能亏损。

![](https://mmbiz.qpic.cn/mmbiz_png/IPq0428B56iad1wSlodCBIsa945qryJSdFrZrPpNJRDsH98hKhmucFgu3QyicWzGuXD1icXc3pibweicf0zDZODd0YMNibyDUN6A8sT7xaicoDhndM/640?wx_fmt=png&from=appmsg)

比如利用率  50%  、售价  3  元时，回收期高达  7.14  年，基本不可行。而利用率  90%  、售价  7  元时，回收期仅  1.13
年，  ROI  高达  88.6%  。

总结一下这套模型的底层公式：

  * 年产出  =  单卡吞吐  × 3600 × (8760 ×  利用率  ) ×  卡数 

  * 收入  =  年产出  (  百万  tokens) × token  售价 

  * 电力成本  =  综合功耗  × 8760 ×  利用率  ×  卡数  ×  电价 

  * 固定成本  =  折旧  \+  人工  \+  租金  \+  带宽  \+  维护  \+  其他 

  * 总成本  =  电力  \+  固定成本 

  * 净利润  = (  收入  \-  总成本  ) × (1 -  所得税率  ) 

  * 净现金流  =  净利润  \+  折旧 

  * 静态回收期  =  初始投资  ÷  年净现金流 

  * ROI =  年净利润  ÷  初始投资 

  * 盈亏平衡利用率  =  固定成本  ÷ (  每单位利用率带来的收入  \-  电力  ) 

看似复杂，其实就是  ** 收入减成本等于利润  ** 这个最基本的道理。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/IPq0428B56iaEAGFIPoVyFqTWPDkELeIlpRHcgjjuKzxP6NzyEQYZP3sCwYIXHgEWXmMqYHj0W6k0ib2owbT9nFBhWj4jHIMjRicVn42zoqRTs/640?wx_fmt=png&from=appmsg)

** 使用方法和免责声明  **

** 这套工具是我个人的学习作品，仅供学习和交流使用。  **

1\.  ** 不构成任何投资建议  ** 。  建一个真实的  Token  工厂涉及的因素远比模型中多得多  ——
供应链风险、技术迭代、市场竞争、政策变化等等，都不是几个参数能概括的。

2\.  ** 数据仅供参考  ** 。  模型中的硬件价格、电价、  token  售价等均为估算值，实际市场行情随时在变。  2026  年的  GPU
价格、电费标准、  token  市场价格，请以最新官方公告和实际询价为准。

3\.  ** 不建议用于生产评估  ** 。  如果你真的要做  Token  工厂的商业计划书，请找专业机构做尽职调查和财务顾问。我这个业余爱好者做的
Excel  ，看看就好。

4\.  ** 后果自负  ** 。  任何人基于本工具做出的任何决策，无论盈亏，都与作者无关。我只是个想搞明白  Token  经济的码农。

** 如何使用这套工具？  **

1\.  打开「硬件投资明细」，根据实际情况修改各项硬件投入

2\.  切换到「参数假设」，修改黄色单元格的参数

3\.  看「仪表盘」获取核心结论

4\.  用「敏感性分析」和「情景对比」检验抗风险能力

5\.  看「  5  年现金流」确认回本年份

公式在  Excel/WPS  中打开会自动重算，建议保存为  .xlsx  格式。

![](https://mmbiz.qpic.cn/mmbiz_png/IPq0428B56gIPTyv56PEoiao6o3qyQLsRCT3ptOUqfznCxpjTNbxCRHQUBF9kwT2EXrn9PfsGJFu4ibutPgnw1hnAef6rJB9XPp3vXQ1uCNqI/640?wx_fmt=png&from=appmsg)

** 写在最后  **

做这个工具的过程中，我最大的收获不是算出那些数字，而是理解了  ** Token  工厂这门生意的本质  ** ：

它是一个  ** 重资产、长周期、靠规模效应  ** 的生意。前期投入巨大，但只要利用率够高、定价合理，现金流其实不错。最大的风险在于技术迭代太快  ——
你可能刚买完  GPU  ，下一代更便宜的算力就出来了。

这也是为什么我特意加了  “  售价年降幅  ”  这个参数。在  AI  这个卷得飞起的行业，今天的  5  元  /  百万  tokens
，明年可能就变成  4.75  元，后年变成  4.51  元  ……  如果不能持续提升利用率或降低单位成本，利润空间会被迅速压缩。

希望这套工具能帮到和我一样想搞懂  Token  经济的朋友们。如果有任何问题或改进建议，欢迎交流。

但记住：  ** 这只是个学习工具，不是发财秘籍。  **

💡  ** 学习心得：  ** 这个测算工具最大的价值不是给出一个标准答案，而是让我看到不同变量对结果的影响幅度（量化了）。比如我会发现，  “  每卡每秒
Token  产出  ”  和  “Token  售价  ”  这两个变量对  ROI  的弹性最大  ——  它们任何一个指标恶化  50%
，都可能让项目从盈利变成亏损。

这个你果理解了，你就真的理解了为啥运营方使出浑身解数都要做  Token  产出优化的深层原因了。

** 下载方式  ：  我把资料  通过  夸克网盘  的形式分享给大家，需要先关注  “  两年砍柴  ”  ** 公众号  ** ，  在  “  私信
”  处  发送口令  “  Token工厂财务测算工具  ”  可自动获取网盘下载链接  （  如果没有自动回复，那就是口令没输入对，强烈建议
直接复制粘贴这个口令  就行  ）。  **

以上是我的学习笔记，希望对你有所启发！

此文，毕。

提醒：  为了提升大家的阅读体验，我做了个图文链接。  点击下列图片，即可链接到对应的文章分类  ：

[
![](https://mmbiz.qpic.cn/sz_mmbiz_png/IPq0428B56hgsEfmeuCibXUgyKvLNlC2FF8syeZ1mZsbUKdw2fj3bVgod2thvSuCBibMdwI5Vjwf5DMPuD2huewcYL7iabY2EqhecmlCajluvg/640?wx_fmt=png&from=appmsg)
](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzIyMzA5NTU2NA==&action=getalbum&album_id=3957619729417699328#wechat_redirect)
[
![](https://mmbiz.qpic.cn/sz_mmbiz_png/IPq0428B56jZDflbF6AkqEkVBmaGC1K2BSZaufEZzOEgKakAd3sypapK48BxOwiaOViaP9tlticNzxMqXTn4GZp48iarejKHFr0uhqt65b9q1mU/640?wx_fmt=png&from=appmsg)
](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzIyMzA5NTU2NA==&action=getalbum&album_id=3944852037237538818#wechat_redirect)
[
![](https://mmbiz.qpic.cn/sz_mmbiz_png/IPq0428B56iaVN7hvuyd58eG0vHOzsDnibm86hh5q1iaic6sg7uxDW03eiao3HyxAxzdTia35FAhic4ZrI5Z5qlSmVR0LiavNH3eFu7ZYHSnaBqv7tU/640?wx_fmt=png&from=appmsg)
](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzIyMzA5NTU2NA==&action=getalbum&album_id=3944849422726529027#wechat_redirect)
[
![](https://mmbiz.qpic.cn/mmbiz_png/IPq0428B56jd8W2QoOSryUa0ecsabGc6S5Wsr3M6ibgCCLpBRhN9tIPT2krItzDibWCxN1I3KMvNSYia96g1gCNcA15wvxZsDLZicPpjg16HQw4/640?wx_fmt=png&from=appmsg)
](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzIyMzA5NTU2NA==&action=getalbum&album_id=3944856827350810627#wechat_redirect)
[
![](https://mmbiz.qpic.cn/mmbiz_png/IPq0428B56hzibeRdrEMA68bhr3rWibAowdX4StiaqMRS4OPRUNtaZPx8JkT9Bz0Zg3wZC6ricOHpib652QBLy0JdtcgbWia2OqKasdSfFc6qiawx4/640?wx_fmt=png&from=appmsg)
](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzIyMzA5NTU2NA==&action=getalbum&album_id=3944847120036855823#wechat_redirect)
[
![](https://mmbiz.qpic.cn/sz_mmbiz_png/IPq0428B56gmGEJuStLibPIqicjh8WIrOXxDLJsCvuKXw4YibIydjvztYdKoc57RwictAyM0tf4nIggnlb5A4N8hhBYq7DKCTt1r4foW5206aN4/640?wx_fmt=png&from=appmsg)
](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzIyMzA5NTU2NA==&action=getalbum&album_id=4048827878958694419#wechat_redirect)

![图片](https://mmbiz.qpic.cn/sz_mmbiz_gif/YibM4Wmicj8g5hbpN9qhkm9WicJFrPDkeia8U7IaWnPzevPibDpKwGCttshrfha988biaZDUsJztZ2adcZbglyzgRy0A/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=15)
![](https://mmbiz.qpic.cn/mmbiz_gif/fobTYqvvI2fZaoudk4Yibqgk722gdy2fYlFMoqrdicbBOOku71kv8pTkHWVdwI6TicpWEmZvHb6VW65Dlpk3YL9kA/640?wx_fmt=gif&from=appmsg)

  * •  [ 出路，出路，走出来才有路！困难，困难，困在家里就会觉得难！ ](https://mp.weixin.qq.com/s?__biz=MzIyMzA5NTU2NA==&mid=2247486133&idx=1&sn=fd4a481add28506b4ae129134e9cc6d2&scene=21#wechat_redirect)
  * •  [ “表达“和”销售“这两把刀，磨得越久，路越宽！ ](https://mp.weixin.qq.com/s?__biz=MzIyMzA5NTU2NA==&mid=2247486131&idx=1&sn=2000ef0d3b0d2ed15e64a0b2a50e7293&scene=21#wechat_redirect)
  * • [ 工作破局的最快方式是模仿，而不是坚持学习 ](https://mp.weixin.qq.com/s?__biz=MzIyMzA5NTU2NA==&mid=2247486113&idx=1&sn=ad9b7dedffaa29dfeea1ff13d3c6df37&scene=21#wechat_redirect)
  * • [ 《长安的荔枝》：善良若无锋芒，便是递给恶人的刀！ ](https://mp.weixin.qq.com/s?__biz=MzIyMzA5NTU2NA==&mid=2247486094&idx=1&sn=5732528d59a80f1113843c55432bd6c3&scene=21#wechat_redirect) • [ 敢于吵架、敢于冲突、敢于强势、敢于不要脸的人，才是真正有能力的人！ ](https://mp.weixin.qq.com/s?__biz=MzIyMzA5NTU2NA==&mid=2247485993&idx=1&sn=a6879e6fbaed6e08b5b5ec26739c583a&scene=21#wechat_redirect) • [ 和市场一线讲大道理，纯属自嗨！ ](https://mp.weixin.qq.com/s?__biz=MzIyMzA5NTU2NA==&mid=2247485992&idx=1&sn=3a4648259cfe0d07f42c35ed8ddb03c5&scene=21#wechat_redirect) • [ 余承东与雷军的口水战，吃瓜群众看什么？4个角度读懂商业博弈！ ](https://mp.weixin.qq.com/s?__biz=MzIyMzA5NTU2NA==&mid=2247485975&idx=1&sn=07fb9bd290c0655a375e838164a950f3&scene=21#wechat_redirect)
  * • [ 刷短视频爽？还是看公众号爽？我们每天刷的到底是信息，还是别人嚼过的观点 ](https://mp.weixin.qq.com/s?__biz=MzIyMzA5NTU2NA==&mid=2247485957&idx=1&sn=99197f27a579d2f2ce574000f25a3670&scene=21#wechat_redirect)

  

* * *

  

🌟 感谢阅读！喜欢就点个赞吧~  📌  ** 三步支持我  ** ，  让更多朋友看到优质内容：

  1. 1\. 👍  ** 点赞  ** \- 喜欢的话别忘了右下角点个赞~ 

  2. 2\. ⭐  ** 关注  ** \- 点击顶部蓝字，订阅不迷路 

  3. 3\. 🔄  ** 转发  ** \- 分享到朋友圈，知识需要传递 

（关注后右上角设为星标✨，更新第一时间看）

> 📩  ** 商务合作/转载授权  ** →  ** 先关注公众号，然后加WX（Henry_xhl）  ** ** ，加入读者qun，备注“入群“或”合作
> ** ** ”即可  ** 。

郑重申明：本公众号内容仅代表个人观点，因能力有限，恐难窥视全貌。信息上难免有错误或者疏忽，万望海涵。

![图片](https://mmbiz.qpic.cn/mmbiz_png/fobTYqvvI2fZaoudk4Yibqgk722gdy2fYzAvDzfYKJxlP4KEeyu4j37NSianBtZ20hG1FhHTv1icWRmbia4Ck0m7iaw/640?wx_fmt=png&from=appmsg)

求

![图片](https://mmbiz.qpic.cn/mmbiz_gif/fobTYqvvI2fZaoudk4Yibqgk722gdy2fYjYM15rBLxicriaUia50MemfBmWicia69gm3TJ6iaKAGbBKQzevd7TjmnAEEg/640?wx_fmt=gif&from=appmsg)

求分享

  

![图片](https://mmbiz.qpic.cn/mmbiz_gif/fobTYqvvI2fZaoudk4Yibqgk722gdy2fYMlvD2pFdhsHXqqO0XaYYtmvFMMMpJQpgTwcrn9JpFkxrtDlzelZOKQ/640?wx_fmt=gif&from=appmsg)

求喜欢

  

![图片](https://mmbiz.qpic.cn/mmbiz_gif/fobTYqvvI2fZaoudk4Yibqgk722gdy2fYluEqPaWIxoJOfs62SyfMbpsTNGFXouOaCDjSez33SiclTwQziaGFNaCg/640?wx_fmt=gif&from=appmsg)

预览时标签不可点

微信扫一扫  
关注该公众号

知道了

微信扫一扫  
使用小程序

****

取消  允许

****

取消  允许

****

取消  允许

×  分析

__

![作者头像](http://mmbiz.qpic.cn/mmbiz_png/fobTYqvvI2dkCgia0uYiaUHSl083zZYv0wcCo8avCPUlfbJGj37Tibx6v0dyOXLribiciad8OVXsibYLXYPrIvEflUKnQ/0?wx_fmt=png)

微信扫一扫可打开此内容，  
使用完整服务

：  ，  ，  ，  ，  ，  ，  ，  ，  ，  ，  ，  ，  。  视频  小程序  赞  ，轻点两下取消赞  在看  ，轻点两下取消在看
分享  留言  收藏  听过

