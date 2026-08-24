# 输出差了一点点？用 logprobs 分清「噪声」还是「bug」

> 一个真实的排查故事：KV Cache 端到端验证中，同一请求在两条执行路径上稳定输出 `artificial`（小写）与 `Artificial`（大写）。是采样噪声，还是数据 bug？这篇文章给出一个定量判别工具，并演示它如何在一整天内把根因收敛到一行代码。

## 一个让工程师失眠的问题

LLM 推理服务上线前的常见一幕：端到端测试输出与基线「几乎一致」——只有一两个 token 的差异。`Artificial` 变成了 `artificial`，开头多了个空格，某个近义词被替换。更折磨人的是：同一个差异有时出现有时不出现，同一套代码昨天复现今天消失。

这种「几乎一致」的状态是排查的泥潭。它不够错——错到一眼能看出乱码，直接进定位流程；它又不够对——对到可以直接放行。工程师卡在中间：这是硬件浮点噪声（可以容忍），还是数据在缓存链路的某个环节丢了 bit（必须修复）？两条路线的修复动作完全不同，而肉眼、逐字 diff、甚至回归测试都无法可靠地区分它们。

这篇文章介绍一个定量工具——logprobs 判别法——它把「看起来一样」变成「可度量」。方法来自一次真实的排查（Falcon-H1-7B 的 KV Cache 端到端验证），全文的数据都是那次实战的实测值。

---

## 问题：肉眼分不清的两种差异

LLM 推理的输出差异只有两个来源。

**浮点非确定性**：bf16 存储的舍入误差、kernel 执行顺序的微扰、并行归约的合并方式——这些因素让同一输入在两次计算中产生微小扰动（同代码路径的 run-to-run 抖动通常在 1e-3 nats 量级；不同实现之间的差异可能更大，见「边界与取舍」）。它天然存在，不构成 bug。

**数据位级差异**：KV Cache 或 mamba state 在缓存链路（写入缓存 → 驱逐 → 读回 → 续算）中丢了数据、写错了位置、或恢复了错误的内容。它是 bug，且往往藏在「输出基本正确」的表象下——因为一个 bit 的差异通常只会让个别低确信度 token 的 argmax 翻转。

两种来源的表现几乎相同：都表现为个别 token 的翻转。而传统的排查手段在这里全部失灵。逐字 diff 只能告诉你「有差异」，不能告诉你差异需要多大的力；回归测试同样只输出 pass/fail；逐字节对比最强，但它要改代码、挂诊断、跑真机轮次——在还不知道值不值得投入之前就用了最重的武器。

需要的是一把「分流器」：先定量判断这差异是噪声还是 bug，再决定投入多少。

---

## 工具：logprobs——模型不确定性的定量读数

OpenAI 兼容 API 有一个常被忽略的参数：`logprobs=N`。它让服务在每个生成位置返回 top-N 候选 token 的**对数概率**（logprob），而不只是 top-1 的文本。`logprobs=2` 即每个位置返回 top-2：

```text
位置 6：top-1 = " Artificial"  logprob = -0.53
        top-2 = " artificial"  logprob = -0.98
```

logprob 的单位是 nats（自然对数）。它有一个关键的数学性质：**logprob 差等于 logits 差**——softmax 的归一化常数在相减中消掉：

```text
logprob(a) - logprob(b) = (z_a - ln Z) - (z_b - ln Z) = z_a - z_b
```

这意味着 top1 与 top2 的 logprob 差**直接度量了模型在两个候选之间的原始分值距离**——不经过概率压缩、不受归一化干扰（前提：服务按原始 logits、即温度 1 的分布报告 logprobs；greedy 解码下主流引擎均如此，详见「边界与取舍」）。它是「翻转需要多大的力」的准确读数：要翻转 argmax，就必须施加超过这个 gap（top1 与 top2 的 logprob 之差）的 logits 偏移。

这个性质是整套判别方法的地基。下面的三个判别都建立在它之上。

---

## 判别一：gap——翻转需要多大的力

第一个判别看**分歧位置的 top-2 gap**——top1 与 top2 的 logprob 之差。

逻辑直接：gap 小，模型本身近等概率——小到与噪声同量级时，任何微小扰动都能翻转 argmax；gap 大，翻转就需要同量级以上的系统性 logits 偏移——远在浮点噪声之上——此时翻转只能来自数据层面的差异。

Falcon 案例的第一组数据：大小写分歧位置（`Artificial` vs `artificial`）的 gap = **0.455 nats**。这是一个很高的值——模型本来相当确信大写。它却稳定地翻成了小写。仅凭这一个数字，结论就足够明确：要让 0.455 的 gap 翻转，需要约 0.5 nats 的系统性 logits 偏移，而 bf16 浮点噪声只有 1e-3 量级——差两个数量级以上——**数据问题实锤**，采样噪声的解释被定量排除。

经验阈值：gap < 0.3 时翻转难以归因——近等概率位置本就容易受常规波动影响，判别分辨率不足；gap > 0.4 时的翻转需要系统性偏移。这两个数不是真理，是起点——不同模型、不同任务位置的近等概率分布不同，阈值的精确校准要结合自己系统的对照实验。

---

## 判别二：跨路径差——实际偏移了多大

gap 告诉我们「翻转需要多大的力」——它证伪了噪声解释，但没有测量偏移本身。第二个判别补上这一环：**同一 token 在两条执行路径上的 logprob 之差**。

方法：让同一个 prompt 走两条路径（例如基线路径与可疑路径），对每个生成位置取同一 token 的 logprob 相减。这个差值反映该位置的 logits 相对偏移（严格说是扣除分布整体漂移后的相对偏移——归一化常数只在同一分布内相减才消掉）。判据同样定量：bf16 浮点噪声对 logits 的扰动在 1e-3 nats 量级——**跨路径差超过 0.1 nats 即排除浮点噪声**（高两个数量级）。

Falcon 案例的实测：位置 1、3、4、6 的同 token 跨路径差为 **-0.20/+0.28/-0.36/+0.51 nats**（以基线减可疑路径）。每个可观测的 decode 步都在偏移，量级是噪声阈值的 2-5 倍。这不是「某个位置的特例」，而是整条生成序列的系统性偏移。数据差异从「怀疑」变成了「测量值」。

---

## 判别三：全局 vs 局部——偏移在哪里产生

偏移量级已定量，下一个问题是空间分布——它指向根因的位置。

**全局偏移**：每个 decode（解码）步都偏移相近量级（如 0.2-0.5 nats）。这意味着差异存在于 decode 开始之前的**共享输入**——prefill（预填充）阶段算出的 KV Cache、或续算依赖的 mamba state。所有 decode 步都从这个共享输入派生，所以每步都偏移。

**局部偏移**：只有个别位置偏移，其余位置正常。这指向该位置的特殊依赖——比如特定 token 组合触发的 kernel 路径差异、特定位置的注意力稀疏性。

Falcon 案例是典型的全局偏移：观测到的全部位置（1、3、4、6）都偏移 0.2-0.5 nats。这把根因锁定到 prefill 计算输入（KV 或 state），而不是 decode 期的任何随机扰动——定位范围从整个系统收敛到一条数据链路。

---

## 统一解释：随机与稳定是一回事

排查中最反直觉的现象：同一根因下，空格翻转**随机**出现（时有时无），大小写翻转**稳定**出现（每次必现）。第一反应是把它们分成两个问题——随机的是噪声，稳定的是 bug——这个二分浪费了整整一天的定位精力。

阈值模型统一解释了它们。假设存在单一的系统性 logits 偏移机制（偏移量在运行间有抖动）：

- **低 gap token**（空格，gap 0.2-0.3）：偏移量与 gap 同量级甚至更大——偏移大时翻、偏移小时不翻——所以表现随机。
- **高 gap token**（大小写，gap 0.455）：只有大偏移（0.5 级）能翻它——偏移量不足时翻不动——所以只在偏移稳定超过阈值的路径上表现稳定。

「随机 vs 稳定」不是两个问题，是同一偏移在不同 gap 阈值下的表现。低 gap token 是灵敏但吵闹的探测器，高 gap token 是迟钝但可靠的指示器——两者组合使用：空格告诉你「这里有差异」，大小写告诉你「差异有多大」。这个认识把两个看似矛盾的现象合并为一个根因假设，是当天定位从泥潭转入正轨的转折点。

---

## 定位闭环：从现象到根因的三对照

logprobs 判别回答了「是不是数据问题」（是）和「量级多大」（0.2-0.5 nats，全局）——但它不能定位根因。判别的终点是定位的起点，而定位用数据面的三对照：

1. **写侧对照**：在数据写入缓存介质前 dump 一份 CPU 副本（物化点），与写入缓存后的读回对比——验证写路径无损。
2. **介质对照**：直接 mmap 缓存设备内存，按真实偏移读数据——验证数据确实存在于介质上。
3. **读侧对照**：load 恢复后，与写侧的 dump 逐层对比——找到差异出现的位置。

Falcon 案的三对照结果：写侧无损（dump 与物化完全一致）；介质有数据；**读侧丢层**——load 恢复的 temporal state 只有 layer 0 有值，layer 1-43 全零。由此收敛到代码级根因：temporal state 是单个 tensor 的 44 层视图，共享同一 IPC handle。注册时只有每层长度、没有层偏移，SAVE/BLOAD 的寻址全部落在层 0。44 层写的是同一份数据，读回的也只有层 0。续算从这个半零的 state 出发，每一步都偏移 0.2-0.5 nats——与 logprobs 判别的全部结论吻合。

根因是一行修复（注册时记录层偏移），但找到它的路径是完整的判别 + 三对照链条。

---

## 边界与取舍

logprobs 判别有明确的边界，使用前应当知道：

- **API 依赖**：需要服务支持 `logprobs` 参数（OpenAI 兼容服务基本都有）；判别在 greedy 解码（temperature=0）下才有意义——采样会引入自己的随机性，污染 gap 读数；且部分引擎（如 SGLang 的非 greedy 路径、vLLM v0）在 temperature≠1 时会按 1/T 缩放报告值，gap 读数随之变形，greedy 解码恰好规避了这一问题（主流引擎在 greedy 下均按温度 1 的原始分布报告）。
- **阈值是经验值**：0.3/0.4（gap）、0.1（跨路径差）来自 bf16 噪声量级与实战标定。注意噪声量级的口径：同代码路径的 run-to-run 抖动在 1e-3 nats 量级，而不同实现/不同 kernel 路径之间的 bf16 logits 差异可能达到 1e-2 量级——0.1 阈值相对后者的余量约一个数量级。其他精度体系（fp8/fp4 量化、不同模型的 logits 标度）需要重新标定；最稳妥的做法是用两条路径各自的重复运行直接测量噪声地板。
- **只证有不证无**：大 gap 翻转强有力地证明「存在数据差异」；但小 gap 翻转**不能**证明「没有数据差异」——低 gap token 对偏移敏感，翻转了可能是噪声，没翻也可能只是偏移还没到阈值。判别法擅长「抓出数据问题」，不擅长「放行」。

与替代方案相比：逐字节对比最强，但成本最高——要改代码、挂诊断、跑真机轮次，适合判别之后的定位阶段。回归测试只报差异、不报差异的力，适合门禁、不适合定位。**logprobs 判别几乎零成本**（一个 API 参数），却能把「噪声 vs bug」这个最贵的岔路口提前分掉。三者是流水线，不是替代品：先 logprobs 分流，再逐字节定位，最后回归测试守门。

---

## 结语：把「看起来一样」变成「可度量」

一句话方法论：**先定量判别，再定位根因**。

输出差异排查的第一动作不是 grep 代码，而是拿 logprobs——gap 告诉你翻转需要多大的力，跨路径差告诉你实际偏移了多大，两者的对比把「采样噪声」和「数据 bug」分开。分开之后，全局偏移指向 prefill 输入，三对照收敛根因。

那天花在判别上的时间，换来的是省下的数天错误定位——以及最重要的：一个可以重复使用的判断框架，而不是又一次「重启试试」。

---

## 附录：一次请求的完整判别示例

从请求到结论的完整数据流（SGLang `/v1/completions`，greedy 解码 + `logprobs=2`；数值取自 Falcon 实测，为便于演示做了简化整理：gap 取 0.45（正文 0.455 的两位近似）、跨路径差取位置 6（0-based 索引 5）的 -0.51，方向约定为基线减可疑路径，量级与正文一致）：

**1. 请求**

```bash
curl -s http://localhost:30001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Falcon-H1-7B-Instruct",
    "prompt": "Consider the following context about artificial intelligence and machine learning: ...  Question: what is AI?",
    "max_tokens": 20,
    "temperature": 0.0,
    "logprobs": 2
  }'
```

**2. 响应**（OpenAI completions 格式——`tokens` / `token_logprobs` / `top_logprobs` 三个并行数组；R1 为可疑路径的响应）

```json
{
  "id": "cmpl-xxxxxxxx",
  "object": "text_completion",
  "created": 1756000000,
  "model": "Falcon-H1-7B-Instruct",
  "choices": [
    {
      "index": 0,
      "text": " Answer: AI stands for artificial intelligence, which refers to the simulation of human intelligence in machines that",
      "logprobs": {
        "tokens": [
          " Answer",
          ":",
          " AI",
          " stands",
          " for",
          " artificial",
          " intelligence",
          ",",
          " which",
          " refers"
        ],
        "token_logprobs": [
          -1.3, -0.02, -0.001, -0.01, -0.05, -1.04, -0.02, -0.03, -0.01, -0.04
        ],
        "top_logprobs": [
          { " Answer": -1.3, "  Answer": -1.303 },
          { ":": -0.02, ";": -5.1 },
          { " AI": -0.001, " ai": -7.2 },
          { " stands": -0.01, " Stand": -8.5 },
          { " for": -0.05, " For": -6.8 },
          { " artificial": -1.04, " Artificial": -1.49 },
          { " intelligence": -0.02, " Intelligence": -5.3 },
          { ",": -0.03, ".": -3.2 },
          { " which": -0.01, " Which": -7.9 },
          { " refers": -0.04, " Refer": -6.4 }
        ]
      }
    }
  ],
  "usage": {
    "prompt_tokens": 2206,
    "completion_tokens": 20,
    "total_tokens": 2226
  }
}
```

**3. 从响应计算两个判别量**（解析逻辑；`top_r1` / `top_r3` 为两条路径响应的解析结果，本示例中 R1 为可疑路径、R3 为基线路径）

```python
import json

def load(p):
    d = json.load(open(p))
    c = d["choices"][0]
    lp = c.get("logprobs", {})
    return c["text"], lp.get("tokens", []), lp.get("token_logprobs", []), lp.get("top_logprobs", [])

# 判别一：分歧位置的 top-2 gap
def top2_gap(top_logprobs, i):
    s = sorted(top_logprobs[i].items(), key=lambda kv: -kv[1])[:2]
    return s, s[0][1] - s[1][1]

_, _, _, top = load("/tmp/lp_r1.json")   # 可疑路径响应
top2, gap = top2_gap(top, 5)             # 0-based 位置 5 = " artificial"（正文的「位置 6」）
# top2 = [(" artificial", -1.04), (" Artificial", -1.49)]
# gap = 0.45  ← 大 gap：翻转需要约 0.45 nats 的力——噪声（1e-3 量级）不够 → 数据问题实锤

# 判别二：同 token 跨路径差（R1 可疑路径 vs R3 基线路径，同一 token 的 logprob）
def cross_path_diff(top1, top3, i, token):
    return top1[i].get(token) - top3[i].get(token)

# R1: " artificial" = -1.04；R3: " artificial" = -0.53
diff = cross_path_diff(top_r1, top_r3, 5, " artificial")
# diff = -0.51  ← |diff| > 0.1 nats → 排除 bf16 浮点噪声
```

**4. 判别结论对照**

| 步骤     | 计算                | 值           | 判据                     | 结论                |
| -------- | ------------------- | ------------ | ------------------------ | ------------------- |
| gap      | top1 - top2 logprob | 0.45 nats    | > 0.4 → 翻转需系统性偏移 | 数据问题（非噪声）  |
| 跨路径差 | 同 token R1 - R3    | -0.51 nats   | > 0.1 → 排除 bf16        | 偏移量级实测        |
| 全局性   | 各位置逐一算        | 每步 0.2-0.5 | 全部位置偏移             | 根因在 prefill 输入 |
