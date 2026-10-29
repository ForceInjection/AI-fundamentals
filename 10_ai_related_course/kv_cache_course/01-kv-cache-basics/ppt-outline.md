# 第 1 讲：KV Cache 为什么存在 — PPT 大纲

## 设计说明

- **总页数**：约 21 页
- **演讲时长**：120 分钟（30 min 概念 + 45 min 公式/演示 + 35 min 实验 + 10 min 答疑）
- **建议风格**：深蓝 + 暖灰；Q 用蓝色、K 用橙色、V 用绿色；当前 token 高亮，历史 token 降低饱和度
- **字体**：标题 32–36 pt，正文 18–22 pt，等宽字体用于公式和命令
- **教学策略**：每个概念后立即手算一次；QKV 只讲“投影 + 一行注意力”，不展开完整 Transformer
- **可视化策略**：
  - QKV 投影参考 Jay Alammar 的静态图，只取 self-attention 部分；不复制原图，课堂打开原图或自己重绘
  - 一行注意力用本讲自绘的 3-token、`d=2` 玩具例子，数字可手算
  - 现场交互用 Transformer Explainer，固定短句、layer 0/head 0、只点一个 token，不展开 12 层
  - 3Blue1Brown 放课前/课后；bbycroft 3D 只作拓展；BertViz 只看 attention 模式，与 KV 存储无关
- **版权**：Alammar 图只放链接或重绘；Transformer Explainer 保留 MIT 出处；视频和 3D 工具只放官方链接
- **动手环节**：第 17–19 页，学生在终端运行 `code/kv_estimator.py`

---

## 开场（第 1–2 页）

### 第 1 页：封面

- **标题**：KV Cache 为什么存在
- **副标题**：从一次自回归生成，算清一个 token 的账
- **课程信息**：KV Cache 八讲 · 第 1 讲 · 120 分钟
- 日期、讲师信息
- **一张图**：左侧一条生成中的 token 序列，右侧一本逐项累加的“显存账本”

### 第 2 页：本讲议程（标注动手环节）

- **第一部分**：decode 为什么需要 KV Cache（30 min）— 第 3–9 页
  - 自回归 / QKV 投影 / 3-token 手算 / 历史 K/V 不变 / prefill 与 decode
- **第二部分**：一个 token 的账（45 min）— 第 10–16 页
  - 两行公式 / 精度 / MHA/GQA/MQA / MLA / TTFT、TPOT、吞吐
- **第三部分**：动手实验（35 min）— 第 17–19 页
  - 5 min 自检 + 30 min 填表；Qwen3-0.6B、Qwen2.5-7B、Qwen2.5-72B
- **总结与答疑**：10 min — 第 20–21 页
- 本讲只回答两个问题：decode 为什么必须缓存 K/V；一个 token 的 KV 账怎么算

---

## 第一部分：decode 为什么需要 KV Cache（30 min，第 3–9 页）

### 第 3 页：自回归生成 — 每步只多一个 token

- 序列：`x1, x2, ..., xt`，第 t 步预测 `x_{t+1}`
- 因果掩码：位置 t 只能看到 `1..t`，看不到未来
- 每步的输入只比上一步多一个 token

  ```text
  x1 → x2 → x3 → ... → xt
        ↑ 每步只新增一个 token
  历史位置 1..t-1 的 K/V 不再改变（第 6 页解释为什么）
  ```

- 训练时整段并行；推理 decode 时逐 token 串行
- **一句话**：decode 的成本不在“再算一个 token”，而在“不要重算历史”

### 第 4 页：Q/K/V 是三个投影

- 每个 token 的表示 `x_i` 经过三个线性投影：

  ```text
  q_i = W_Q x_i
  k_i = W_K x_i
  v_i = W_V x_i
  ```

- **Q**：当前 token 发出的查询
- **K**：历史位置和当前 token 的键，用来算相关性
- **V**：历史位置和当前 token 的值，用来加权求和
- **搜索类比**：Q 是搜索词，K 是结果标题，V 是网页内容；注意力就是“用 Q 去匹配所有 K，再按权重取回 V”
- **主图**：打开 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)，看 “Self-Attention at a High Level” 这一节；只讲 `XW^Q`、`XW^K`、`XW^V` 和一个 query 行，跳过 encoder-decoder、多头和位置编码
- **可选现场交互**：打开 [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)，固定短句、layer 0/head 0，只点一个 token；不要展开 12 层
- **这一页只回答“Q/K/V 是什么”**；形状和 softmax 下一页手算

### 第 5 页：3-token 手算 — 一个 query 行

- 为手算方便，取 3 个 token、`d=2`、单头，并令 `W_Q=W_K=W_V=I`，直接给 Q/K/V：

  | token | q | k | v |
  | --- | --- | --- | --- |
  | 1 | `[1, 0]` | `[1, 0]` | `[1, 0]` |
  | 2 | `[0, 1]` | `[0, 1]` | `[0, 1]` |
  | 3 | `[1, 1]` | `[1, 1]` | `[1, 1]` |

- 只算 token 3 的 query：`q3 = [1, 1]`
- 与三个 K 做点积：`[q3·k1, q3·k2, q3·k3] = [1, 1, 2]`
- 除以 `sqrt(d)=1.414`：`[0.707, 0.707, 1.414]`
- softmax：`[0.248, 0.248, 0.503]`（约）
- 加权 V：`0.248×[1,0] + 0.248×[0,1] + 0.503×[1,1] ≈ [0.751, 0.751]`
- **注意**：这只是“一个 head、一个 query 行”的演示；真实模型有多个 head、多个 token，但计算顺序相同

### 第 6 页：历史 K/V 为什么不变

- 因果掩码让位置 i 的隐藏状态只依赖 `x1..xi`
- 权重固定，因此 `k_i = W_K h_i`、`v_i = W_V h_i` 一旦算出就不再变化
- 新 token 只追加自己的 `k_t`、`v_t`，不会改写历史 K/V
- 不缓存：第 t 步重算 `1..t` 的所有 K/V
- 缓存：第 t 步只算新 token 的 K/V，历史 K/V 从显存读

  ```text
  不缓存：每步重算整段历史
  缓  存：历史 K/V 存起来，只算新 token
  ```

- **一句话**：历史 K/V 不变，是因为因果掩码让每个位置只依赖前缀，且权重固定

### 第 7 页：Prefill — 一次算完整段 prompt

- 输入：整段 prompt 的 `T_prompt` 个 token
- 计算形态：矩阵 × 矩阵（GEMM），可以并行
- 长 prompt 下通常 compute-bound；短 prompt、小 batch 时要看权重加载和 kernel 启动开销
- KV Cache 在 prefill 阶段写入；chunked prefill 时逐块写入，降低单步 activation 峰值，不减少 KV 总量
- TTFT 包含：排队 + prefill + 首 token 采样（第 14 页展开）

  ```text
  时间轴：排队 → prefill → 首 token
                        └── TTFT ──┘
  ```

- **一句话**：prefill 是“把 KV 写进去”的阶段

### 第 8 页：Decode — 每步一个 token

- 每步输入：1 个新 token
- 计算形态：矩阵 × 向量（GEMV），并行度低
- 单请求每步要读历史 K/V：`per_token_bytes × 当前序列长度`；系统总量再乘 batch
- 算术强度 = FLOPs / 读取字节；decode 的 GEMV 算术强度低，典型小 batch、长上下文下受显存带宽限制
- batch 增大后权重读取被摊薄，算术强度上升，可能转向 compute-bound；序列变长通常增加 KV 流量，更偏 memory-bound
- 例：Qwen3-0.6B、FP16、`T=32768`、batch=1，一份 KV 约 3.5 GiB；每步都要读一遍

  ```text
  HBM（历史 K/V） ──────→ SM（只算 1 个 token）
         ↑ 每步读取全部历史
  ```

- **一句话**：decode 的瓶颈经常不是算力，而是“把 KV 搬到 SM”

### 第 9 页：第一部分小结 — 因果链与缓存交易

- 因果掩码 → 历史位置只依赖前缀 → 固定权重下 K/V 不变 → 可以缓存
- 缓存把重复计算换成显存占用和显存带宽
- prefill 写入 KV，decode 读取 KV；两段瓶颈不同
- 下一部分：把“一个 token 占多少”算成两行公式，并补缓存形状
- **过渡**：先算清 token 账，才能讨论 block、page 和容量规划

---

## 第二部分：一个 token 的账（45 min，第 10–16 页）

### 第 10 页：KV 公式与缓存形状

```text
per_token_bytes = 2 × b_kv × L × H_kv × D
total_bytes     = per_token_bytes × T_total × batch
```

- 缓存形状：K、V 各一份 `[L, T_total, H_kv, D]`；元素总数 `2 × L × T_total × H_kv × D`，再乘 `b_kv`

| 符号 | 含义 | 说明 |
| --- | --- | --- |
| `L` | Transformer 层数 | 每层各存一份 K/V |
| `H_kv` | KV head 数 | 不是 Q head 数 |
| `D` | 每个 head 的维度 | 取 config 的 `head_dim` |
| `b_kv` | 每元素字节 | FP16/BF16=2，FP8/INT8=1，INT4=0.5 |
| `T_total` | 当前总序列长度 | 包含 prompt 和已生成 token；prefill 的 prompt 长度写 `T_prompt` |
| `batch` | 并发请求数 | 不要再用单字母 `B` 同时表示 batch 和 bytes |

- 公式开头的 `2`：K 和 V 各存一份
- 精度只改 `b_kv`；scale、打包和 block 对齐另算
- 权重和 KV 分开记账

### 第 11 页：Qwen3-0.6B 算例与精度表

- Qwen3-0.6B：`L=28`、`H_q=16`、`H_kv=8`、`D=128`
- FP16：`2 × 2 × 28 × 8 × 128 = 114,688 B/token`，即 112 KiB/token
- `T=32768`、batch=1：`112 KiB × 32,768 ≈ 3.5 GiB`
- 注意：`head_dim` 是 128，不是 `hidden_size / H_q = 64`

| 精度 | `b_kv` | 说明 |
| --- | --- | --- |
| FP16 / BF16 | 2 B | 容量相同，BF16 动态范围更大 |
| FP8 / INT8 | 1 B | 需要 kernel 支持；scale 开销另算，脚本输出是不含 scale 的下界 |
| INT4 | 0.5 B | 打包后实际占用通常更高；本讲不展开，第 7 讲涉及 FP4 |

- Qwen3 系列没有 7B / 72B；本课程用 Qwen3-0.6B 搭配 Qwen2.5-7B、Qwen2.5-72B

### 第 12 页：MHA / GQA / MQA — 只改 `H_kv`

| 变体 | `H_kv` | 缓存内容 | 例子 |
| --- | --- | --- | --- |
| MHA | `= H_q` | 每个 Q head 一份 K/V | 传统多头注意力 |
| GQA | `< H_q` | 多个 Q head 共享一组 K/V | Qwen3-0.6B 为 16:8；Qwen2.5-7B 为 28:4 |
| MQA | `= 1` | 所有 Q head 共享一组 K/V | 更激进的压缩 |

- 公式不变，只换 `H_kv`；`H_kv` 越小，KV 越省，但表达能力和效果要重新评估
- 注意形状：`H_kv` 是 KV head 数，不是序列长度 `S`

  ```text
  单 query（decode，单头）：
    q [D] × K^T [D, S] → scores [S]
    scores × V [S, D] → output [D]

  批量（prefill，多头）：
    Q [H_q, T_q, D] × K^T [D, T_kv] → scores [H_q, T_q, T_kv]
    scores × V [T_kv, D] → output [H_q, T_q, D]
  ```

- K/V 张量是 `[H_kv, T_kv, D]`；GQA 时每个 KV head 服务一组 Q head；`scores` 的最后一维是序列长度，不是 `H_kv`
- 这一页不讨论 MLA；MLA 不是简单改 `H_kv`

### 第 13 页：MLA — latent + RoPE，不乘 2

```text
per_token_bytes = b_kv × L × (kv_lora_rank + qk_rope_head_dim)
```

- MLA 不按每个 head 存 K/V，而是存一份所有 head 共享的 latent 和一份共享 RoPE key
- 公式里没有 `2`，也没有 `H_kv`
- DeepSeek-V3：`kv_lora_rank=512`、`qk_rope_head_dim=64`、61 层
  - BF16：每层每 token 约 `(512 + 64) × 2 = 1,152 B`（约 1.13 KiB）
  - 61 层合计约 68.6 KiB/token；FP8 下每层 576 B，61 层合计 35,136 B（约 34.3 KiB/token）
- GLM-5.3-Flash 的 `qk_rope_head_dim=0`，只算 512，不要套 DeepSeek 的 64
- **一句话**：MLA 改变的是缓存对象，不是只把 `H_kv` 调小

### 第 14 页：TTFT / TPOT / 吞吐 — 三个指标

- **TTFT**（Time To First Token）：请求发出到第一个输出 token 返回，包含排队、prefill 和首步采样
- **TPOT**（Time Per Output Token）：首 token 之后平均每个输出 token 的间隔
  - `TPOT = (总时延 − TTFT) / (输出 token 数 − 1)`；输出 token 数 < 2 时 TPOT 无定义，只报 TTFT
- **吞吐**：单位时间完成的 token 数或请求数
- **KV → 指标**：
  - KV 变大 → decode 每步读更多 → TPOT 变大
  - prefill 写 KV 多 → TTFT 变大
  - batch 增大 → 吞吐升，但 TPOT 可能升
  - FP16→FP8 → 同显存可放更多 batch/token

  ```text
  时间轴：排队 → prefill → 首 token → decode token 1 → ... → 最后一个 token
                        └── TTFT ──┘   └────────── TPOT ──────────┘
  ```

- 报指标时带 batch、输入长度、输出长度和硬件，否则数字不可比

### 第 15 页：五个常见算错点

- **把 `H_q` 当成 `H_kv`**：GQA 模型会把 KV 算大几倍
- **把 Qwen3-0.6B 的 `head_dim` 当成 64**：会把 112 KiB/token 算成 56 KiB
- **把 per-token 公式当成总量公式**：忘了乘 `T_total` 和 `batch`
- **把 `T_prompt` 当成 `T_total`**：prefill 长度和当前总长度不是一回事
- **TP 时忘了切 `H_kv`**：TP=8 时每卡 `H_kv = H_kv / 8`，不是每卡仍用全局 `H_kv`
- 另外两条口径：权重和 KV 分开记账；报容量时写清模型、精度、序列长度、batch

### 第 16 页：第二部分小结 — 把 token 账变成可计算模型

- `per-token` 由 `L`、`H_kv`、`D`、`b_kv` 决定；`total` 再乘 `T_total` 和 `batch`
- 缓存形状是 `[L, T_total, H_kv, D]`，K/V 各一份
- MHA/GQA/MQA 改 `H_kv`；MLA 改缓存形态
- 这些数字将在第 2 讲变成 block/page/slot 账
- **过渡**：下一页开始动手，用脚本把三个模型算一遍

---

## 第三部分：动手实验（35 min，第 17–19 页）

### 第 17 页：[动手] 实验引导 — 先跑自检

- **目标**：用 `kv_estimator.py` 算 Qwen3-0.6B、Qwen2.5-7B、Qwen2.5-72B 的每 token KV 与总量
- **工具**：`10_ai_related_course/kv_cache_course/01-kv-cache-basics/code/kv_estimator.py`
- 只依赖 Python 标准库，不需要 GPU、PyTorch 或模型权重

  ```bash
  cd 10_ai_related_course/kv_cache_course/01-kv-cache-basics
  python3 code/kv_estimator.py --self-check
  python3 code/kv_estimator.py --model qwen3-0.6b --seq 4096 --batch 8 --dtype fp16
  ```

- `--self-check` 至少应包含：Qwen3-0.6B 112 KiB、Qwen2.5-7B 56 KiB、Qwen2.5-72B 320 KiB（均 FP16）；还会打印 Qwen3-0.6B FP8 56 KiB 和 3.50 GiB 总量自检
- 预计 5 min 安装与自检 + 30 min 填表和讨论

### 第 18 页：[动手] 实验任务 — 填一张容量表

- 固定 `T=4096`，覆盖 2 个模型 × 2 种精度 × 3 个 batch = 12 行；Qwen2.5-72B 另做预算判断

  ```bash
  for model in qwen3-0.6b qwen2.5-7b; do
    for dtype in fp16 fp8; do
      for batch in 1 8 32; do
        python3 code/kv_estimator.py --model "$model" --dtype "$dtype" --seq 4096 --batch "$batch"
      done
    done
  done
  python3 code/kv_estimator.py --model qwen2.5-72b --gpu-gib 40
  python3 code/kv_estimator.py --model qwen2.5-72b --gpu-gib 80 --tp 8 --seq 4096
  ```

| 模型 | L | H_kv | D | 精度 | T | batch | 每 token KV | 总 KV | 权重（FP16） | KV 与权重谁更大 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 28 | 8 | 128 | FP16 | 4096 | 8 | 114,688 B（112 KiB） | 3.5 GiB | 1.12 GiB | KV |
|  |  |  |  |  |  |  |  |  |  |  |

- 72B 单卡 40 GiB：脚本并发列显示“—”，预算状态显示“权重（超出预算）”；只记录每 token KV，不写负并发
- 8×80 GiB、`--tp 8`：每卡权重约 16.9 GiB；`T=4096`、FP16 KV 时并发约 350，FP8 KV 约 700
- 给了 `--gpu-gib` 时，脚本输出“每卡总 KV / 每卡权重 / 并发上限 / 预算状态”，列名和上面的总表不同
- **注意**：`T=4096、batch=8` 和 `T=32768、batch=1` 都约 3.5 GiB，不是同一笔账

### 第 19 页：[动手] 结果讨论 — 哪本账先爆

- 预期数字：
  - Qwen3-0.6B：112 KiB/token；`T=4096、batch=8` 约 3.5 GiB
  - Qwen2.5-7B：56 KiB/token
  - Qwen2.5-72B：320 KiB/token；单卡 40 GiB 权重 135.41 GiB，先爆
- 同一模型：FP16→FP8，每元素字节减半 → 每 token KV 减半；batch 翻倍 → 总 KV 翻倍
- 讨论：在什么模型、`T` 和 batch 下，KV 会先成为瓶颈？用 0.6B 和 72B 各举一个正例/反例
- 常见错误：`H_q` 当 `H_kv`、`head_dim` 用 64、漏乘 `batch`、把两个 3.5 GiB 当成同一账
- **过渡**：第 2 讲把这些 token 放进 block/page/slot

---

## 总结与答疑（10 min，第 20–21 页）

### 第 20 页：本讲小结

- 三句话：
  1. decode 每步只多一个 token，历史 K/V 不必重算
  2. `per_token_bytes = 2 × b_kv × L × H_kv × D`，总量再乘 `T_total × batch`
  3. MHA/GQA/MQA 改 `H_kv`；MLA 改缓存形态
- 一个 token 的账，决定后面的 block/page/容量规划
- 课后作业：实现一个 MHA/GQA 估算器，覆盖 Qwen3-0.6B 与 Qwen2.5-7B、FP16/FP8
- **一句话**：KV 是显存问题，不只是算法问题

### 第 21 页：下次预告 + 参考资料

- 下次：第 2 讲《KV Cache 怎么放》— token、block、page、slot、block table
- 课后练习：用自己的模型 config 复算一遍，并和 `kv_estimator.py` 对答案
- **课堂主用可视化**：
  - [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)：只看 “Self-Attention at a High Level” 的 QKV 投影图
  - [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)：固定短句、layer 0/head 0、只点一个 token
- **课前/课后**：
  - [3Blue1Brown: Attention in transformers](https://www.3blue1brown.com/lessons/attention)
  - [LLM Visualization](https://bbycroft.net/llm)（3D 拓展）
  - [BertViz](https://github.com/jessevig/bertviz)（只看 attention 模式，与 KV 存储无关）
- **本地参考**：
  - [AI-fundamentals KV Cache 技术体系](../../../09_inference_system/kv_cache/README.md)
  - [KV Cache 基本概念](../../../09_inference_system/kv_cache/01_concepts/basic/)
- **版权**：Alammar 图只放链接或重绘；Transformer Explainer 保留 MIT 出处；视频和 3D 工具只放官方链接
- Q&A：留 3 分钟，重点检查 `H_kv`、`head_dim`、per-token/total、`T_prompt`/`T_total` 四个易错点
