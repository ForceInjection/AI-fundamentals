# 手搓系列导航

这个系列做一件事：把框架里「拿来就用」的机制，用最小可运行的代码搓出来。每个主题一段 numpy 或 Python（百行级），不依赖框架内部接口，跑得动、算得清、能复现，配套手绘图把机制画开。读的时候建议开着代码对照正文，每篇的数字都来自实测。

阅读顺序按依赖递进：

| #   | 篇目                                                                                                                                | 搓的是什么                  | 实测产出                                                        |
| --- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------- | --------------------------------------------------------------- |
| 1   | [手搓 Tokenizer：从字符级到 BPE](06_llm_theory_and_fundamentals/llm_basic_concepts/token/hand_roll_bpe_tokenizer.md)                | 子词切分怎么长出来          | 压缩曲线与过拟合实测，tiktoken 同语料对照                       |
| 2   | [手搓 RoPE：一百行 numpy 讲透旋转位置编码](06_llm_theory_and_fundamentals/llm_basic_concepts/positional_encoding/hand_roll_rope.md) | 旋转位置编码与外推          | 配对约定差 3.385、相对不变性 3e-13、三种外推策略频率谱          |
| 3   | [手搓 Attention：从三行循环到看懂 FlashAttention](02_gpu_programming/02_cuda/21_write_attention_from_scratch.md)                    | 注意力及其 IO-aware 改写    | 四版递进实跑，从 3 行朴素版到分块 online softmax                |
| 4   | [手搓 KV Cache：从静态预留到 PagedAttention 的页表](09_inference_system/kv_cache/01_concepts/01_basic/03_hand_roll_kv_cache.md)     | KV 显存管理与分页           | 模拟器实测三种策略：碎片、入住失败与 5 倍容量的数字来源         |
| 5   | [手搓 MLA：从 MHA 到 weight absorption](09_inference_system/kv_cache/01_concepts/01_basic/07_hand_roll_mla.md)                      | KV 的 latent 压缩与吸收变换 | 真实权重上吸收等价性 fp32 逐位成立；KV 压到同形状 MHA 的约 1/57 |

前四篇搭出「一个 Transformer 怎么算、怎么存」，第五篇开始动缓存的形状。各篇散落在对应主题区（06 / 02 / 09），本页是唯一入口。

## 系列风格约定

每篇同一套骨架：开头元信息（核对基准与日期）→ 场景式开场 → 中文数字章节的过程叙事 → 实测数字与配套脚本（同目录 `.py`，全部数字可复现）→ 总评与适用边界 → 源文件索引与参考资料。语言上平实陈述，机制的递进自带悬念。
