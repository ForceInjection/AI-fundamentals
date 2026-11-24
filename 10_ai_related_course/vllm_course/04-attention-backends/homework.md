# 第 4 讲课后作业

## 作业（必做两题）

**题 1（决策树复算报告）**：对三个「模型 × 硬件」组合手推决策树，给出优先级列表与最终选择：

1. Qwen3-1.7B(GQA)on A100(SM80);
2. DeepSeek V3.2(MLA+DSA)on B200,fp8 KV;
3. Qwen3-Next（GDN+full 混合）on H100——注意这题要走两棵树（full 层走标准树，GDN 层走 Mamba 注册表），分别给出。

每题附：走查过程（哪一层、哪个判断条件，带 `file:line`）+ 最终后端名。独立完成后再用启动日志对答案。

**题 2（报错考古）**：找一段网上（或你自己以前写的）使用 `VLLM_ATTENTION_BACKEND` 的教程/脚本，写一份「迁移说明」：原文怎么写的 → 在 HEAD 上会怎样（报什么错或静默失效）→ 正确写法。附 HEAD 上该环境变量被移除的证据（grep 结果 + PR 号）。

## 附加题（不计考核）

`attention_backends.md` 是由 gen 脚本从 `validate_configuration` 自动生成的。读 `docs/mkdocs/gen_files/generate_attention_backends.py`（约几十行），回答：自动生成解决了手写文档的哪类滞后?它解决不了哪类（提示：本讲第 3 讲发现的「能力边界」滞后 vs「类名存亡」滞后）?200 字以内。

## 自测题（不提交）

1. 为什么 MLA 与标准后端必须互斥（`is_mla()` 严格相等），而不是「MLA 后端优先」的软优先?
2. 一个新硬件厂商想接入自定义 attention 后端，最小接入路径是什么（注册 + 哪几个 `supports_*`）?
3. composite 后端(TRITON_FLASH_ATTN)为什么用工厂生成而不是手写类?（提示：它的行为=两个子后端按请求类型分工，手写会复制大量胶水）
