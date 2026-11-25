# 第 8 讲课后作业

## 作业（必做两题）

**题 1（kernel 路由报告）**：三个「checkpoint × 硬件」组合，各推演加载路径与 kernel：

1. AWQ checkpoint(uint4,group 128)on H100;
2. 同 AWQ checkpoint 开 batch invariance；
3. DeepSeek-V3.2 官方 checkpoint(block-wise FP8,128×128)on H200。

每题给出：被认领的方法名（overrides 表哪个条目）、选中的 kernel（`file:line` 依据）、可能的回退路径。

**题 2（迁移说明）**：找一段使用 `--quantization fp8` 做「BF16 在线量化」的旧教程/脚本，写迁移说明：原写法 → HEAD 上的实际行为（`fp8.py:105-108` 报错）→ 正确写法（`fp8_per_tensor` shorthand）→ 两者语义差异（如有）。

## 附加题（不计考核）

读 `docs/features/quantization/README.md` 的两条老口径更正（Marlin 恒 16-bit 激活；`fp_quant`/QuTLASS 移除），结合 `humming` 的定位（W4A8 替代），写 200 字：Marlin 在 vLLM kernel 体系中的角色如何从「全能反量化」收缩为「W4A16 专用」？这个收缩对维护者意味着什么？

## 自测题（不提交）

1. `_verify_quantization` 的 overrides 优先级表解决什么问题？（多个 quant 描述共存时的认领顺序）
2. `process_weights_after_loading` 为什么必须存在？举两件它做的事（scale requant / 权重转置）。
3. `--kv-cache-dtype fp8` 后 prefix caching 可能被谁关掉？（引用第 4 讲）
