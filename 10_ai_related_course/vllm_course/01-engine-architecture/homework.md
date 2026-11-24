# 第 1 讲课后作业

## 作业（计入考核，作业 1）

**画出一次请求在当前 vLLM（v0.34 基线）的全生命周期图。**

要求：

1. 进程级：三类进程 + coordinator（如适用），画出 ZMQ 与共享内存两类通道，标明 bind/connect 方向。
2. 类/方法级：至少覆盖七个环节的关键函数（README 七环节的函数名），每个节点标 `file:line`（按基线 `1acec5c009` 核对过的行号）。
3. 数据形态：标注关键数据在边界上的形态（`EngineCoreRequest` / `SchedulerOutput` / `ModelRunnerOutput` / `EngineCoreOutputs` / `RequestOutput`），序列化方式(msgpack / shm)。
4. 一页为限（手绘拍照或绘图工具均可）。

验收要点：tokenizer 的位置；两套 IPC 不混淆；`EngineCore.step` 的五步完整；数据形态在进程边界处标注正确。

## 附加题（不计考核）

仿照本讲「文档说的 vs 代码看到的」小节，在 `docs/design/arch_overview.md` 或 `docs/usage/v1_guide.md` 里再找一处与 HEAD 代码不一致的表述，给出：文档原话、代码实际（`file:line`）、你判断谁对谁错及理由。写三条以内，宁缺毋滥。

## 自测题（不提交）

1. 为什么 tokenizer 不放在 EngineCore 进程里?（提示：EngineCore 是计算 busy loop,decode 字符串是 IO 型工作；且多 EngineCore 时会重复做）
2. `AsyncLLM.generate` 里客户端拿到的是不是模型输出张量?在哪一层变成了文本?
3. `--tensor-parallel-size 2` 时进程数怎么变化?哪一段代码拉起了新进程?
