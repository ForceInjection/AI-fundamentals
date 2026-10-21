# assets

本目录存放 memory_native 两篇文章的配图。

## MemT-1.0

来源文章：[当记忆成为模型状态：MemT-1.0 的架构、训练与边界](../memt_1_0_memory_native.md)

- `memt-01-memory-block.png`：记忆块的读路径与写路径
- `memt-02-training-pipeline.png`：MemT-1.0 的训练管线

## EdgeRock-4B

来源文章：[EdgeRock-4B 拆解：把 KV Cache 变成固定容量记忆](../edgerock-4b-constant-memory.md)

- `edgerock-01-rma-flow.png`：RMA 的写入、准入与读取
- `edgerock-02-memory-account.png`：每条序列的持久历史账本
- `edgerock-03-throughput.png`：单请求与高并发吞吐的口径

手绘图由 rough.js + 霞鹜文楷程序化生成，1080 逻辑宽，2x 渲染 PNG；图内为示意，数字与机制以论文和官方仓库为准。
