# Ascend C 算子开发入门

## 1. 什么时候需要自定义算子

大多数 PyTorch 标准算子（Conv、BN、Attention 等）已有 CANN 内置优化实现。以下场景需要自定义算子：

- 模型中使用了自定义 CUDA kernel（如 FlashAttention 变体、特殊的激活函数）。
- 需要将多个小算子融合为一个来减少内存访问。
- 使用非标准数值计算（如自定义量化格式）。

## 2. 算子开发工具链

CANN 提供三层算子开发体系（内置算子 OPP → Ascend C → TBE），详见 [CANN 软件栈详解 — §2.3 算子层](../02_ascend_architecture/02_cann_software_stack.md)。本文聚焦中间层 Ascend C 的开发流程。

工具：

| 工具 | 用途 |
|------|------|
| `msopgen` | 自动生成算子项目模板 |
| Ascend C Compiler (`ccec`) | 编译 Ascend C 算子代码为 .o 文件 |
| `opc` | 算子编译工具 |

## 3. Ascend C 算子开发流程

```text
1. msopgen gen → 生成项目模板
2. 编写算子 .cpp + .h (Ascend C API)
3. msopgen compile → 编译为 .o
4. 注册到 OPP → 框架可调用
```

## 4. 一个简单的自定义 ReLU 算子

> [!NOTE]
> 以下为算子框架代码。完整的可编译示例（含数据搬运和计算逻辑）参见 [昇腾社区 Ascend C 示例](https://www.hiascend.com/document/detail/zh/canncommercial/800/operatordevelopment/ascendcdevg/ascendcdevg_0002.html)。

```cpp
// custom_relu.cpp
#include "kernel_operator.h"

class CustomReLU {
public:
    __aicore__ void operator()(GM_ADDR x, GM_ADDR y, uint32_t total_size) {
        // Ascend C 使用 LocalTensor + Pipe 模型
        LocalTensor<DT_FLOAT> inLocal;
        LocalTensor<DT_FLOAT> outLocal;
        // 数据搬运 + 计算...
    }
};
```

完整开发指南参考 Ascend C 官方文档。对于学习阶段，建议先从标准算子入手理解 NPU 行为，有需要时再深入算子开发。

## 5. 参考链接

- [CANN 文档 — Ascend C 编程](https://www.hiascend.com/document/detail/en/canncommercial/800/devguide/ascendc/ascendc_0001.html)
