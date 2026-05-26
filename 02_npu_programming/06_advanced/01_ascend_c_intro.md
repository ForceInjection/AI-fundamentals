# 05 — 进阶主题: 算子开发、模型转换与迁移决策

## 1. Ascend C 算子开发入门

### 1.1 什么时候需要自定义算子

大多数 PyTorch 标准算子（Conv、BN、Attention 等）已有 CANN 内置优化实现。以下场景需要自定义算子：

- 模型中使用了自定义 CUDA kernel（如 FlashAttention 变体、特殊的激活函数）
- 需要将多个小算子融合为一个来减少内存访问
- 使用非标准数值计算（如自定义量化格式）

### 1.2 算子开发工具链

CANN 提供的开发工具：

| 工具                       | 用途                             |
| -------------------------- | -------------------------------- |
| `msopgen`                  | 自动生成算子项目模板             |
| Ascend C Compiler (`ccec`) | 编译 Ascend C 算子代码为 .o 文件 |
| `opc`                      | 算子编译工具                     |

### 1.3 Ascend C 算子开发流程

```text
1. msopgen gen → 生成项目模板
2. 编写算子 .cpp + .h (Ascend C API)
3. msopgen compile → 编译为 .o
4. 注册到 OPP → 框架可调用
```

#### 1.3.1 一个简单的自定义 ReLU 算子 (Ascend C)

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

---

## 2. 模型转换: PyTorch → ONNX → OM

### 2.1 ATC (Ascend Tensor Compiler)

`atc` 将训练好的模型转换为昇腾离线模型（.om 文件），用于高性能推理部署。

**支持框架**：Caffe、TensorFlow、MindSpore、**ONNX**

### 2.2 转换流程

```bash
# Step 1: PyTorch 导出 ONNX
python3 -c "
import torch
model = torch.load('resnet50.pth')
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'resnet50.onnx',
                  input_names=['input'],
                  output_names=['output'],
                  opset_version=11)
"

# Step 2: ATC ONNX → OM
atc --model=resnet50.onnx \
    --framework=5 \
    --output=resnet50_910B3 \
    --soc_version=Ascend910B3 \
    --input_shape="input:1,3,224,224" \
    --input_format=NCHW
```

### 2.3 关键参数

| 参数             | 说明                                            |
| ---------------- | ----------------------------------------------- |
| `--framework`    | 0=Caffe, 1=MindSpore, 3=TF, **5=ONNX**          |
| `--soc_version`  | `Ascend910B3` (训练卡) / `Ascend310P3` (推理卡) |
| `--output`       | 输出的 .om 文件名（无后缀）                     |
| `--input_shape`  | 如 `"input:1,3,224,224"`                        |
| `--input_format` | NCHW / NHWC / ND                                |

转换后的 `.om` 文件可通过 AscendCL API 加载并执行推理。

### 2.4 OM 模型推理示例 (Python)

```python
import acl

# 1. 初始化 ACL
acl.init()

# 2. 加载 OM 模型
model_id, model_desc = acl.mdl.load_from_file("resnet50_910B3.om")

# 3. 创建输入/输出 dataset
input_dataset = acl.mdl.create_dataset()
output_dataset = acl.mdl.create_dataset()

# 4. 执行推理
acl.mdl.execute(model_id, input_dataset, output_dataset)

# 5. 清理
acl.mdl.unload(model_id)
acl.finalize()
```

> 注意：直接用 `acl` API 的推理代码较底层。通常推理会通过框架或服务化框架（如 MindSpore Lite、TensorRT-LLM-like 的服务层）来实现。

---

## 3. NPU 应用场景总结

### 3.1 训练场景

| 场景                              | 适合度        | 说明                                                          |
| --------------------------------- | ------------- | ------------------------------------------------------------- |
| **CV 模型训练** (ResNet/ViT/YOLO) | 优秀          | 标准算子覆盖完整，AMP 加速显著                                |
| **LLM 预训练**                    | 良好 (需适配) | 需使用 MindSpore + MindFormer 或 PyTorch NPU 适配的分布式方案 |
| **LLM SFT/LoRA**                  | 良好          | 计算量适中，单卡 64GB HBM 可容纳 13B 模型的 LoRA 训练         |
| **推荐模型** (DLRM 等)            | 良好          | Embedding 查表 + MLP 的负载模式适合 NPU                       |
| **强化学习**                      | 一般          | 非标准的计算模式，需较多自定义算子                            |
| **科学计算**                      | 一般          | 缺少类似 CUDA 的通用并行编程生态                              |

### 3.2 推理场景

| 场景             | 适合度        | 说明                                          |
| ---------------- | ------------- | --------------------------------------------- |
| **CV 推理**      | 优秀          | ATC 转换 + AscendCL，延迟和吞吐表现好         |
| **LLM 推理**     | 良好 (发展中) | 需使用 MindSpore Lite 或 vLLM-Ascend 适配版   |
| **边缘推理**     | 优秀          | Ascend 310 系列 + MindSpore Lite 专为边缘优化 |
| **实时视频分析** | 优秀          | DVPP 硬件解码 + AI Core 推理的 pipeline       |

### 3.3 HBM 容量优势

64 GB HBM 是 910B3 的突出优势（vs A100 80GB 版本的 80GB，但比 40GB 版本大一倍）。这使得：

- 单卡可训练更大 batch size
- 7B-13B 模型的 LoRA 微调可行
- 模型并行时切分更少分片，通信开销更低

---

## 4. GPU → NPU 迁移决策树

```text
需要迁移到 NPU 吗?
│
├─ 使用标准 PyTorch 模型 (ResNet/BERT/ViT/Llama)?
│   YES → 迁移成本低。替换 .cuda() → .npu()，验证精度即可
│
├─ 有自定义 CUDA kernel?
│   YES → 需评估 kernel 复杂度:
│   │
│   ├─ 简单 kernel (element-wise, custom activation)
│   │   → 用 Ascend C 重写，成本中等 (1-2 周)
│   │
│   └─ 复杂 kernel (FlashAttention, fused ops)
│       → 检查是否已有社区 NPU 适配 (如 torch_npu 的 fused operators)
│       否则成本高 (4-8 周)，考虑替代方案
│
├─ 依赖 NVIDIA 特有库 (cuDNN/cuBLAS/NCCL)?
│   → CANN 有对应替代，但 API 不同:
│   cuDNN/cuBLAS → CANN OPP (内置算子)
│   NCCL → HCCL (API 类似)
│   TensorRT → ATC
│
├─ 使用 torch.compile / Dynamo / Triton?
│   → NPU 不直接支持 Dynamo/Triton
│   Triton kernel 需改写为 Ascend C
│   torch.compile 替换为 MindSpore Graph 模式 或 CANN GE
│
└─ 训练框架是 PyTorch Lightning / HF Trainer?
    → 需确认这些框架的 Ascend 适配状态
    PyTorch Lightning: 有限支持
    HuggingFace Trainer: 通过 accelerate + torch_npu 基本可用
```

### 4.1 迁移成本估算

| 场景                               | 代码改动                | 时间   |
| ---------------------------------- | ----------------------- | ------ |
| 标准 CV 模型训练                   | 改设备字符串 (< 5 行)   | < 1 天 |
| 标准 NLP 模型训练                  | 改设备字符串 + 调试环境 | 1-3 天 |
| 有自定义 CUDA kernel (简单)        | 重写 kernel (Ascend C)  | 1-2 周 |
| 有自定义 CUDA kernel (复杂)        | 重写 + 调优             | 1-2 月 |
| 完整训练 pipeline (数据+训练+部署) | 框架适配 + 工具链切换   | 2-4 周 |

### 4.2 不建议迁移的场景

- 严重依赖 NVIDIA 专有生态（如 cuQuantum、CV-CUDA 的特殊功能）
- 使用大量 Triton kernel 且无社区替代方案
- 模型频繁变更且算子需求不稳定（研发早期）
- 对延迟要求极高（微秒级）的推理场景（需评估 ATC 转换后的延迟）

---

## 5. 学习路径建议

基于本系列文档的覆盖内容，推荐的进阶学习顺序：

```text
1. Hello NPU (环境+基础)
    ↓
2. 架构认知 (硬件+软件栈)
    ↓
3. PyTorch NPU 实战 (主要开发方式)
    ↓
4. MindSpore (备选框架)
    ↓
5. 进阶主题 (本节)
    ├→ 需要自定义算子 → Ascend C 开发
    ├→ 需要推理部署 → ATC + AscendCL
    └→ 需要评估迁移 → 决策树 + 迁移成本估算
```

## 6. 参考链接

- [CANN 文档 — ATC 工具](https://www.hiascend.com/document/detail/en/canncommercial/800/devtool/atc/atc_0001.html)
- [CANN 文档 — Ascend C 编程](https://www.hiascend.com/document/detail/en/canncommercial/800/devguide/ascendc/ascendc_0001.html)
- [CANN 文档 — AscendCL 应用开发](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/appdevgapi/aclpythondevg_0001.html)
- [昇腾社区 — 迁移工具](https://www.hiascend.com/document/detail/en/canncommercial/800/devguide/migrate/migrate_0001.html)
- [ONNX 官方文档](https://onnx.ai/onnx/)
