# ATC 模型转换

## 1. ATC (Ascend Tensor Compiler)

`atc` 将训练好的模型转换为昇腾离线模型（.om 文件），用于高性能推理部署。

**支持框架**：Caffe (0)、MindSpore (1)、TensorFlow (3)、**ONNX (5)**。

## 2. 转换流程

```bash
# Step 1: PyTorch 导出 ONNX
python3 -c "
import torch
import torchvision.models as models
model = models.resnet50(weights=None)
# 如加载已训练权重: model.load_state_dict(torch.load('resnet50.pth', weights_only=True))
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'resnet50.onnx',
                  input_names=['input'],
                  output_names=['output'],
                  opset_version=17)  # 建议 >=17, 避免 aten::adaptive_avg_pool2d 等算子 fallback
# 如需可变 batch size, 添加 dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
"

# Step 2: ATC ONNX → OM
atc --model=resnet50.onnx \
    --framework=5 \
    --output=resnet50_910B3 \
    --soc_version=Ascend910B3 \
    --input_shape="input:1,3,224,224" \
    --input_format=NCHW
```

实测：ResNet-50 随机权重 ONNX 98M → OM 50M（图编译优化和算子融合后的结果）。

## 3. 关键参数

| 参数             | 说明                                             |
| ---------------- | ------------------------------------------------ |
| `--framework`    | 0=Caffe, 1=MindSpore, 3=TF, **5=ONNX**           |
| `--soc_version`  | `Ascend910B3`（训练卡）/ `Ascend310P3`（推理卡） |
| `--output`       | 输出的 .om 文件名（无后缀）                      |
| `--input_shape`  | 如 `"input:1,3,224,224"`                         |
| `--input_format` | NCHW / NHWC / ND                                 |

## 4. OM 模型推理示例 (AscendCL)

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

> [!NOTE]
> 直接用 `acl` API 的推理代码较底层。通常推理会通过框架或服务化框架（如 MindSpore Lite、vLLM-Ascend 适配版）来实现。

## 5. 参考链接

- [CANN 文档 — ATC 工具](https://www.hiascend.com/document/detail/en/canncommercial/800/devtool/atc/atc_0001.html)
- [ONNX 官方文档](https://onnx.ai/onnx/)
