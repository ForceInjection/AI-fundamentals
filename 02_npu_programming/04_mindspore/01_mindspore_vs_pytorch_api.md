# MindSpore 与 PyTorch API 对照

## 1. 模型定义

**PyTorch** (`nn.Module`):

```python
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)))
```

**MindSpore** (`nn.Cell`):

```python
import mindspore.nn as nn

class Bottleneck(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.relu = nn.ReLU()

    def construct(self, x):
        return self.relu(self.bn1(self.conv1(x)))
```

关键差异：

- PyTorch 的 `forward` → MindSpore 的 `construct`。
- PyTorch 的 `nn.Module` → MindSpore 的 `nn.Cell`。
- MindSpore 的 `nn.Conv2d` 默认无 bias（`has_bias=False`），PyTorch 默认有 bias。

## 2. 训练循环

**PyTorch**:

```python
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

**MindSpore**:

```python
# 定义前向+损失函数
def forward_fn(data, label):
    logits = net(data)
    loss = loss_fn(logits, label)
    return loss

# 获取梯度函数
grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
loss, grads = grad_fn(images, labels)
optimizer(grads)  # optimizer 作为函数调用
```

关键差异：

- MindSpore 使用函数式梯度：`ms.value_and_grad()` 返回梯度函数。
- 不需要显式 `zero_grad()`，梯度在每次 `value_and_grad` 调用时自动清零。
- `optimizer(grads)` 而不是 `optimizer.step()`。

## 3. 动态图 vs 静态图

| 特性         | PyTorch                                    | MindSpore                            |
| ------------ | ------------------------------------------ | ------------------------------------ |
| 动态图       | 默认 (eager)                               | `PYNATIVE_MODE`                      |
| 静态图       | `torch.compile()` (Python 字节码→FX Graph) | `GRAPH_MODE` (源码级 JIT 编译)       |
| 设置方式     | `torch.compile(model)`                     | `ms.set_context(mode=ms.GRAPH_MODE)` |
| 编译时机     | 运行期根据输入形状触发                     | 模型定义后首次执行时编译             |
| debug 友好度 | 高（用 `print`/`pdb`）                     | Graph 模式下不支持动态 `print`       |

## 4. 常用 API 对照速查

| 功能       | PyTorch                                      | MindSpore                                                      |
| ---------- | -------------------------------------------- | -------------------------------------------------------------- |
| 模块基类   | `nn.Module`                                  | `nn.Cell`                                                      |
| 前向方法   | `forward()`                                  | `construct()`                                                  |
| 卷积       | `nn.Conv2d(c_in, c_out, k, stride, padding)` | `nn.Conv2d(c_in, c_out, k, stride, pad_mode='pad', padding=p)` |
| 批归一化   | `nn.BatchNorm2d(c)`                          | `nn.BatchNorm2d(c)`                                            |
| 激活函数   | `nn.ReLU(inplace=True)`                      | `nn.ReLU()` (无 inplace)                                       |
| 最大池化   | `nn.MaxPool2d(k, s, p)`                      | `nn.MaxPool2d(k, s, pad_mode='pad', padding=p)`                |
| 全连接     | `nn.Linear(in_f, out_f)`                     | `nn.Dense(in_f, out_f)`                                        |
| Flatten    | `nn.Flatten()`                               | `nn.Flatten()`                                                 |
| 序列容器   | `nn.Sequential(*layers)`                     | `nn.SequentialCell(layers)`                                    |
| 优化器     | `optim.SGD(params, lr, momentum)`            | `nn.SGD(params, learning_rate, momentum)`                      |
| 交叉熵     | `nn.CrossEntropyLoss()`                      | `nn.SoftmaxCrossEntropyWithLogits(sparse=True)`                |
| 梯度计算   | `loss.backward()`                            | `ms.value_and_grad(forward_fn)(...)`                           |
| 设备同步   | `torch.cuda.synchronize()`                   | `ms.hal.synchronize()`                                         |
| 设置上下文 | `torch.cuda.set_device(i)`                   | `ms.set_context(device_target='Ascend', device_id=i)`          |
| 张量创建   | `torch.randn(3, 4)`                          | `ms.Tensor(np.random.randn(3, 4).astype(np.float32))`          |
| 张量到设备 | `tensor.to('cuda')`                          | (上下文设置后自动)                                             |

## 5. 参考链接

- [MindSpore API 文档](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- [MindSpore 编程指南](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
