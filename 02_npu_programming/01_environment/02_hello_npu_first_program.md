# 01 — Hello NPU: 华为昇腾初体验

## 1. 环境概览

| 项目      | 版本/型号                    |
| --------- | ---------------------------- |
| 服务器    | Ubuntu 22.04.4 LTS (aarch64) |
| NPU 型号  | Ascend 910B3 × 8             |
| CANN 版本 | 8.0.1 (runtime 7.6.0.2.220)  |
| Python    | 3.10.12                      |
| PyTorch   | 2.1.0                        |
| torch_npu | 2.1.0.post13                 |
| 虚拟环境  | `/root/npu-learning/venv`    |
| 指定设备  | NPU 7 (device id: 7)         |

---

## 2. 安装步骤

### 2.1 创建虚拟环境

```bash
# 安装 venv 支持（仅首次）
apt install -y python3.10-venv

# 创建虚拟环境
python3 -m venv /root/npu-learning/venv
```

### 2.2 加载 CANN 环境并安装依赖

**关键顺序**：必须先 source CANN 环境变量，再激活 venv。CANN 的 `set_env.sh` 会设置 `LD_LIBRARY_PATH`、`PYTHONPATH` 等，这些是 torch_npu 正常工作的前提。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /root/npu-learning/venv/bin/activate

# ARM 架构 (aarch64) — 不需要额外 index URL
pip install torch==2.1.0
pip install pyyaml setuptools 'numpy<2'
pip install torch-npu==2.1.0.post13
```

版本兼容性：torch_npu 的版本必须与 CANN 版本和 PyTorch 版本同时匹配。本环境使用 CANN 8.0.1，对应 torch 2.1.x + torch_npu 2.1.0.postX 系列。numpy 需要 <2 以兼容 PyTorch 2.1.0。

### 2.3 指定 NPU 设备

通过环境变量 `ASCEND_RT_VISIBLE_DEVICES` 控制可见设备（类似 CUDA 的 `CUDA_VISIBLE_DEVICES`）：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 python3 your_script.py
```

---

## 3. CUDA → NPU API 映射速查

从 CUDA 代码迁移到 NPU 时，API 的对应关系非常直接。几乎所有 `cuda` 替换为 `npu` 即可：

| 操作         | CUDA                            | NPU                            |
| ------------ | ------------------------------- | ------------------------------ |
| 检查设备可用 | `torch.cuda.is_available()`     | `torch.npu.is_available()`     |
| 设备数量     | `torch.cuda.device_count()`     | `torch.npu.device_count()`     |
| 张量迁移     | `tensor.cuda()`                 | `tensor.npu()`                 |
| 张量迁移     | `tensor.to('cuda')`             | `tensor.to('npu')`             |
| 设备同步     | `torch.cuda.synchronize()`      | `torch.npu.synchronize()`      |
| 内存查询     | `torch.cuda.memory_allocated()` | `torch.npu.memory_allocated()` |
| 空缓存       | `torch.cuda.empty_cache()`      | `torch.npu.empty_cache()`      |
| 设备名称     | `torch.cuda.get_device_name(i)` | `torch.npu.get_device_name(i)` |
| 当前设备     | `torch.cuda.current_device()`   | `torch.npu.current_device()`   |
| 设置设备     | `torch.cuda.set_device(i)`      | `torch.npu.set_device(i)`      |

关键差异：

- 驱动层 API 不同：NPU 使用 `torch_npu` 包作为 PyTorch 与 CANN 之间的适配层，需要在代码中 `import torch_npu` 来注册 NPU 后端
- 流 (Stream) API：`torch.cuda.Stream()` → `torch.npu.Stream()`
- AMP 自动混合精度：CUDA 用 `torch.cuda.amp`，NPU 用 `torch.npu.amp`

## 4. 第一个 NPU 程序

下面的最小示例验证 NPU 后端注册、设备查询与张量运算三件事：

```python
import torch
import torch_npu  # 必须导入以注册 NPU 后端

# 检查环境
print(f"NPU 可用: {torch.npu.is_available()}")   # True
print(f"设备名: {torch.npu.get_device_name(0)}")  # Ascend910B3

# 在 NPU 上创建张量
x = torch.randn(3, 3).npu()
y = torch.randn(3, 3).npu()
z = torch.matmul(x, y)

print(f"设备: {z.device}")  # npu:0
```

完整脚本见 `01_environment/hello_npu.py`。

在远端服务器上的运行方式：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /root/npu-learning/venv/bin/activate
ASCEND_RT_VISIBLE_DEVICES=7 python3 01_environment/hello_npu.py
```

---

## 5. 性能测试结果

在 NPU 7 上运行 4096×4096 矩阵乘法：

| 平台        | 耗时      | 加速比     |
| ----------- | --------- | ---------- |
| CPU (ARM)   | 351.01 ms | 1× (基准)  |
| NPU (910B3) | 2.13 ms   | **164.6×** |

计算结果误差在 10⁻⁴ 量级（最大误差 0.000656），可忽略。

> [!NOTE]
> 这仅是单次矩阵乘法的 micro-benchmark，不代表端到端模型训练的加速效果。实际模型训练的加速比受计算/通信比、算子适配度等多个因素影响。

---

## 6. 关键概念

### 6.1 ASCEND_RT_VISIBLE_DEVICES

与 `CUDA_VISIBLE_DEVICES` 行为一致，但值为物理 NPU 编号（0-7）。设置为 `7` 后，进程只能看到并使用 7 号 NPU，代码中的设备索引从 0 开始重新映射（即 `npu:0` 实际对应物理 NPU 7）。

### 6.2 torch_npu 导包

`import torch_npu` 的副作用是向 PyTorch 注册 `npu` 设备后端。如果不 import，`torch.npu.is_available()` 返回 `False`，`tensor.npu()` 会报错。这类似于早期 CUDA 在 PyTorch 中的注册方式。

在 torch_npu 2.5.1+ 版本中，官方说明“可以不用手动导包”，但仍建议显式 import 以确保注册。

### 6.3 内存模型

NPU 的内存分为 HBM (High Bandwidth Memory) 和 DDR。Ascend 910B3 每卡配备 64GB HBM（从 `npu-smi info` 可见 65536 MB ≈ 64 GB，完整用法见 `01_npu_smi_reference.md`）。本服务器的 910B3 仅使用 HBM，DDR 容量为 0（`npu-smi info -t usages` 中 `DDR Capacity(MB): 0`）。`torch.npu.memory_allocated()` 和 `memory_reserved()` 的行为与 CUDA 对应 API 一致：

- `allocated`：正在使用的张量内存
- `reserved`：PyTorch 缓存分配器保留的内存

### 6.4 CANN 环境变量加载顺序

必须先 `source set_env.sh` 再激活 venv。原因是 CANN 脚本将 HCCL、AscendCL 等库路径注入 `LD_LIBRARY_PATH`，torch_npu 在 import 时需要解析 `libhccl.so` 等动态库。如果顺序颠倒，venv 激活不会自动补上这些路径。

---

## 7. 下一步

环境已就绪。接下来深入了解 Ascend 910B3 硬件架构和 CANN 软件栈的全貌，建立与 CUDA 生态的结构化对照。

## 8. 参考链接

- [昇腾社区 — PyTorch 适配](https://gitee.com/ascend/pytorch)
- [CANN 商业版文档](https://www.hiascend.com/document)
- [昇腾社区官网](https://www.hiascend.com)
- [Ascend 910B 产品页](https://www.hiascend.com/hardware/ai-chip)
