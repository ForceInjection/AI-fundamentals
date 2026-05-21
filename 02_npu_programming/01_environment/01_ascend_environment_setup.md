# 昇腾环境搭建

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

版本兼容性：torch_npu 的版本必须与 CANN 版本和 PyTorch 版本同时匹配。CANN 8.0.1 对应 torch 2.1.x + torch_npu 2.1.0.postX 系列。numpy 需要 <2 以兼容 PyTorch 2.1.0。

### 2.3 指定 NPU 设备

通过环境变量 `ASCEND_RT_VISIBLE_DEVICES` 控制可见设备（类似 CUDA 的 `CUDA_VISIBLE_DEVICES`）：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 python3 your_script.py
```

## 3. CANN TBE 依赖

CANN 的 Tensor Boost Engine (TBE) 依赖于以下 Python 包，在虚拟环境中必须完整安装：

```bash
pip install attrs cloudpickle decorator ml-dtypes psutil scipy tornado
```

缺少任何一个都会导致 GE (Graph Engine) 初始化失败：

```text
ModuleNotFoundError: No module named 'attr'
GELib::InnerInitialize failed
GEInitialize failed
```

这是因为即使使用 PyTorch NPU，底层 CANN 的图编译器仍需要 TBE 的 Python 环境才能将计算图编译为 Ascend 可执行指令。

## 4. CANN 环境变量加载顺序

必须先 `source set_env.sh` 再激活 venv。原因是 CANN 脚本将 HCCL、AscendCL 等库路径注入 `LD_LIBRARY_PATH`，torch_npu 在 import 时需要解析 `libhccl.so` 等动态库。如果顺序颠倒，venv 激活不会自动补上这些路径。

## 5. 环境验证

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /root/npu-learning/venv/bin/activate
ASCEND_RT_VISIBLE_DEVICES=7 python3 -c "
import torch
import torch_npu
print(f'PyTorch {torch.__version__}, torch_npu {torch_npu.__version__}')
print(f'NPU available: {torch.npu.is_available()}')
print(f'Device: {torch.npu.get_device_name(0)}')
"
```

预期输出：`NPU available: True`，`Device: Ascend910B3`。

## 6. 参考链接

- [昇腾社区 — PyTorch 适配](https://gitee.com/ascend/pytorch)
- [CANN 商业版文档](https://www.hiascend.com/document)
