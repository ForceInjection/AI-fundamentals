# NCCL Benchmark 测试套件

这是一个完善的测试套件，用于验证 `nccl_benchmark.sh` 脚本的功能和性能。

## 1. 简介

本测试套件包含多个专门的测试脚本，覆盖了从基础语法、配置管理、Mock 环境到性能基准的各个方面。

### 1.1 文件结构

```bash
test/
├── README.md                          # 本文档
├── run_all_tests.sh                   # 主测试运行器
├── nccl_benchmark_mock.sh             # Mock 脚本（支持多种测试场景）
├── test_*.sh                          # 各个功能测试脚本
├── mock/                              # Mock 支持文件目录
└── results/                           # 测试结果目录（自动生成）
```

---

## 2. 快速开始

### 2.1 常用命令

| 目标             | 命令                                |
| ---------------- | ----------------------------------- |
| **运行所有测试** | `./run_all_tests.sh`                |
| **运行快速测试** | `./run_all_tests.sh --quick`        |
| **运行特定套件** | `./run_all_tests.sh --suite <name>` |
| **列出所有套件** | `./run_all_tests.sh --list`         |

### 2.2 测试套件列表

| 套件名称       | 描述                    | 运行时间   |
| -------------- | ----------------------- | ---------- |
| `syntax`       | 语法和基础功能验证      | ~1 分钟    |
| `config`       | 配置管理器功能测试      | ~1-2 分钟  |
| `mock`         | Mock 环境功能测试       | ~2-3 分钟  |
| `pxn`          | PXN 模式功能测试        | ~30 秒     |
| `nvlink`       | NVLink 计数测试         | ~30 秒     |
| `dns`          | DNS 解析逻辑测试        | ~1 分钟    |
| `optimization` | 优化级别功能测试        | ~1 分钟    |
| `network-fix`  | 网络配置修复验证测试    | ~30 秒     |
| `performance`  | 性能基准测试            | ~5-10 分钟 |
| `integration`  | 集成测试 (多种网络后端) | ~2 分钟    |

---

## 3. Mock 系统与跨平台支持

### 3.1 跨平台支持

本测试套件完美支持 Linux 和 macOS 平台：

- **Linux (原生)**: 无需 Mock 即可运行所有脚本，支持调用真实的硬件工具 (`nvidia-smi`, `ibv_devinfo`)。
- **macOS (Mock)**: 自动启用 Mock 模式，通过模拟系统命令来验证逻辑，无需真实 GPU。

### 3.2 Mock 场景

在 macOS 或无 GPU 的 Linux 环境下，可以使用 Mock 场景进行测试：

| 场景               | 描述                   |
| ------------------ | ---------------------- |
| `single_gpu`       | 单 GPU 环境            |
| `multi_gpu_nvlink` | 多 GPU + NVLink 环境   |
| `multi_gpu_pcie`   | 多 GPU + PCIe 环境     |
| `cluster_ib`       | 集群 + InfiniBand 环境 |

**使用示例:**

```bash
# 使用特定场景运行
./run_all_tests.sh --mock-scenario=multi_gpu_nvlink

# 结合特定测试套件
./run_all_tests.sh --suite pxn --mock-scenario=cluster_ib
```

---

## 4. 高级用法

### 4.1 环境变量

- `TEST_RESULTS_DIR`: 自定义测试结果目录
- `VERBOSE_OUTPUT`: 启用详细输出 (设置为 1)
- `TEST_TIMEOUT`: 设置测试超时时间 (秒)

### 4.2 命令行选项

```bash
./run_all_tests.sh [选项]

选项:
  --verbose               详细输出模式
  --quick                 快速模式 (跳过耗时的性能测试)
  --all                   运行所有测试 (默认)
  --performance           仅运行性能测试
  --integration           仅运行集成测试
  --suite NAME            运行指定测试套件
  --mock-scenario NAME    指定 Mock 场景
  --no-mock               禁用 Mock 模式
  --list                  列出可用套件
  --help                  显示帮助
```

---

## 5. 故障排除

### 5.1 常见问题

1. **权限错误**: 运行 `chmod +x *.sh` 确保脚本可执行。
2. **Python 依赖**: 确保已安装 `torch` (Mock 模式下会自动模拟)。
3. **Mock 失败**: 检查 `mock/` 目录下的辅助脚本是否完整。

### 5.2 调试

```bash
# 启用调试输出运行
bash -x ./run_all_tests.sh --suite mock
```

---
