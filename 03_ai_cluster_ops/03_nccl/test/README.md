# NCCL Benchmark 脚本单元测试

> **定位**：这是 `nccl_benchmark.sh` 脚本的单元测试，验证参数解析、配置注入、网络后端选择等脚本层面的正确性。**不验证 NCCL 通信性能**——真实带宽测试见 [通信路径逐层压测](../06_nccl_path_benchmark.md) 和 [基准测试方法论](../04_nccl_benchmark.md)。

## 快速开始

```bash
./run_all_tests.sh                # 全部 5 个套件
./run_all_tests.sh --suite args   # 单个套件
./run_all_tests.sh --list         # 列出可用套件
./run_all_tests.sh --verbose      # 详细输出
```

无外部依赖，macOS 和无 GPU 环境均可运行。

## 测试套件

| 套件           | 脚本                          | 测试数 | 内容                                                             |
| -------------- | ----------------------------- | ------ | ---------------------------------------------------------------- |
| `syntax`       | `test_syntax_basic.sh`        | 1      | `bash -n` 静态语法检查                                           |
| `args`         | `test_arg_parsing.sh`         | 11     | `--size` / `--time` / `--network` / `--dry-run` 参数解析         |
| `config`       | `test_config_manager.sh`      | 20     | `set_nccl_config` / `setup_network_config` / `cache_system_info` |
| `optimization` | `test_optimization_levels.sh` | 8      | conservative / balanced / aggressive 三种优化级别                |
| `pxn`          | `test_pxn_mode.sh`            | 7      | PXN 模式参数和配置                                               |

## 文件结构

```text
test/
├── README.md
├── run_all_tests.sh              # 测试运行器
├── test_syntax_basic.sh          # 语法检查
├── test_arg_parsing.sh           # 参数解析
├── test_config_manager.sh        # 配置管理器（函数内联，无外部依赖）
├── test_optimization_levels.sh   # 优化级别
└── test_pxn_mode.sh              # PXN 模式
```
