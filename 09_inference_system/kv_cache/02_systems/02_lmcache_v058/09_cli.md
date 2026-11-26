# LMCache v0.5.8 · `lmcache` CLI

> 核对基线:LMCache v0.5.8(tag `651617ef`);写作日期 2026-11-25
> 本文所有路径均可在 `git show v0.5.8:<path>` 核对。

## 入口收敛

v0.5.8 的 `pyproject.toml` `[project.scripts]` 只剩一行:

```toml
lmcache="lmcache.cli.main:main"
```

v0.5.1 同一节有三行:`lmcache`、`lmcache_server = lmcache.v1.server.__main__:main`、`lmcache_controller = lmcache.v1.api_server.__main__:main`(注意是下划线命名)。后两个入口连同源码目录在 v0.5.8 树上都已删除(`lmcache/v1/server/` 与 `lmcache/v1/api_server/` 均不存在)。

`lmcache/cli/main.py` 的 `main()` 是唯一读 `sys.argv` 的地方:`register_commands`(`lmcache/cli/commands/__init__.py`)把命令注册到 subparsers,命令类继承 `lmcache/cli/commands/base.py` 的 `BaseCommand`;`bench` / `query` / `quota` / `tool` / `trace` 五个是 `CompositeCommand`,按约定自动发现同包子包里的 `BaseCommand` 子类——加子命令只需新建子包,不改注册代码。

一个安装细节:`server` 与 `bench` 依赖完整安装(含 CUDA 扩展);轻量包(`lmcache-cli`)下这些命令的参数注册被跳过,`execute` 时打印安装提示退出(`lmcache/cli/commands/server.py`)。

## 子命令总表

| 子命令        | 类                   | 说明                                                                                                      |
| ------------- | -------------------- | --------------------------------------------------------------------------------------------------------- |
| `server`      | `ServerCommand`      | 启动 MP cache server + HTTP 前端(见下节)                                                                  |
| `coordinator` | `CoordinatorCommand` | 启动 MP coordinator(HTTP);`--host` 默认 127.0.0.1,`--port` 默认 9300                                      |
| `bench`       | `BenchCommand`       | 性能基准,子命令:`engine` / `server` / `quality` / `l2` / `trace-analyze`                                  |
| `describe`    | `DescribeCommand`    | 查看运行中服务状态;目标 `kvcache` / `engine`,默认 URL 分别 http://localhost:8080 与 http://localhost:8000 |
| `ping`        | `PingCommand`        | 活性检查;`kvcache` 打 `/healthcheck`,`engine` 打 `/health`                                                |
| `kvcache`     | `KVCacheCommand`     | 子命令 `clear`(`--url` 必给),清空 L1 缓存                                                                 |
| `memory`      | `MemoryCommand`      | 启动 memory orchestrator(gRPC):一个共享内存 region 的分配与就绪权威(配合 DEVDAX L1 的 `shared` 对象)      |
| `mock`        | `MockCommand`        | 框架示例 / 自测,不连任何 server                                                                           |
| `query`       | `QueryCommand`       | 打一条推理请求并报指标;子命令 `coordinator` / `engine` / `kvcache`                                        |
| `quota`       | `QuotaCommand`       | per-cache_salt 配额管理;子命令 `list` / `get` / `set` / `delete`                                          |
| `tool`        | `ToolCommand`        | 离线分析;子命令 `cache-simulator` / `flamegraph` / `transfer-channel-benchmark`                           |
| `trace`       | `TraceCommand`       | `.lct` trace 文件分析;子命令 `info` / `replay` / `replay-events`                                          |

几个常用形态:

- `lmcache quota set <salt> --limit-gb <GB> --url http://localhost:8080`:salt 用 `_default` 表示匿名流量;对应 server 侧 `distributed/quota_manager.py` `QuotaManager`;
- `lmcache trace replay FILE`:逐条回放 `.lct` trace,`--verbose` 逐条打印,`--no-csv` 跳过 CSV 汇总、`--json` 追加 JSON 汇总,`--jsonl-out` 导出逐记录 JSON(`lmcache/cli/commands/trace/replay_command.py`);
- `lmcache bench quality` 分两段:`run` 把相同 prompt 发给 reference / candidate / control 各 arm 存盘,`analyze` 对答案打分出 per-question 行与结论——两个 arm 不必同时在线。

## server 子命令要点

`lmcache/cli/commands/server.py` 的 `add_arguments` 把六个参数组拼进一个 parser,执行时各自解析成 config 对象交给 `run_http_server`:

| 参数组来源                                            | 职责                  |
| ----------------------------------------------------- | --------------------- |
| `multiprocess/config.py` `add_mp_server_args`         | 请求传输与引擎对接    |
| 同文件 `add_p2p_args`                                 | P2P 传输              |
| `distributed/config.py` `add_storage_manager_args`    | L1 / L2 / 驱逐 / 策略 |
| 同文件 `add_http_frontend_args`                       | HTTP 前端             |
| `mp_observability/config.py` `add_observability_args` | OTel 观测             |
| 同文件 `add_coordinator_args`                         | 向 coordinator 注册   |

常用旗标(均为已核对的默认值):

| 旗标                          | 默认             | 说明                                                                                                                                    |
| ----------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `--host` / `--port`           | localhost / 5555 | ZMQ/gRPC 请求端口                                                                                                                       |
| `--transport`                 | `zmq`            | `zmq` 或 `grpc`;grpc 另有 `--grpc-server-workers`(32)、`--grpc-server-mode`(auto/sync/aio/native)                                       |
| `--http-host` / `--http-port` | 127.0.0.1 / 8080 | HTTP 前端                                                                                                                               |
| `--l1-size-gb`                | 无               | DRAM L1 容量;GDS 下是 slab 容量;legacy L1 路径与 `--eviction-policy` 一起必配                                                           |
| `--eviction-policy`           | 无               | `LRU` / `ARC` / `IsolatedLRU` / `noop`;配 `--eviction-trigger-watermark`(0.8)与 `--eviction-ratio`(0.2)                                 |
| `--l1-manager '<JSON>'`       | 无               | 多 L1 途径,type ∈ DRAM/DEVDAX/GDS/SHADOW(详见 [04_storage_l1_l2.md](04_storage_l1_l2.md))                                               |
| `--l2-adapter '<JSON>'`       | 无               | 可重复,每个 JSON 一个 adapter                                                                                                           |
| `--hash-algorithm`            | `blake3`         | builtin / sha256_cbor / blake3                                                                                                          |
| `--chunk-size`                | 256              | connector 可按模型 KV 几何向上协商                                                                                                      |
| `--engine-type`               | `default`        | `default` 或 `blend`                                                                                                                    |
| `--supported-transfer-mode`   | `lmcache_driven` | `lmcache_driven` / `engine_driven` / `auto`                                                                                             |
| `--coordinator-url` 等        | 无               | 注册 coordinator;旗标缺省时回落到 `LMCACHE_COORDINATOR_*` 环境变量                                                                      |
| `--api-key-file`              | 无               | 可重复;别名 `--http-api-key-file`,环境变量 `LMCACHE_API_KEY_FILE` / `LMCACHE_API_KEY`(`lmcache/security/api_key.py` `add_api_key_args`) |

## 与 v0.5.1 对照

| 项                | v0.5.1                                                                                                                                                                      | v0.5.8                                                                                                                          |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 可执行入口        | 3 个:`lmcache`、`lmcache_server`、`lmcache_controller`                                                                                                                      | 1 个:`lmcache`                                                                                                                  |
| legacy TCP server | `lmcache_server <host> <port> [storage]`,位置参数式(`v1/server/__main__.py` `main()`)                                                                                       | 入口与目录已删;由 `lmcache server`(旗标式,MP 架构)承接                                                                          |
| 控制面入口        | `lmcache_controller`(`v1/api_server/__main__.py`)                                                                                                                           | 入口与目录已删;由 `lmcache coordinator` 承接                                                                                    |
| 子命令集合        | server / coordinator / describe / kvcache / mock / ping / query / quota / bench(engine、l2、server)/ tool(cache-simulator、transfer-channel-benchmark)/ trace(info、replay) | 原有全保留;新增 `memory` 子命令;`bench` 新增 `quality` 与 `trace-analyze`;`tool` 新增 `flamegraph`;`trace` 新增 `replay-events` |

两点澄清,避免误读「收敛」:

- v0.5.1 就已经有 `lmcache server` 子命令(当时的 MP server,ZMQ + HTTP),所以消失的是 legacy 的独立入口 `lmcache_server`,不是整个 server 概念;
- 子命令面在这两版之间是扩张而非收缩,收敛只发生在 pyproject 入口层。

## 相关篇

- server 配置面与端点:[03_mp_server.md](03_mp_server.md)
- L1/L2 存储与 `--l1-manager` / `--l2-adapter` JSON:[04_storage_l1_l2.md](04_storage_l1_l2.md)
- coordinator 控制面:[01_mp_architecture.md](01_mp_architecture.md)
