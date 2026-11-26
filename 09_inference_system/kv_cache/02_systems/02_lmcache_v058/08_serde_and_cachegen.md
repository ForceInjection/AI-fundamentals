# LMCache v0.5.8 · 序列化选项与 CacheGen 外移

> 核对基线：LMCache v0.5.8（tag `651617ef`）；写作日期 2026-11-25
> 本文所有路径均可在 `git show v0.5.8:<path>` 核对。

## serde 机制：注册表 + L2 adapter 的 `serde` 字段

v0.5.8 的序列化挂在 **L2 adapter** 上，不再有 v0.5.1 「remote connector 配 serde」的概念。三个部件：

| 部件       | 位置                                                                                                                                      | 职责                                                                            |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 工厂注册表 | `lmcache/v1/distributed/serde/factory.py`（`register_serde_factory` / `create_serde_processor` / `get_registered_serde_types`）           | serde 类型按名注册；各实现在自己模块尾部调 `register_serde_factory`             |
| 配置解析   | `lmcache/v1/distributed/l2_adapters/config.py`（`L2AdapterConfigBase._parse_serde_config` → `serde_config` 字段）                         | adapter JSON spec 里可选的 `"serde"` 子字典                                     |
| 运行时包装 | `lmcache/v1/distributed/l2_adapters/serde_wrapper.py`（`SerdeL2AdapterWrapper`，在 `distributed/storage_manager.py` 组装 adapter 时套上） | store 前序列化、load 后反序列化；lookup/unlock/delete/eviction 直通内层 adapter |

`serde/base.py` 定义接口：`Serializer` / `Deserializer`（sync 原语，签名都带 `ObjectKey`）、`SerdeProcessor`（事件 fd + submit/query 的异步任务面）、`SerdeConfig`。`async_processor.py` `AsyncSerdeProcessor` 把 sync 原语包成带线程池的任务处理器；`multi.py` `MultiSerializer` / `MultiDeserializer` / `single_to_multi_serializer` 处理多张量组。

adapter JSON 里的写法（factory.py docstring 原例）：

```json
{
  "type": "fs",
  "base_path": "/cache",
  "serde": { "type": "fp8", "fp8_dtype": "float8_e4m3fn" }
}
```

包装层的语义边界（serde_wrapper.py docstring）：临时字节 buffer 从注入的 L1Manager 分配，和普通临时 KV buffer 一样计入 L1 账目；失败按「全部或 nothing」处理——部分 temp 分配失败即整个任务失败，保持 `L2AdapterInterface` 粗粒度成功语义，调用方的锁与生命周期约束不变。

## serde 选项表

factory 注册表里按名可用的 serde 有 **三种**：

| 注册名       | 实现                                                                               | 关键 kwargs                                                                                                                                                             | 用途                                                                                                                                                                                                         |
| ------------ | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `fp8`        | `serde/fp8.py` `Fp8QuantizationSerializer` / `Fp8QuantizationDeserializer`         | `fp8_dtype`（默认 `float8_e4m3fn`）、`max_workers`（默认 1）                                                                                                            | KV 张量整体 cast 到 fp8 存 L2，恰好 1 字节/元素，load 时 cast 回原 dtype                                                                                                                                     |
| `turboquant` | `serde/turboquant/turboquant.py` `TurboQuantSerializer` / `TurboQuantDeserializer` | 经 `TurboQuantSerdeConfig`：`preset`（默认 `turboquant_k8v4`）、`head_dim`（128）、`block_size`（16）、`skip_first_layers`（2）、`skip_last_layers`（2）、`cuda_device` | 低比特 KV 压缩，首尾若干层跳过量化为原样存储；kernel 在 `turboquant/store_kernel.py` / `decode_kernel.py`                                                                                                    |
| `aesgcm`     | `serde/aesgcm.py` `AesGcmSerializer` / `AesGcmDeserializer`                        | `key_provider`（默认 `hkdf`）、`aes_bits`（默认 128）、`master_key_path`、`max_workers`                                                                                 | L2 静态数据加密（docstring：「AES-GCM encryption serde for L2-at-rest confidentiality」）；密钥经 `serde/key_provider.py` `KeyProvider` / `HkdfKeyProvider`，`ObjectKey` identity 作 AES-GCM associated data |

第四个实现 `serde/asym_k16_v8.py`（`AsymK16V8MultiSerializer` / `AsymK16V8MultiDeserializer`，`MultiSerializer` 子类）基于 `lmcache/v1/kv_codec/asym_k16_v8.py` 的 `AsymK16V8Codec` 做 K16/V8 非对称编码，含拆分层场景的 V-only 变体，测试在 `tests/v1/distributed/serde/test_asym_k16_v8_multi.py` 与 `test_asym_k16_v8_v_only.py`。但它 **没有注册进 factory**：模块内无 `register_serde_factory` 调用，`serde/__init__.py` 也不导出它——v0.5.8 的 `"serde": {"type": ...}` 配置还选不到它，属于已实现未接线。

## CacheGen：主仓不内置

官方口径（`docs/source/mp/cachegen.rst` 全文）：

> CacheGen KV cache compression for multiprocess mode is **not implemented yet** (coming soon). The original implementation is maintained in [LMCache_inprocess](https://github.com/LMCache/LMCache_inprocess).

v0.5.8 主仓残留两类与 CacheGen 相关的代码：

| 残留                      | 位置                                                                                                              | 状态                                                                                                                                                                                    |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 算术编码 CUDA kernel 头   | `csrc/cuda/cachegen_kernels.cuh`（另有 `csrc/musa/cachegen_kernels_musa.h`、`csrc/sycl/cachegen_kernels_sycl.h`） | 仍被 `csrc/cuda/ac_enc.cu` / `ac_dec.cu` / `pybind.cpp` 引用编译                                                                                                                        |
| kernel 的 Python 等价包装 | `lmcache/v1/platform/torch_ops/cachegen_kernels.py`                                                               | numba 实现：`encode_fast_new` / `decode_fast_new` / `decode_fast_prefsum` / `calculate_cdf`，docstring 自述「Python equivalent of C++ …」；经 `platform/torch_ops/__init__.py` 星号导出 |

主仓内 **没有** 调用这些 kernel 的 CacheGen 编解码器：v0.5.1 的 `lmcache/v1/storage_backend/naive_serde/cachegen_encoder.py`（`CacheGenSerializer`）、`cachegen_decoder.py`（`CacheGenDeserializer`）、`cachegen_basics.py` 与 v0 代的 `lmcache/storage_backend/serde/cachegen_*` 全部不在 v0.5.8 树上。`LMCache_inprocess` 外部仓在 `docs/source/getting_started/quickstart.rst` 等多处被指为 in-process 包的维护地。

## 与 v0.5.1 配置面的对照

| v0.5.1                                                                                                                                                                           | v0.5.8                                                     |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `LMCacheEngineConfig` 的 `remote_serde` 键（`lmcache/v1/config.py`，默认 `"naive"`，可选 `"cachegen"`/`"kivi"`）                                                                 | 键已移除，`lmcache/` 内无 `remote_serde` 残留              |
| `LMCACHE_REMOTE_SERDE` 环境变量（`docs/source/api_reference/configurations.rst` 有收录，redis.rst 示例 `export LMCACHE_REMOTE_SERDE="naive" # "naive" (default) or "cachegen"`） | 已移除                                                     |
| `naive_serde/__init__.py` `CreateSerde`（naive/kivi/cachegen 三选一，接在 `remote_backend.py` 上）                                                                               | 无对应物；serde 选择改走 L2 adapter JSON 的 `"serde"` 字段 |

## 与旧版文档的关系

[../01_lmcache/15_cachegen.md](../01_lmcache/15_cachegen.md) 的 §1（论文与设计原理：SIGCOMM '24、自适应量化 + 流式算术编码）**可原样引用，不受主仓版本影响**——实现外移到 `LMCache_inprocess` 后，论文叙述仍然成立；该篇后续的源码章节（`naive_serde/cachegen_*` 路径）只对 v0.5.1 树有效，v0.5.8 部署不适用。官方文档里 MP 模式的 CacheGen 等一个「coming soon」。

## 相关篇

- 本目录总览与旧新映射表：[README.md](README.md)
- v0.5.1 CacheGen 论文与源码详解（档案）：[../01_lmcache/15_cachegen.md](../01_lmcache/15_cachegen.md)
- server 侧 adapter 组装入口：[03_mp_server.md](03_mp_server.md)
