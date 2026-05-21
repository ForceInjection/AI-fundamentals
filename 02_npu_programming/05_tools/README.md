# 昇腾工具链

昇腾生态的工具链与 CUDA 工具有清晰的对应关系。`npu-smi` 对标 `nvidia-smi`（轻量巡检），`ascend-dmi` 对标 `deviceQuery` + `bandwidthTest`（硬件诊断与性能基准），`atc` 对标 TensorRT（模型编译）。三者在排障流程中形成链条：`npu-smi` 发现问题 → `ascend-dmi` 诊断根因 → `atc` 转换部署。

## 1. [npu-smi 使用参考](01_npu_smi_reference.md)

不依赖 CANN toolkit，直接通过驱动获取状态。60+ 种查询类型，默认输出含设备概览和进程表。重点覆盖资源使用（`-t usages`）、健康与错误（ECC/PCIe）、HCCS 链路状态、拓扑查询。

## 2. [ascend-dmi 使用参考](02_ascend_dmi_reference.md)

对应 `deviceQuery` + `bandwidthTest`。通过 DCMI 和 AscendCL 提供设备详情、带宽/算力实测、版本兼容性检查和 12 项故障诊断。

## 3. [ATC 模型转换](03_atc_model_conversion.md)

PyTorch → ONNX → OM 的完整转换流程。覆盖关键参数（`--framework`、`--soc_version`、`--input_shape`）、转换后 OM 模型的 AscendCL 推理加载示例，以及 ONNX 到 OM 的尺寸缩减（约 50%）。
