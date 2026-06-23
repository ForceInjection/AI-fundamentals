# GPU 共享调度——MIG、MPS、Time-slicing 在 K8s 中的表达

不是所有 GPU 作业都需要独占一张卡。推理服务、Jupyter Notebook、小规模实验——这些场景下，一张 A100 跑 8 个推理实例比 8 张 A100 各跑 1 个经济得多。问题是在 K8s 中怎么把「共享」表达为调度器能理解的资源模型。

三种共享方式——MIG（硬切分）、MPS（计算共享）、Time-slicing（时分复用）——对调度器来说是三种完全不同的资源抽象。

---

## 一、MIG：显存和 SM 的硬切分

MIG 将一张物理 GPU 切分为多个独立的 GPU 实例，每个实例有专属的显存、SM 和带宽。A100-80GB 可以切为 7 个 10GB 实例（`1g.10gb` × 7）或 2 个 40GB 实例（`2g.20gb` × 2）等多种组合。

对 K8s 调度器来说，每个 MIG 实例暴露为独立的 `nvidia.com/mig-<profile>` 资源——不是 `nvidia.com/gpu`，而是更细粒度的资源类型：

```yaml
# MIG 切分后的资源类型
nvidia.com/mig-1g.10gb      # 1 个 GPU 计算切片 + 10GB 显存
nvidia.com/mig-2g.20gb      # 2 个 GPU 计算切片 + 20GB 显存
nvidia.com/mig-3g.40gb      # 3 个 GPU 计算切片 + 40GB 显存

# Pod 请求一个 MIG 实例
apiVersion: v1
kind: Pod
spec:
  containers:
  - resources:
      limits:
        nvidia.com/mig-1g.10gb: 1    # 一个 10GB MIG 实例
```

**调度器的视角**：MIG 切分后，一张物理 GPU 对调度器来说变成了 7 个可调度单元。每个 MIG 实例的显存和计算资源被硬隔离——一个实例 OOM 不会影响同一张 GPU 上的其他实例。代价是 MIG 与 P2P 互斥：启用了 MIG 的 GPU 不能再做 MPS 或跨 GPU 的 NVLink P2P 通信。

MIG 配置需要在节点层面预先设置（`nvidia-smi mig -cgi`），调度器无法动态切分——它只能调度到已切好的实例上。因此 MIG 适合推理场景中 GPU 资源规划相对固定的情况。

---

## 二、MPS：计算资源共享，显存不隔离

MPS 让多个进程的 CUDA kernel 在同一张 GPU 上真正并行执行，而不是 Default 模式下的时间片切换。这与 MIG 的关键区别：

| 维度         | MIG                                 | MPS                                   |
| ------------ | ----------------------------------- | ------------------------------------- |
| 显存         | 硬隔离，每个实例有固定显存上限      | 共享，一个进程 OOM 影响整张 GPU       |
| 计算资源     | 硬隔离，每个实例有固定 SM 配额      | 共享，CUDA kernel 并行执行            |
| P2P          | 不支持                              | 支持（但 MPS 进程共享同一 GPU）       |
| K8s 资源模型 | 独立的资源类型 (`nvidia.com/mig-*`) | 不改变资源模型，调度器不知道 MPS 存在 |

对 K8s 调度器来说，MPS **不改变资源模型**——Pod 仍然请求 `nvidia.com/gpu: 1`，调度器不知道多个 Pod 被 MPS 调度到同一张 GPU 上。这意味着显存超分是 MPS 在 K8s 中的核心挑战：

```text
GPU 有 80GB 显存
  Pod A 请求 nvidia.com/gpu: 1 → 调度器看到 1/1 GPU，调度成功 → 实际占用 50GB
  Pod B 请求 nvidia.com/gpu: 1 → 调度器看到 1/1 GPU，已经满了
  （但 Pod A 声明了 1 GPU，默认调度器认为这张 GPU 已经被占用了）
```

解决这个矛盾需要 Device Plugin 支持 GPU 共享策略——将 `nvidia.com/gpu` 同时暴露为可共享资源，或在 Device Plugin 层面做虚拟化（如 `nvidia.com/gpu.shared: 1`，表示请求 GPU 计算时间片而非独占整卡）。NVIDIA GPU Operator 的 Time-slicing 配置就是这一思路。

---

## 三、Time-slicing：时分复用

NVIDIA GPU Operator 支持 Time-slicing，在 Device Plugin 层面将一张物理 GPU 虚拟化为多个逻辑 GPU：

```yaml
# GPU Operator 的 time-slicing 配置
# 将一张物理 GPU 暴露为 4 个逻辑 GPU
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
data:
  any: |-
    version: v1
    flags:
      migStrategy: none
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4          # 每张物理 GPU → 4 个逻辑 GPU
```

配置后，K8s 看到的 GPU 资源从 `8 × nvidia.com/gpu` 变为 `32 × nvidia.com/gpu`——每个 Pod 请求 `nvidia.com/gpu: 1` 时，实际获得的是 1/4 张 GPU 的时间片。

**代价**：Time-slicing 不提供显存隔离。4 个 Pod 共享 80GB 显存，调度器不知道每个 Pod 实际用了多少。一个 Pod 的显存泄漏可以导致同 GPU 上的其他 Pod OOM。同时，时间片切换引入了调度延迟——推理服务的 P99 latency 会明显恶化。

---

## 四、三种共享方式的决策矩阵

| 场景                           | 推荐方式           | 原因                                                  |
| ------------------------------ | ------------------ | ----------------------------------------------------- |
| 多租户推理，需要严格 SLA 隔离  | MIG                | 显存和 SM 硬隔离，互不影响                            |
| 多个轻量推理服务，显存占用可控 | MPS + Time-slicing | MPS 提供计算并行，Time-slicing 让调度器能分配多个 Pod |
| 开发/实验环境，隔离需求低      | Time-slicing       | 配置最简单，不需要手动切 GPU                          |
| 训练作业                       | 不共享，独占整卡   | 训练对显存和带宽要求高，共享模式性能损失不可接受      |

---

## 五、相关资源

- [NVIDIA GPU Operator Time-slicing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html)
- [GPU 进程与资源管理](../01_gpu_ops/07_gpu_process_management.md) — MPS 和 MIG 的命令行操作
- [GPU 调度问题总览](01_gpu_scheduling_problem.md)
