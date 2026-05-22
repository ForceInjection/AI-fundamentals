"""
PyTorch NPU 实战: ResNet-50 模型训练与混合精度

在 Ascend 910B3 NPU 上训练 ResNet-50，对比:
  - FP32 训练
  - AMP (Automatic Mixed Precision) 训练
  - 性能对比与 profiling

用法:
  ASCEND_RT_VISIBLE_DEVICES=7 python3 train_resnet50.py [--amp]
"""

import torch
import torch_npu
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import argparse
import os


def get_dummy_dataloader(batch_size=64, num_batches=50, image_size=224):
    """使用随机数据创建 dataloader (避免下载真实数据集)"""
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=5000):
            self.size = size
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            img = torch.randn(3, image_size, image_size)
            img = torch.clamp(img, -3, 3)
            img = self.normalize(img)
            label = idx % 1000
            return img, label

    dataset = DummyDataset(size=batch_size * num_batches)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader


def train_one_epoch(model, dataloader, criterion, optimizer, scaler=None, device="npu:0"):
    """训练一个 epoch，可选 AMP"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    batch_times = []

    # warmup: 用 no_grad 避免梯度累积
    with torch.no_grad():
        for _ in range(3):
            model(torch.randn(4, 3, 224, 224).to(device))

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        t0 = time.time()

        if scaler is not None:
            # AMP 路径
            optimizer.zero_grad()
            # torch_npu 的 NPU AMP 接口，等价于 CUDA 的 torch.cuda.amp.autocast()
            with torch.npu.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # FP32 路径
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 为准确计时强制同步，会牺牲部分流水线并行
        torch.npu.synchronize()
        batch_time = time.time() - t0
        batch_times.append(batch_time)

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_batch_time = sum(batch_times[1:]) / len(batch_times[1:])  # 跳过 warmup
    throughput = dataloader.batch_size / avg_batch_time

    return avg_loss, throughput, batch_times[1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amp", action="store_true", help="使用 AMP 混合精度")
    parser.add_argument("--epochs", type=int, default=3, help="训练 epoch 数")
    args = parser.parse_args()

    device = "npu:0"
    batch_size = 64
    num_batches = 50

    print("=" * 60)
    print(f"  ResNet-50 训练 (device={device}, AMP={args.amp})")
    print("=" * 60)

    # 模型
    model = models.resnet50(weights=None).to(device)
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 优化器和损失
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # AMP scaler
    scaler = torch.npu.amp.GradScaler() if args.amp else None

    dataloader = get_dummy_dataloader(batch_size=batch_size, num_batches=num_batches)

    # 记录训练前 NPU 状态
    mem_before = torch.npu.memory_allocated() / 1024**2

    all_throughputs = []
    all_max_mem = []
    for epoch in range(args.epochs):
        avg_loss, throughput, batch_times = train_one_epoch(
            model, dataloader, criterion, optimizer, scaler, device
        )
        all_throughputs.append(throughput)
        print(f"  Epoch {epoch + 1}/{args.epochs}: "
              f"loss={avg_loss:.4f}, "
              f"throughput={throughput:.1f} img/s, "
              f"avg batch time={sum(batch_times)/len(batch_times)*1000:.1f}ms")

    avg_throughput = sum(all_throughputs) / len(all_throughputs)
    mem_after = torch.npu.memory_allocated() / 1024**2
    max_mem = torch.npu.max_memory_allocated() / 1024**2

    print("\n" + "=" * 60)
    print("  训练结果汇总")
    print("=" * 60)
    print(f"  混合精度:     {'AMP (FP16)' if args.amp else 'FP32'}")
    print(f"  平均吞吐:     {avg_throughput:.1f} img/s")
    print(f"  NPU 内存变化: {mem_after - mem_before:.0f} MB (峰值 {max_mem:.0f} MB)")

    if args.amp:
        print("\n  注: AMP 使用 FP16 训练，理论内存占用约为 FP32 的 50%")
        print("      实际加速取决于算子的 NPU 适配程度")


if __name__ == "__main__":
    main()
