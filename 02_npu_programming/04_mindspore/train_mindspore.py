"""
MindSpore 实战: 在 Ascend 上训练卷积网络

对比:
  - PyNative 模式 (动态图，类似 PyTorch eager)
  - Graph 模式 (静态图，编译优化)
  - MindSpore vs PyTorch API 差异

用法:
  ASCEND_RT_VISIBLE_DEVICES=7 python3 train_mindspore.py [--graph]
"""

import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.dataset import GeneratorDataset
import numpy as np
import time
import argparse


# ── Bottleneck Block (ResNet-50 核心组件) ──
class Bottleneck(nn.Cell):
    """ResNet-50 Bottleneck: 1x1→3x3→1x1 卷积 + 残差连接"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        mid_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride,
                               pad_mode='pad', padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.downsample_layer = downsample

    def construct(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample_layer is not None:
            identity = self.downsample_layer(x)
        out = out + identity
        return self.relu(out)


# ── ResNet-50 ──
class ResNet50(nn.Cell):
    """简化的 ResNet-50，包含 4 个 stage"""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, pad_mode='pad', padding=3, has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, pad_mode='pad', padding=1)

        # 4 stages, 每个 stage 包含多个 Bottleneck
        self.layer1 = self._make_layer(64, 256, 3, stride=1)
        self.layer2 = self._make_layer(256, 512, 4, stride=2)
        self.layer3 = self._make_layer(512, 1024, 6, stride=2)
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(2048, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels, 1, stride, has_bias=False),
                nn.BatchNorm2d(out_channels),
            ])
        layers = [Bottleneck(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, out_channels))
        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ── Dataset (复用 PyTorch 版本的 dummy 逻辑) ──
def create_dataset(batch_size=64, num_batches=50, image_size=224):
    def generator():
        for i in range(batch_size * num_batches):
            img = np.random.randn(3, image_size, image_size).astype(np.float32)
            label = i % 1000
            yield (img, np.array(label, dtype=np.int32))

    ds = GeneratorDataset(
        source=generator,
        column_names=["image", "label"]
    )
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


# ── Training ──
def train_one_epoch(net, dataset, optimizer, loss_fn):
    net.set_train(True)

    def forward_fn(data, label):
        logits = net(data)
        loss = loss_fn(logits, label)
        return loss

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    total_loss = 0.0
    total_samples = 0
    batch_times = []

    for images, labels in dataset.create_tuple_iterator():
        labels = labels.astype(ms.int32)

        t0 = time.time()

        loss, grads = grad_fn(images, labels)
        optimizer(grads)

        batch_time = time.time() - t0
        batch_times.append(batch_time)

        total_loss += loss.asnumpy().item() * images.shape[0]
        total_samples += images.shape[0]

    avg_loss = total_loss / total_samples
    avg_batch_time = sum(batch_times[1:]) / len(batch_times[1:]) if len(batch_times) > 1 else batch_times[0]
    throughput = images.shape[0] / avg_batch_time

    return avg_loss, throughput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", action="store_true", help="使用 Graph 模式")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    mode_name = "Graph" if args.graph else "PyNative"
    mode = ms.GRAPH_MODE if args.graph else ms.PYNATIVE_MODE
    ms.set_context(mode=mode, device_target="Ascend", device_id=0)

    print("=" * 60)
    print(f"  MindSpore ResNet-50 Training (mode={mode_name})")
    print(f"  MindSpore version: {ms.__version__}")
    print("=" * 60)

    net = ResNet50(num_classes=1000)
    num_params = sum(p.size for p in net.get_parameters())
    print(f"  Parameters: {num_params / 1e6:.2f}M")

    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01, momentum=0.9)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    dataset = create_dataset(batch_size=64, num_batches=50)

    # Warmup
    print("  Warming up...")
    warmup = Tensor(np.random.randn(4, 3, 224, 224).astype(np.float32))
    _ = net(warmup)
    print("  Ready.\n")

    all_throughputs = []
    for epoch in range(args.epochs):
        avg_loss, throughput = train_one_epoch(net, dataset, optimizer, loss_fn)
        all_throughputs.append(throughput)
        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, throughput={throughput:.1f} img/s")

    avg_throughput = sum(all_throughputs) / len(all_throughputs)

    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  Mode:       {mode_name}")
    print(f"  Throughput: {avg_throughput:.1f} img/s")


if __name__ == "__main__":
    main()
