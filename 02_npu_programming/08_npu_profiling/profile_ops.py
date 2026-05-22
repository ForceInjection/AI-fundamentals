"""
NPU 性能分析 — 算子级 profiling、带宽基准、实时监控

工具:
  - torch_npu.profiler: 算子级耗时追踪，输出 Chrome trace JSON
  - ascend-dmi --bw:     HBM/DDR 带宽基准测试
  - npu-smi:             实时 AI Core / HBM 利用率监控

用法:
  ASCEND_RT_VISIBLE_DEVICES=7 python3 profile_ops.py --bench all
  ASCEND_RT_VISIBLE_DEVICES=7 python3 profile_ops.py --bench matmul --size 16384
  ASCEND_RT_VISIBLE_DEVICES=7 python3 profile_ops.py --bench resnet50

注意: AI Core 硬件指标 (PipeUtilization 等) 需要 CANN >= 8.2.RC1，
      当前 CANN 8.0.1 仅支持 Level0 算子级 profiling。
"""

import argparse
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
import torch_npu
import torchvision.models as models
from torch_npu.profiler import (
    ProfilerActivity,
    profile,
    tensorboard_trace_handler,
)


# ── ProfileRunner ──

class ProfileRunner:
    """封装 torch_npu.profiler，统一管理 trace 输出"""

    def __init__(self, output_dir: str = "./profiler_output",
                 record_shapes: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.record_shapes = record_shapes

    @contextmanager
    def trace(self, name: str):
        """创建 profiling 上下文，输出 Chrome trace JSON"""
        trace_path = str(self.output_dir / name)
        activities = [ProfilerActivity.CPU, ProfilerActivity.NPU]

        with profile(
            activities=activities,
            record_shapes=self.record_shapes,
            on_trace_ready=tensorboard_trace_handler(trace_path),
        ):
            yield

        # 输出文件信息
        trace_dir = Path(trace_path)
        if trace_dir.is_dir():
            json_files = list(trace_dir.rglob("*.json"))
            if json_files:
                total_kb = sum(f.stat().st_size for f in json_files) / 1024
                print(f"  trace: {trace_dir}/ ({len(json_files)} 文件, {total_kb:.0f} KB)")


# ── npu-smi 监控 ──

def npu_smi_snapshot(device_id: int = 7) -> dict:
    """获取 npu-smi 实时指标快照"""
    result = subprocess.run(
        ["npu-smi", "info", "-t", "usages", "-i", str(device_id)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"npu-smi 执行失败 (device {device_id}): {result.stderr}")
    metrics = {}
    for line in result.stdout.split("\n"):
        line = line.strip()
        if "Aicore Usage Rate" in line:
            metrics["aicore_pct"] = int(line.split(":")[-1].strip().replace("%", ""))
        elif "HBM Usage Rate" in line:
            metrics["hbm_usage_pct"] = int(line.split(":")[-1].strip().replace("%", ""))
        elif "HBM Bandwidth Usage Rate" in line:
            metrics["hbm_bw_pct"] = int(line.split(":")[-1].strip().replace("%", ""))
        elif "Aicpu Usage Rate" in line:
            metrics["aicpu_pct"] = int(line.split(":")[-1].strip().replace("%", ""))
        elif "Aivector Usage Rate" in line:
            metrics["aivector_pct"] = int(line.split(":")[-1].strip().replace("%", ""))
        elif "DDR Bandwidth Usage Rate" in line:
            metrics["ddr_bw_pct"] = int(line.split(":")[-1].strip().replace("%", ""))
    return metrics


def print_npu_monitor_header():
    print(f"\n{'─'*70}")
    print(f"{'指标':<20} {'空闲时':>10} {'负载中':>10} {'变化':>10}")
    print(f"{'─'*70}")


def print_npu_metric(label: str, before: dict, after: dict, key: str, unit: str = "%"):
    v_before = before.get(key, 0)
    v_after = after.get(key, 0)
    delta = v_after - v_before
    delta_str = f"+{delta}{unit}" if delta > 0 else f"{delta}{unit}"
    print(f"{label:<20} {v_before:>8}{unit}  {v_after:>8}{unit}  {delta_str:>10}")


# ── ascend-dmi 带宽测试 ──

ASCEND_DMI = os.environ.get("ASCEND_DMI_PATH", "")
if not ASCEND_DMI:
    import glob as _glob
    candidates = _glob.glob("/usr/local/Ascend/toolbox/*/Ascend-DMI/bin/ascend-dmi")
    ASCEND_DMI = candidates[0] if candidates else "ascend-dmi"


def bandwidth_benchmark(device_id: int = 7):
    """运行 ascend-dmi HBM 带宽基准测试"""
    print(f"\n{'='*60}")
    print(f"  ascend-dmi HBM 带宽基准测试")
    print(f"{'='*60}")

    # HBM d2d (device-to-device) 带宽
    print("\n[1] HBM device-to-device 带宽:")
    try:
        result = subprocess.run(
            [ASCEND_DMI, "--bw", "-i", str(device_id), "-t", "h2d,d2h,d2d"],
            capture_output=True, text=True, timeout=120
        )
    except FileNotFoundError:
        print(f"  错误: 未找到 ascend-dmi (ASCEND_DMI={ASCEND_DMI})")
        return
    except subprocess.TimeoutExpired:
        print("  错误: ascend-dmi 带宽测试超时")
        return
    for line in result.stdout.split("\n"):
        if any(kw in line for kw in ["BW", "bandwidth", "Bandwidth", "GB/s", "MB/s", "throughput"]):
            print(f"  {line.strip()}")

    if result.stderr:
        # 精简输出关键信息
        for line in result.stderr.split("\n"):
            if any(kw in line for kw in ["BW", "bandwidth", "Bandwidth", "GB/s", "MB/s"]):
                print(f"  {line.strip()}")

    print("\n  完整输出请手动运行: ascend-dmi --bw -i 7 -t h2d,d2h,d2d")


# ── Benchmark functions ──

def profile_matmul(runner: ProfileRunner, size: int = 8192,
                   device: str = "npu:0", silent: bool = False):
    """矩阵乘法 profiling — 多尺寸对比"""
    if not silent:
        print(f"\n{'='*60}")
        print(f"  矩阵乘法 profiling ({size}×{size})")
        print(f"{'='*60}")

    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)

    # NPU 测试前的数据准备与 warmup
    a_npu = a_cpu.to(device)
    b_npu = b_cpu.to(device)
    # warmup
    for _ in range(5):
        _ = torch.matmul(a_npu, b_npu)
    torch.npu.synchronize()

    with runner.trace(f"matmul_{size}_npu"):
        t0 = time.time()
        _ = torch.matmul(a_npu, b_npu)
        torch.npu.synchronize()
        npu_time = time.time() - t0

    flops = 2 * size ** 3
    tflops = flops / npu_time / 1e12
    if not silent:
        print(f"  NPU 耗时: {npu_time * 1000:.1f} ms")
        print(f"  TFLOPS:   {tflops:.2f}")
        print(f"  HBM 读写: {4 * size ** 2 / 1024**2:.0f} MB × 2 = {8 * size ** 2 / 1024**2:.0f} MB")

    return {"size": size, "time_ms": npu_time * 1000, "tflops": tflops}


def profile_conv2d(runner: ProfileRunner, device: str = "npu:0"):
    """2D 卷积 profiling — 不同 kernel/stride 配置"""
    print(f"\n{'='*60}")
    print(f"  2D 卷积 profiling")
    print(f"{'='*60}")

    configs = [
        ("3x3/s1", 3, 1, 64, 64),    # 最常见配置
        ("3x3/s2", 3, 2, 64, 64),    # 带下采样
        ("7x7/s2", 7, 2, 3, 64),     # 大 kernel（ResNet stem）
        ("1x1/s1", 1, 1, 256, 256),  # bottleneck 降维
    ]

    results = []
    for name, ksize, stride, in_ch, out_ch in configs:
        x = torch.randn(16, in_ch, 224, 224).to(device)
        conv = torch.nn.Conv2d(in_ch, out_ch, ksize, stride=stride,
                               padding=ksize // 2, bias=False).to(device)
        # warmup
        for _ in range(5):
            _ = conv(x)
        torch.npu.synchronize()

        with runner.trace(f"conv2d_{name}"):
            t0 = time.time()
            for _ in range(10):
                _ = conv(x)
            torch.npu.synchronize()
            avg_time = (time.time() - t0) / 10

        param_count = sum(p.numel() for p in conv.parameters())
        macs = 2 * in_ch * ksize * ksize * out_ch * 224 * 224 / stride ** 2
        results.append((name, avg_time, param_count, macs))
        print(f"  Conv2d({name}): {avg_time*1000:.2f} ms, "
              f"params={param_count/1e6:.2f}M, MACs={macs/1e9:.1f}G")

    return results


def profile_resnet50(runner: ProfileRunner, device: str = "npu:0"):
    """ResNet-50 forward + backward profiling"""
    print(f"\n{'='*60}")
    print(f"  ResNet-50 profiling")
    print(f"{'='*60}")

    model = models.resnet50(weights=None).to(device).train()
    dummy = torch.randn(8, 3, 224, 224).to(device)
    label = torch.randint(0, 1000, (8,)).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # warmup
    for _ in range(3):
        out = model(dummy)
        loss = criterion(out, label)
        loss.backward()
    torch.npu.synchronize()

    mem_before = torch.npu.memory_allocated() / 1024 ** 2

    # Forward profiling
    with runner.trace("resnet50_forward"):
        t0 = time.time()
        out = model(dummy)
        torch.npu.synchronize()
        fwd_time = time.time() - t0

    # Forward + backward profiling
    model.zero_grad()
    with runner.trace("resnet50_full"):
        t0 = time.time()
        out = model(dummy)
        loss = criterion(out, label)
        loss.backward()
        torch.npu.synchronize()
        full_time = time.time() - t0

    bwd_time = full_time - fwd_time
    mem_after = torch.npu.memory_allocated() / 1024 ** 2
    peak_mem = torch.npu.max_memory_allocated() / 1024 ** 2

    params = sum(p.numel() for p in model.parameters())
    print(f"  参数量:     {params / 1e6:.2f}M")
    print(f"  Forward:    {fwd_time * 1000:.1f} ms")
    print(f"  Backward:   {bwd_time * 1000:.1f} ms")
    print(f"  总计:       {full_time * 1000:.1f} ms")
    print(f"  HBM 占用:   {mem_before:.0f} → {mem_after:.0f} MB (峰值 {peak_mem:.0f} MB)")
    print(f"  吞吐:       {8 / full_time:.1f} img/s")

    return {"fwd_ms": fwd_time * 1000, "bwd_ms": bwd_time * 1000,
            "total_ms": full_time * 1000, "peak_mb": peak_mem}


def monitor_during_workload(device_id: int = 7, device: str = "npu:0"):
    """在负载前后采集 npu-smi 快照，对比利用率变化"""
    print(f"\n{'='*60}")
    print(f"  npu-smi 实时监控")
    print(f"{'='*60}")

    # 空闲快照
    before = npu_smi_snapshot(device_id)
    time.sleep(0.5)

    # 跑一次重负载 (大矩阵乘法)
    size = 16384
    a = torch.randn(size, size).to(device)
    b = torch.randn(size, size).to(device)
    torch.npu.synchronize()

    # 持续跑 3 秒让利用率上来
    t0 = time.time()
    iters = 0
    while time.time() - t0 < 3:
        _ = torch.matmul(a, b)
        iters += 1
    torch.npu.synchronize()

    after = npu_smi_snapshot(device_id)

    # 释放大矩阵显存
    del a, b
    torch.npu.empty_cache()

    print_npu_monitor_header()
    print_npu_metric("AI Core 利用率", before, after, "aicore_pct")
    print_npu_metric("AI Vector 利用率", before, after, "aivector_pct")
    print_npu_metric("AI CPU 利用率", before, after, "aicpu_pct")
    print_npu_metric("HBM 带宽利用率", before, after, "hbm_bw_pct")
    print_npu_metric("DDR 带宽利用率", before, after, "ddr_bw_pct")
    print(f"{'─'*70}")
    print(f"  3s 内完成 {iters} 次 {size}×{size} matmul")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="NPU 性能分析 — 算子级 profiling、带宽基准、实时监控"
    )
    parser.add_argument("--bench", default="all",
                        choices=["matmul", "conv2d", "resnet50", "monitor", "bandwidth", "all"],
                        help="benchmark 目标 (default: all)")
    parser.add_argument("--size", type=int, default=16384,
                        help="矩阵乘法尺寸 (default: 16384)")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", default="./profiler_output",
                        help="trace 输出目录 (default: ./profiler_output)")

    args = parser.parse_args()

    print("=" * 60)
    print("  NPU 性能分析")
    print(f"  Device: {args.device}")
    print(f"  Benchmark: {args.bench}")
    print(f"  Output: {args.output}")
    print(f"  CANN: 8.0.1 | torch_npu: {torch_npu.__version__}")
    print("=" * 60)

    torch.npu.set_device(args.device)
    runner = ProfileRunner(output_dir=args.output)

    if args.bench in ("matmul", "all"):
        for s in [4096, 8192, 16384]:
            profile_matmul(runner, size=s, device=args.device)

    if args.bench in ("conv2d", "all"):
        profile_conv2d(runner, args.device)

    if args.bench in ("resnet50", "all"):
        profile_resnet50(runner, args.device)

    if args.bench in ("monitor", "all"):
        device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
        monitor_during_workload(device_id=device_id, device=args.device)

    if args.bench in ("bandwidth", "all"):
        device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0
        bandwidth_benchmark(device_id=device_id)

    print(f"\n所有 trace 文件保存在: {Path(args.output).absolute()}")
    print("用 Chrome 打开 trace JSON: chrome://tracing → Load")


if __name__ == "__main__":
    main()
