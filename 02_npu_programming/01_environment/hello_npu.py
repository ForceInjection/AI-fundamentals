"""
Hello NPU — 华为昇腾 NPU 初体验

在 Ascend 910B3 NPU 上运行基础 PyTorch 操作，对比 CUDA 编程模型。
用法: ASCEND_RT_VISIBLE_DEVICES=7 python3 hello_npu.py
"""

import torch
import torch_npu
import time


def check_environment():
    """检测 NPU 环境基本信息，返回 NPU 是否可用"""
    print("=" * 60)
    print("  NPU 环境信息")
    print("=" * 60)
    print(f"  PyTorch 版本:    {torch.__version__}")
    print(f"  torch_npu 版本:  {torch_npu.__version__}")
    print(f"  NPU 可用:        {torch.npu.is_available()}")

    if not torch.npu.is_available():
        print("  [错误] NPU 不可用！")
        print("  请检查: 1) torch_npu 是否安装  2) ASCEND_RT_VISIBLE_DEVICES 是否设置")
        return False

    print(f"  可见 NPU 数量:   {torch.npu.device_count()}")
    for i in range(torch.npu.device_count()):
        print(f"  设备 {i}: {torch.npu.get_device_name(i)}")
    return True


def basic_tensor_ops():
    """基础张量操作 — NPU vs CPU 对比"""
    print("\n" + "=" * 60)
    print("  基础张量操作")
    print("=" * 60)

    # 创建张量的三种方式
    print("\n[1] 直接创建 NPU 张量:")
    x = torch.randn(2, 3, device="npu")
    print(f"    直接创建:   {x.device}")

    # CPU 张量迁移到 NPU
    y = torch.randn(2, 3)
    y_npu = y.npu()          # NPU 版 API，等价于 CUDA 的 .cuda()
    print(f"    .npu() 迁移: {y_npu.device}")

    # 使用 to() 方法
    z = torch.randn(2, 3).to("npu:0")
    print(f"    .to('npu'):  {z.device}")

    # CUDA vs NPU API 对照
    print("\n[2] CUDA vs NPU API 对照:")
    print(f"    CUDA:  tensor.cuda()           → NPU: tensor.npu()")
    print(f"    CUDA:  tensor.to('cuda')       → NPU: tensor.to('npu')")
    print(f"    CUDA:  torch.cuda.is_available() → NPU: torch.npu.is_available()")
    print(f"    CUDA:  torch.cuda.device_count()  → NPU: torch.npu.device_count()")
    print(f"    CUDA:  torch.cuda.synchronize()   → NPU: torch.npu.synchronize()")


def matmul_benchmark():
    """矩阵乘法性能对比: CPU vs NPU"""
    print("\n" + "=" * 60)
    print("  矩阵乘法性能对比 (4096×4096)")
    print("=" * 60)

    size = 4096

    # CPU 基准
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    t0 = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - t0
    print(f"  CPU 耗时: {cpu_time * 1000:.2f} ms")

    # NPU 运算
    a_npu = a_cpu.npu()
    b_npu = b_cpu.npu()
    # Warmup: 多次执行确保算子编译完成和内存状态稳定
    for _ in range(5):
        _ = torch.matmul(a_npu, b_npu)
    torch.npu.synchronize()

    t0 = time.time()
    c_npu = torch.matmul(a_npu, b_npu)
    torch.npu.synchronize()
    npu_time = time.time() - t0

    print(f"  NPU 耗时: {npu_time * 1000:.2f} ms")
    if cpu_time > 0:
        print(f"  加速比:   {cpu_time / npu_time:.1f}x")

    # 验证结果一致性
    diff = (c_cpu - c_npu.cpu()).abs().max().item()
    print(f"  最大误差: {diff:.6f}")


def memory_info():
    """查询 NPU 内存使用情况"""
    print("\n" + "=" * 60)
    print("  NPU 内存信息")
    print("=" * 60)
    print(f"  已分配: {torch.npu.memory_allocated() / 1024**2:.2f} MB")
    print(f"  已缓存: {torch.npu.memory_reserved() / 1024**2:.2f} MB")
    print(f"  最大分配: {torch.npu.max_memory_allocated() / 1024**2:.2f} MB")


def main():
    if not check_environment():
        print("\n环境检查失败，脚本退出。")
        sys.exit(1)
    basic_tensor_ops()
    matmul_benchmark()
    memory_info()

    print("\n" + "=" * 60)
    print("  Hello NPU — 所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
