#!/usr/bin/env python3
"""
NCCL 分布式通信测试模板
用于测试 GPU 间的 NCCL 通信性能
"""

import os
import sys
import time
import signal
import socket
import torch
import torch.distributed as dist

def get_local_ip():
    """获取本机IP地址"""
    try:
        # 创建一个UDP socket来获取本机IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # 连接到一个远程地址（不会实际发送数据）
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception:
        return "127.0.0.1"

def print_system_info():
    """打印系统信息"""
    print("=== 系统信息 ===")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"NCCL 版本: {torch.cuda.nccl.version()}")
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")
    
    # 打印GPU信息
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    local_ip = get_local_ip()
    print(f"本机 IP: {local_ip}")
    print()

def print_nccl_info():
    """打印 NCCL 相关信息"""
    print("=== NCCL 环境信息 ===")
    
    # NCCL 版本
    if hasattr(torch.cuda.nccl, 'version'):
        print(f"NCCL 版本: {torch.cuda.nccl.version()}")
    else:
        print("NCCL 版本: 未知")
    
    # 环境变量
    nccl_vars = [
        'NCCL_IB_DISABLE', 'NCCL_NET_DISABLE', 'NCCL_NET_GDR_LEVEL', 'NCCL_IB_HCA',
        'NCCL_IB_GID_INDEX', 'NCCL_DEBUG', 'NCCL_DEBUG_SUBSYS',
        'NCCL_P2P_DISABLE', 'NCCL_P2P_LEVEL', 'NCCL_NVLS_ENABLE',
        'NCCL_SOCKET_IFNAME', 'NCCL_IB_TIMEOUT', 'NCCL_IB_RETRY_CNT',
        'NCCL_SHM_DISABLE', 'NCCL_SOCKET_FAMILY', 'NCCL_SOCKET_NTHREADS'
    ]
    
    print("NCCL 环境变量:")
    for var in nccl_vars:
        value = os.environ.get(var, '未设置')
        print(f"  {var}: {value}")
    print()

def run_nccl_test(rank, world_size, local_rank, backend='nccl', tensor_elements=1000000, test_duration=30):
    """运行 NCCL 测试"""
    try:
        # 检查 CUDA 可用性
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        
        # 检查 GPU 数量
        gpu_count = torch.cuda.device_count()
        if local_rank >= gpu_count:
            raise RuntimeError(f"local_rank {local_rank} 超出可用 GPU 数量 {gpu_count}")
        
        print(f"[Rank {rank}] 开始初始化分布式环境...")
        print(f"[Rank {rank}] 后端: {backend}")
        print(f"[Rank {rank}] 世界大小: {world_size}")
        print(f"[Rank {rank}] 本地排名: {local_rank}")
        print(f"[Rank {rank}] 张量大小: {tensor_elements} 个元素 ({tensor_elements * 4 / 1024 / 1024:.2f} MB)")
        
        # 设置设备 - 使用 local_rank 而不是 rank
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        
        # 初始化分布式环境
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        
        print(f"[Rank {rank}] 分布式环境初始化成功")
        print(f"[Rank {rank}] 使用设备: {device}")
        print(f"[Rank {rank}] GPU: {torch.cuda.get_device_name(device)}")
        
        # 创建测试张量
        tensor = torch.randn(tensor_elements, device=device)
        
        # 同步所有进程
        dist.barrier()
        
        # 预热
        print(f"[Rank {rank}] 开始预热...")
        for _ in range(5):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
        
        # 同步所有进程，确保同时开始测试
        dist.barrier()
        
        print(f"[Rank {rank}] 预热完成，开始性能测试...")
        print(f"[Rank {rank}] 测试时长: {test_duration} 秒")
        
        # 性能测试
        start_time = time.time()
        iteration = 0
        total_data_transferred = 0
        latencies = []  # 存储延迟数据
        
        # 创建停止标志张量（所有进程共享）
        stop_flag = torch.zeros(1, device=device, dtype=torch.int32)
        
        while True:
            iter_start = time.time()
            
            # AllReduce 操作
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            
            iter_end = time.time()
            iteration += 1
            
            # 计算性能指标
            iter_time = (iter_end - iter_start) * 1000  # 转换为毫秒
            latencies.append(iter_time)
            data_size_mb = tensor_elements * 4 / 1024 / 1024  # MB
            # AllReduce 的理论传输量是 2 * (world_size - 1) * data_size
            theoretical_transfer = 2 * (world_size - 1) * data_size_mb
            total_data_transferred += theoretical_transfer
            
            # 减少打印频率以提高测试准确性
            if iteration <= 10 or iteration % 50 == 0:
                if iter_time > 0:
                    throughput_gbps = (theoretical_transfer / 1024) / (iter_time / 1000)  # GB/s
                    print(f"[Rank {rank}] 迭代 {iteration}: {iter_time:.2f} ms, {throughput_gbps:.2f} GB/s")
            
            # 每100次迭代报告一次进度
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"[Rank {rank}] 已完成 {iteration} 次迭代，耗时 {elapsed:.1f} 秒")
            
            # 每50次迭代检查一次是否需要停止（减少同步开销）
            if iteration % 50 == 0:
                # 只有 rank 0 检查时间并设置停止标志
                if rank == 0:
                    if time.time() - start_time >= test_duration:
                        stop_flag[0] = 1
                        print(f"[Rank {rank}] 测试时间到达，设置停止标志")
                
                # 同步停止标志到所有进程
                dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
                
                # 如果停止标志被设置，所有进程同时退出
                if stop_flag[0].item() == 1:
                    print(f"[Rank {rank}] 收到停止信号，退出测试循环")
                    break
        
        # 测试完成，计算总体统计
        total_time = time.time() - start_time
        avg_latency = sum(latencies) / len(latencies)  # 平均延迟（毫秒）
        min_latency = min(latencies)
        max_latency = max(latencies)
        total_throughput = (total_data_transferred / 1024) / total_time  # 总吞吐量（GB/s）
        
        print(f"\n[Rank {rank}] === 测试完成 ===")
        print(f"[Rank {rank}] 总迭代次数: {iteration}")
        print(f"[Rank {rank}] 总测试时间: {total_time:.2f} 秒")
        print(f"[Rank {rank}] 平均延迟: {avg_latency:.2f} ms")
        print(f"[Rank {rank}] 最小延迟: {min_latency:.2f} ms")
        print(f"[Rank {rank}] 最大延迟: {max_latency:.2f} ms")
        print(f"[Rank {rank}] 平均吞吐量: {total_throughput:.2f} GB/s")
        print(f"[Rank {rank}] 数据大小: {tensor_elements} 个元素 ({tensor_elements * 4 / 1024 / 1024:.2f} MB)")
        print(f"[Rank {rank}] 理论传输量: {total_data_transferred / 1024:.2f} GB")
        print(f"[Rank {rank}] 网络传输倍数: {2 * (world_size - 1):.1f}x")
        
        # 只在 rank 0 打印最终结果
        if rank == 0:
            print("✅ NCCL AllReduce 测试成功完成")
            print("测试成功完成")
        
        print(f"[Rank {rank}] 开始清理资源...")
        
        # 确保所有 CUDA 操作完成
        torch.cuda.synchronize()
        
        # 清理 CUDA 缓存
        try:
            torch.cuda.empty_cache()
            print(f"[Rank {rank}] CUDA 缓存已清理")
        except Exception as cache_error:
            print(f"[Rank {rank}] 清理 CUDA 缓存时出错: {cache_error}")
        
        print(f"[Rank {rank}] 跳过所有同步，直接退出")
        print(f"[Rank {rank}] 进程即将退出")
        
        # 强制退出，避免卡住 - 不调用任何分布式清理函数
        os._exit(0)
        
    except Exception as e:
        print(f"[Rank {rank if 'rank' in locals() else 'Unknown'}] 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """主函数"""
    import signal
    
    def timeout_handler(signum, frame):
        print(f"[Rank {os.environ.get('RANK', 'Unknown')}] 测试超时，强制退出")
        sys.exit(1)
    
    # 设置超时处理（测试时间 + 60秒缓冲）
    test_duration = int(os.environ.get('TEST_DURATION', 30))
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(test_duration + 60)
    
    try:
        # 从环境变量获取分布式参数
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # 从环境变量获取测试参数
        tensor_elements = int(os.environ.get('TENSOR_ELEMENTS', 1000000))
        backend = os.environ.get('NCCL_BACKEND', 'nccl')
        
        # 只在 rank 0 打印系统信息
        if rank == 0:
            print_system_info()
            print_nccl_info()
        
        # 运行测试 - 传递 local_rank 参数
        run_nccl_test(rank, world_size, local_rank, backend, tensor_elements, test_duration)
        
    except KeyboardInterrupt:
        print(f"[Rank {os.environ.get('RANK', 'Unknown')}] 收到中断信号，正在退出...")
        sys.exit(0)
    except Exception as e:
        print(f"[Rank {os.environ.get('RANK', 'Unknown')}] 主函数发生错误: {e}")
        sys.exit(1)
    finally:
        # 取消超时警报
        signal.alarm(0)

if __name__ == "__main__":
    main()