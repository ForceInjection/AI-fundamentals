#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI基准测试脚本
提供全面的AI模型性能测试和基准测试功能
支持PyTorch、TensorFlow、ONNX等框架
"""

import os
import sys
import time
import json
import argparse
import subprocess
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，将跳过PyTorch相关测试")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("警告: TensorFlow未安装，将跳过TensorFlow相关测试")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("警告: ONNX Runtime未安装，将跳过ONNX相关测试")

class AIBenchmark:
    """AI基准测试类"""
    
    def __init__(self, output_dir: str = None, device: str = 'auto'):
        self.output_dir = output_dir or f"ai_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.device = self._setup_device(device)
        self.results = {}
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志文件
        self.log_file = os.path.join(self.output_dir, 'benchmark.log')
        
        print(f"AI基准测试初始化完成")
        print(f"输出目录: {self.output_dir}")
        print(f"计算设备: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """设置计算设备"""
        if device == 'auto':
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return 'cuda'
            elif TF_AVAILABLE and tf.config.list_physical_devices('GPU'):
                return 'gpu'
            else:
                return 'cpu'
        return device
    
    def log(self, message: str, level: str = 'INFO'):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
        info = {
            'timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        # GPU信息
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', 
                                   '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 3:
                            gpu_info.append({
                                'name': parts[0],
                                'memory': parts[1],
                                'compute_capability': parts[2]
                            })
                info['gpus'] = gpu_info
        except Exception as e:
            info['gpu_error'] = str(e)
        
        # 框架版本
        if TORCH_AVAILABLE:
            info['pytorch_version'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['cuda_version'] = torch.version.cuda
                info['cudnn_version'] = torch.backends.cudnn.version()
        
        if TF_AVAILABLE:
            info['tensorflow_version'] = tf.__version__
            info['tf_gpu_available'] = len(tf.config.list_physical_devices('GPU')) > 0
        
        if ONNX_AVAILABLE:
            info['onnxruntime_version'] = ort.__version__
        
        return info
    
    def pytorch_benchmark(self) -> Dict:
        """PyTorch基准测试"""
        if not TORCH_AVAILABLE:
            return {'error': 'PyTorch not available'}
        
        self.log("开始PyTorch基准测试")
        results = {}
        
        device = torch.device(self.device if self.device != 'gpu' else 'cuda')
        
        # 1. 图像分类模型测试
        self.log("测试图像分类模型...")
        classification_results = self._test_pytorch_classification(device)
        results['classification'] = classification_results
        
        # 2. 自然语言处理模型测试
        self.log("测试NLP模型...")
        nlp_results = self._test_pytorch_nlp(device)
        results['nlp'] = nlp_results
        
        # 3. 内存和计算性能测试
        self.log("测试内存和计算性能...")
        compute_results = self._test_pytorch_compute(device)
        results['compute'] = compute_results
        
        return results
    
    def _test_pytorch_classification(self, device) -> Dict:
        """测试PyTorch图像分类模型"""
        results = {}
        
        # 测试不同模型
        models_to_test = [
            ('ResNet18', models.resnet18),
            ('ResNet50', models.resnet50),
            ('VGG16', models.vgg16),
            ('MobileNetV2', models.mobilenet_v2)
        ]
        
        batch_sizes = [1, 4, 8, 16, 32]
        input_size = (3, 224, 224)
        
        for model_name, model_fn in models_to_test:
            self.log(f"测试模型: {model_name}")
            model_results = {}
            
            try:
                model = model_fn(pretrained=False).to(device)
                model.eval()
                
                for batch_size in batch_sizes:
                    # 创建随机输入
                    input_tensor = torch.randn(batch_size, *input_size).to(device)
                    
                    # 预热
                    with torch.no_grad():
                        for _ in range(10):
                            _ = model(input_tensor)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    # 推理性能测试
                    start_time = time.time()
                    with torch.no_grad():
                        for _ in range(100):
                            output = model(input_tensor)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / 100
                    throughput = batch_size / avg_time
                    
                    model_results[f'batch_{batch_size}'] = {
                        'avg_time_ms': avg_time * 1000,
                        'throughput_samples_per_sec': throughput
                    }
                
                results[model_name] = model_results
                
            except Exception as e:
                self.log(f"模型 {model_name} 测试失败: {str(e)}", 'ERROR')
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _test_pytorch_nlp(self, device) -> Dict:
        """测试PyTorch NLP模型"""
        results = {}
        
        # Transformer模型测试
        try:
            # 简单的Transformer编码器
            class SimpleTransformer(nn.Module):
                def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, d_model)
                    self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
                    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
                    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                    self.fc = nn.Linear(d_model, vocab_size)
                
                def forward(self, x):
                    seq_len = x.size(1)
                    x = self.embedding(x) + self.pos_encoding[:seq_len]
                    x = self.transformer(x)
                    return self.fc(x)
            
            model = SimpleTransformer().to(device)
            model.eval()
            
            sequence_lengths = [128, 256, 512]
            batch_sizes = [1, 4, 8, 16]
            
            for seq_len in sequence_lengths:
                for batch_size in batch_sizes:
                    # 创建随机输入
                    input_ids = torch.randint(0, 10000, (batch_size, seq_len)).to(device)
                    
                    # 预热
                    with torch.no_grad():
                        for _ in range(5):
                            _ = model(input_ids)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    # 性能测试
                    start_time = time.time()
                    with torch.no_grad():
                        for _ in range(20):
                            output = model(input_ids)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / 20
                    throughput = batch_size / avg_time
                    
                    key = f'seq_{seq_len}_batch_{batch_size}'
                    results[key] = {
                        'avg_time_ms': avg_time * 1000,
                        'throughput_samples_per_sec': throughput
                    }
        
        except Exception as e:
            self.log(f"NLP模型测试失败: {str(e)}", 'ERROR')
            results['error'] = str(e)
        
        return results
    
    def _test_pytorch_compute(self, device) -> Dict:
        """测试PyTorch计算性能"""
        results = {}
        
        # 矩阵乘法测试
        matrix_sizes = [512, 1024, 2048, 4096]
        
        for size in matrix_sizes:
            try:
                a = torch.randn(size, size).to(device)
                b = torch.randn(size, size).to(device)
                
                # 预热
                for _ in range(10):
                    _ = torch.mm(a, b)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # 性能测试
                start_time = time.time()
                for _ in range(100):
                    c = torch.mm(a, b)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 100
                flops = 2 * size ** 3
                gflops = flops / avg_time / 1e9
                
                results[f'matmul_{size}x{size}'] = {
                    'avg_time_ms': avg_time * 1000,
                    'gflops': gflops
                }
                
            except Exception as e:
                self.log(f"矩阵乘法测试失败 (size={size}): {str(e)}", 'ERROR')
        
        return results
    
    def tensorflow_benchmark(self) -> Dict:
        """TensorFlow基准测试"""
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not available'}
        
        self.log("开始TensorFlow基准测试")
        results = {}
        
        # 设置GPU内存增长
        if self.device == 'gpu':
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    self.log(f"GPU配置错误: {e}", 'ERROR')
        
        # 1. 图像分类模型测试
        self.log("测试TensorFlow图像分类模型...")
        classification_results = self._test_tf_classification()
        results['classification'] = classification_results
        
        # 2. 计算性能测试
        self.log("测试TensorFlow计算性能...")
        compute_results = self._test_tf_compute()
        results['compute'] = compute_results
        
        return results
    
    def _test_tf_classification(self) -> Dict:
        """测试TensorFlow图像分类模型"""
        results = {}
        
        # 测试不同模型
        models_to_test = [
            ('MobileNetV2', tf.keras.applications.MobileNetV2),
            ('ResNet50', tf.keras.applications.ResNet50),
            ('VGG16', tf.keras.applications.VGG16)
        ]
        
        batch_sizes = [1, 4, 8, 16]
        input_shape = (224, 224, 3)
        
        for model_name, model_fn in models_to_test:
            self.log(f"测试TensorFlow模型: {model_name}")
            model_results = {}
            
            try:
                model = model_fn(weights=None, input_shape=input_shape)
                
                for batch_size in batch_sizes:
                    # 创建随机输入
                    input_data = tf.random.normal((batch_size,) + input_shape)
                    
                    # 预热
                    for _ in range(10):
                        _ = model(input_data, training=False)
                    
                    # 性能测试
                    start_time = time.time()
                    for _ in range(50):
                        output = model(input_data, training=False)
                    
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / 50
                    throughput = batch_size / avg_time
                    
                    model_results[f'batch_{batch_size}'] = {
                        'avg_time_ms': avg_time * 1000,
                        'throughput_samples_per_sec': throughput
                    }
                
                results[model_name] = model_results
                
            except Exception as e:
                self.log(f"TensorFlow模型 {model_name} 测试失败: {str(e)}", 'ERROR')
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _test_tf_compute(self) -> Dict:
        """测试TensorFlow计算性能"""
        results = {}
        
        # 矩阵乘法测试
        matrix_sizes = [512, 1024, 2048, 4096]
        
        for size in matrix_sizes:
            try:
                a = tf.random.normal((size, size))
                b = tf.random.normal((size, size))
                
                # 预热
                for _ in range(10):
                    _ = tf.matmul(a, b)
                
                # 性能测试
                start_time = time.time()
                for _ in range(100):
                    c = tf.matmul(a, b)
                
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 100
                flops = 2 * size ** 3
                gflops = flops / avg_time / 1e9
                
                results[f'matmul_{size}x{size}'] = {
                    'avg_time_ms': avg_time * 1000,
                    'gflops': gflops
                }
                
            except Exception as e:
                self.log(f"TensorFlow矩阵乘法测试失败 (size={size}): {str(e)}", 'ERROR')
        
        return results
    
    def onnx_benchmark(self) -> Dict:
        """ONNX Runtime基准测试"""
        if not ONNX_AVAILABLE:
            return {'error': 'ONNX Runtime not available'}
        
        self.log("开始ONNX Runtime基准测试")
        results = {}
        
        try:
            # 创建简单的ONNX模型进行测试
            if TORCH_AVAILABLE:
                results = self._test_onnx_with_pytorch()
            else:
                results['error'] = 'PyTorch required for ONNX model creation'
        
        except Exception as e:
            self.log(f"ONNX测试失败: {str(e)}", 'ERROR')
            results['error'] = str(e)
        
        return results
    
    def _test_onnx_with_pytorch(self) -> Dict:
        """使用PyTorch创建ONNX模型进行测试"""
        results = {}
        
        try:
            # 创建简单的CNN模型
            class SimpleCNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                    self.pool = nn.AdaptiveAvgPool2d((1, 1))
                    self.fc = nn.Linear(64, 10)
                
                def forward(self, x):
                    x = torch.relu(self.conv1(x))
                    x = torch.relu(self.conv2(x))
                    x = self.pool(x)
                    x = x.view(x.size(0), -1)
                    return self.fc(x)
            
            model = SimpleCNN()
            model.eval()
            
            # 导出ONNX模型
            dummy_input = torch.randn(1, 3, 224, 224)
            onnx_path = os.path.join(self.output_dir, 'simple_cnn.onnx')
            
            torch.onnx.export(
                model, dummy_input, onnx_path,
                input_names=['input'], output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            # 创建ONNX Runtime会话
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            session = ort.InferenceSession(onnx_path, providers=providers)
            
            # 测试不同批次大小
            batch_sizes = [1, 4, 8, 16]
            
            for batch_size in batch_sizes:
                input_data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
                
                # 预热
                for _ in range(10):
                    _ = session.run(['output'], {'input': input_data})
                
                # 性能测试
                start_time = time.time()
                for _ in range(100):
                    output = session.run(['output'], {'input': input_data})
                
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 100
                throughput = batch_size / avg_time
                
                results[f'batch_{batch_size}'] = {
                    'avg_time_ms': avg_time * 1000,
                    'throughput_samples_per_sec': throughput
                }
        
        except Exception as e:
            self.log(f"ONNX模型测试失败: {str(e)}", 'ERROR')
            results['error'] = str(e)
        
        return results
    
    def memory_benchmark(self) -> Dict:
        """内存性能测试"""
        self.log("开始内存性能测试")
        results = {}
        
        if TORCH_AVAILABLE and self.device == 'cuda':
            results['pytorch_cuda'] = self._test_pytorch_memory()
        
        if TF_AVAILABLE and self.device == 'gpu':
            results['tensorflow_gpu'] = self._test_tf_memory()
        
        return results
    
    def _test_pytorch_memory(self) -> Dict:
        """测试PyTorch CUDA内存性能"""
        results = {}
        device = torch.device('cuda')
        
        # 内存分配测试
        sizes = [1024, 2048, 4096, 8192]
        
        for size in sizes:
            try:
                # 清空缓存
                torch.cuda.empty_cache()
                
                # 记录初始内存
                initial_memory = torch.cuda.memory_allocated()
                
                # 分配内存
                start_time = time.time()
                tensor = torch.randn(size, size, device=device)
                alloc_time = time.time() - start_time
                
                # 记录分配后内存
                allocated_memory = torch.cuda.memory_allocated() - initial_memory
                
                # 内存复制测试
                start_time = time.time()
                tensor_copy = tensor.clone()
                copy_time = time.time() - start_time
                
                # 释放内存
                del tensor, tensor_copy
                torch.cuda.empty_cache()
                
                results[f'size_{size}x{size}'] = {
                    'allocation_time_ms': alloc_time * 1000,
                    'copy_time_ms': copy_time * 1000,
                    'memory_mb': allocated_memory / 1024 / 1024
                }
                
            except Exception as e:
                self.log(f"PyTorch内存测试失败 (size={size}): {str(e)}", 'ERROR')
        
        return results
    
    def _test_tf_memory(self) -> Dict:
        """测试TensorFlow GPU内存性能"""
        results = {}
        
        # TensorFlow内存测试相对简单
        sizes = [1024, 2048, 4096]
        
        for size in sizes:
            try:
                # 分配内存
                start_time = time.time()
                tensor = tf.random.normal((size, size))
                alloc_time = time.time() - start_time
                
                # 内存复制测试
                start_time = time.time()
                tensor_copy = tf.identity(tensor)
                copy_time = time.time() - start_time
                
                results[f'size_{size}x{size}'] = {
                    'allocation_time_ms': alloc_time * 1000,
                    'copy_time_ms': copy_time * 1000
                }
                
            except Exception as e:
                self.log(f"TensorFlow内存测试失败 (size={size}): {str(e)}", 'ERROR')
        
        return results
    
    def run_all_benchmarks(self) -> Dict:
        """运行所有基准测试"""
        self.log("开始AI基准测试套件")
        
        # 获取系统信息
        system_info = self.get_system_info()
        self.results['system_info'] = system_info
        
        # 运行各项测试
        if TORCH_AVAILABLE:
            self.results['pytorch'] = self.pytorch_benchmark()
        
        if TF_AVAILABLE:
            self.results['tensorflow'] = self.tensorflow_benchmark()
        
        if ONNX_AVAILABLE:
            self.results['onnx'] = self.onnx_benchmark()
        
        # 内存测试
        self.results['memory'] = self.memory_benchmark()
        
        # 保存结果
        self.save_results()
        
        return self.results
    
    def save_results(self):
        """保存测试结果"""
        # 保存JSON格式结果
        json_file = os.path.join(self.output_dir, 'benchmark_results.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self.generate_markdown_report()
        
        self.log(f"测试结果已保存到: {self.output_dir}")
    
    def generate_markdown_report(self):
        """生成Markdown格式的测试报告"""
        report_file = os.path.join(self.output_dir, 'benchmark_report.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# AI基准测试报告\n\n")
            
            # 系统信息
            if 'system_info' in self.results:
                f.write("## 系统信息\n\n")
                system_info = self.results['system_info']
                f.write(f"- 测试时间: {system_info.get('timestamp', 'N/A')}\n")
                f.write(f"- Python版本: {system_info.get('python_version', 'N/A')}\n")
                f.write(f"- 平台: {system_info.get('platform', 'N/A')}\n")
                
                if 'gpus' in system_info:
                    f.write("- GPU信息:\n")
                    for i, gpu in enumerate(system_info['gpus']):
                        f.write(f"  - GPU {i}: {gpu.get('name', 'N/A')} ({gpu.get('memory', 'N/A')})\n")
                
                if 'pytorch_version' in system_info:
                    f.write(f"- PyTorch版本: {system_info['pytorch_version']}\n")
                
                if 'tensorflow_version' in system_info:
                    f.write(f"- TensorFlow版本: {system_info['tensorflow_version']}\n")
                
                f.write("\n")
            
            # PyTorch结果
            if 'pytorch' in self.results:
                f.write("## PyTorch基准测试结果\n\n")
                self._write_pytorch_results(f, self.results['pytorch'])
            
            # TensorFlow结果
            if 'tensorflow' in self.results:
                f.write("## TensorFlow基准测试结果\n\n")
                self._write_tensorflow_results(f, self.results['tensorflow'])
            
            # ONNX结果
            if 'onnx' in self.results:
                f.write("## ONNX Runtime基准测试结果\n\n")
                self._write_onnx_results(f, self.results['onnx'])
            
            # 内存测试结果
            if 'memory' in self.results:
                f.write("## 内存性能测试结果\n\n")
                self._write_memory_results(f, self.results['memory'])
    
    def _write_pytorch_results(self, f, results):
        """写入PyTorch测试结果"""
        if 'classification' in results:
            f.write("### 图像分类模型性能\n\n")
            f.write("| 模型 | 批次大小 | 平均时间(ms) | 吞吐量(samples/s) |\n")
            f.write("|------|----------|--------------|------------------|\n")
            
            for model_name, model_results in results['classification'].items():
                if 'error' not in model_results:
                    for batch_key, batch_results in model_results.items():
                        batch_size = batch_key.replace('batch_', '')
                        avg_time = batch_results.get('avg_time_ms', 0)
                        throughput = batch_results.get('throughput_samples_per_sec', 0)
                        f.write(f"| {model_name} | {batch_size} | {avg_time:.2f} | {throughput:.2f} |\n")
            f.write("\n")
        
        if 'compute' in results:
            f.write("### 计算性能\n\n")
            f.write("| 矩阵大小 | 平均时间(ms) | GFLOPS |\n")
            f.write("|----------|--------------|--------|\n")
            
            for test_name, test_results in results['compute'].items():
                if 'matmul_' in test_name:
                    size = test_name.replace('matmul_', '')
                    avg_time = test_results.get('avg_time_ms', 0)
                    gflops = test_results.get('gflops', 0)
                    f.write(f"| {size} | {avg_time:.2f} | {gflops:.2f} |\n")
            f.write("\n")
    
    def _write_tensorflow_results(self, f, results):
        """写入TensorFlow测试结果"""
        if 'classification' in results:
            f.write("### 图像分类模型性能\n\n")
            f.write("| 模型 | 批次大小 | 平均时间(ms) | 吞吐量(samples/s) |\n")
            f.write("|------|----------|--------------|------------------|\n")
            
            for model_name, model_results in results['classification'].items():
                if 'error' not in model_results:
                    for batch_key, batch_results in model_results.items():
                        batch_size = batch_key.replace('batch_', '')
                        avg_time = batch_results.get('avg_time_ms', 0)
                        throughput = batch_results.get('throughput_samples_per_sec', 0)
                        f.write(f"| {model_name} | {batch_size} | {avg_time:.2f} | {throughput:.2f} |\n")
            f.write("\n")
    
    def _write_onnx_results(self, f, results):
        """写入ONNX测试结果"""
        if 'error' not in results:
            f.write("### ONNX Runtime性能\n\n")
            f.write("| 批次大小 | 平均时间(ms) | 吞吐量(samples/s) |\n")
            f.write("|----------|--------------|------------------|\n")
            
            for batch_key, batch_results in results.items():
                if batch_key.startswith('batch_'):
                    batch_size = batch_key.replace('batch_', '')
                    avg_time = batch_results.get('avg_time_ms', 0)
                    throughput = batch_results.get('throughput_samples_per_sec', 0)
                    f.write(f"| {batch_size} | {avg_time:.2f} | {throughput:.2f} |\n")
            f.write("\n")
    
    def _write_memory_results(self, f, results):
        """写入内存测试结果"""
        for framework, framework_results in results.items():
            f.write(f"### {framework.upper()}内存性能\n\n")
            f.write("| 矩阵大小 | 分配时间(ms) | 复制时间(ms) |\n")
            f.write("|----------|--------------|--------------|\n")
            
            for size_key, size_results in framework_results.items():
                if size_key.startswith('size_'):
                    size = size_key.replace('size_', '')
                    alloc_time = size_results.get('allocation_time_ms', 0)
                    copy_time = size_results.get('copy_time_ms', 0)
                    f.write(f"| {size} | {alloc_time:.2f} | {copy_time:.2f} |\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description='AI基准测试工具')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'gpu'],
                       help='计算设备')
    parser.add_argument('--framework', type=str, nargs='+', 
                       choices=['pytorch', 'tensorflow', 'onnx', 'all'],
                       default=['all'], help='要测试的框架')
    parser.add_argument('--test-type', type=str, nargs='+',
                       choices=['classification', 'nlp', 'compute', 'memory', 'all'],
                       default=['all'], help='要运行的测试类型')
    
    args = parser.parse_args()
    
    # 创建基准测试实例
    benchmark = AIBenchmark(output_dir=args.output_dir, device=args.device)
    
    # 运行测试
    if 'all' in args.framework:
        results = benchmark.run_all_benchmarks()
    else:
        results = {}
        if 'pytorch' in args.framework and TORCH_AVAILABLE:
            results['pytorch'] = benchmark.pytorch_benchmark()
        if 'tensorflow' in args.framework and TF_AVAILABLE:
            results['tensorflow'] = benchmark.tensorflow_benchmark()
        if 'onnx' in args.framework and ONNX_AVAILABLE:
            results['onnx'] = benchmark.onnx_benchmark()
        
        benchmark.results = results
        benchmark.save_results()
    
    print(f"\n🎉 AI基准测试完成！")
    print(f"结果保存在: {benchmark.output_dir}")
    print(f"详细报告: {os.path.join(benchmark.output_dir, 'benchmark_report.md')}")

if __name__ == '__main__':
    main()