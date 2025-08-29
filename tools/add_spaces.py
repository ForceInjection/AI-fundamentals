#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中英文之间自动添加空格的脚本
用于处理 Markdown 文档，确保中英文之间有适当的空格
"""

import re
import sys
import os
from pathlib import Path

def add_spaces_between_chinese_english(text):
    """
    在中文和英文之间添加空格
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 处理后的文本
    """
    # 保护特殊内容：避免在代码块、链接等地方添加不必要的空格
    protected_blocks = []
    def protect_block(match):
        protected_blocks.append(match.group(0))
        return f"__PROTECTED_BLOCK_{len(protected_blocks)-1}__"
    
    # 按优先级保护内容，确保完整性（在添加空格之前保护）
    # 1. 保护代码块（最高优先级）
    text = re.sub(r'```[\s\S]*?```', protect_block, text)
    # 2. 保护行内代码
    text = re.sub(r'`[^`]*?`', protect_block, text)
    # 3. 保护完整的 Markdown 链接（包括链接文本和 URL）
    text = re.sub(r'\[[^\]]*?\]\([^\)]*?\)', protect_block, text)
    # 4. 保护图片链接
    text = re.sub(r'!\[[^\]]*?\]\([^\)]*?\)', protect_block, text)
    # 5. 保护 HTML 标签及其内容
    text = re.sub(r'<[^>]*?>', protect_block, text)
    # 6. 保护直接的 URL 链接
    text = re.sub(r'https?://[^\s]*', protect_block, text)
    # 7. 保护邮箱地址
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', protect_block, text)
    
    # 中文字符范围（包括中文标点）
    chinese_pattern = r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]'
    # 英文字符范围（字母和数字）
    english_pattern = r'[a-zA-Z0-9]'
    
    # 在中文后面跟英文的地方添加空格（如果没有空格的话）
    text = re.sub(f'({chinese_pattern})(?!\\s)({english_pattern})', r'\1 \2', text)
    
    # 在英文后面跟中文的地方添加空格（如果没有空格的话）
    text = re.sub(f'({english_pattern})(?!\\s)({chinese_pattern})', r'\1 \2', text)
    
    # 恢复保护的内容
    for i, protected_block in enumerate(protected_blocks):
        text = text.replace(f"__PROTECTED_BLOCK_{i}__", protected_block)
    
    return text

def process_file(file_path, backup=True):
    """
    处理单个文件
    
    Args:
        file_path (str): 文件路径
        backup (bool): 是否创建备份文件
        
    Returns:
        bool: 处理是否成功
    """
    try:
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 处理内容
        processed_content = add_spaces_between_chinese_english(content)
        
        # 检查是否有变化
        if content == processed_content:
            print(f"文件 {file_path} 无需修改")
            return True
        
        # 创建备份
        if backup:
            backup_path = f"{file_path}.backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"已创建备份文件: {backup_path}")
        
        # 写入处理后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        print(f"已处理文件: {file_path}")
        return True
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False

def create_test_file():
    """
    创建测试文件
    """
    test_content = """# 测试文件

这是一个测试GPU虚拟化的文档。NVIDIA H100架构支持MIG技术。

## CUDA编程

CUDA是NVIDIA开发的并行计算平台。使用CUDA可以加速AI训练。

- 支持FP16精度
- 内存带宽达到3TB/s
- 支持PCIe 5.0接口

代码示例：
```python
import torch
device = torch.device('cuda')
```

更多信息请参考[NVIDIA官方文档](https://docs.nvidia.com)。
"""
    
    test_file_path = "/Users/wangtianqing/Project/AI-fundermentals/gpu_manager/test_spaces.md"
    with open(test_file_path, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"已创建测试文件: {test_file_path}")
    return test_file_path

def main():
    """
    主函数
    """
    if len(sys.argv) < 2:
        print("用法:")
        print("  python add_spaces.py test          # 创建并处理测试文件")
        print("  python add_spaces.py <file_path>   # 处理指定文件")
        print("  python add_spaces.py current       # 处理当前目录的基础理论篇.md")
        return
    
    command = sys.argv[1]
    
    if command == "test":
        # 创建测试文件
        test_file = create_test_file()
        print("\n处理前的内容:")
        with open(test_file, 'r', encoding='utf-8') as f:
            print(f.read())
        
        # 处理测试文件
        process_file(test_file, backup=True)
        
        print("\n处理后的内容:")
        with open(test_file, 'r', encoding='utf-8') as f:
            print(f.read())
            
    elif command == "current":
        # 处理当前的基础理论篇文档
        current_file = "/Users/wangtianqing/Project/AI-fundermentals/gpu_manager/第一部分：基础理论篇.md"
        if os.path.exists(current_file):
            process_file(current_file, backup=True)
        else:
            print(f"文件不存在: {current_file}")
            
    else:
        # 处理指定文件
        file_path = command
        if os.path.exists(file_path):
            process_file(file_path, backup=True)
        else:
            print(f"文件不存在: {file_path}")

if __name__ == "__main__":
    main()