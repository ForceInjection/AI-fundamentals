#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中英文之间自动添加空格的脚本
用于处理 Markdown 文档，确保中英文之间有适当的空格
"""

import re
import sys
import os

def clean_improper_spaces(text):
    """
    清理不规范的空格 - 专注于中英文空格问题
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 清理后的文本
    """
    # 中文标点符号范围
    chinese_punctuation = r'[\u3000-\u303f\uff00-\uffef]'
    
    # 1. 移除中文标点符号前面的空格
    text = re.sub(f'\\s+({chinese_punctuation})', r'\1', text)
    
    # 2. 移除参考文献标记前面的空格
    text = re.sub(r'\\s+(\[[0-9]+\])', r'\1', text)
    
    # 3. 移除中文顿号前后的空格
    text = re.sub(r'\\s*、\\s*', '、', text)
    
    # 4. 移除其他中文标点符号前后的空格
    chinese_punct_chars = ['，', '。', '；', '：', '？', '！', '「', '」', '《', '》']
    for punct in chinese_punct_chars:
        text = re.sub(f'\\s*{re.escape(punct)}\\s*', punct, text)
    
    return text

def add_spaces_between_chinese_english(text):
    """
    管理中文和英文之间的空格（添加必要空格并清理不规范空格）
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 处理后的文本
    """
    
    # 逐行处理，保持原有的行结构
    lines = text.split('\n')
    processed_lines = []
    
    # 标记是否在多行代码块中
    in_code_block = False
    
    for line in lines:
        # 检查是否进入或退出代码块
        stripped_line = line.strip()
        
        # 检查是否以 ``` 开头（可能前面有空格），表示代码块开始
        if stripped_line.startswith('```'):
            in_code_block = not in_code_block
            processed_lines.append(line)  # 直接添加代码块标记行
            continue
        
        # 如果在代码块中，直接跳过处理
        if in_code_block:
            processed_lines.append(line)
            continue
            
        # 跳过空行
        if line.strip() == '':
            processed_lines.append(line)
            continue
            
        # 保护特殊内容：避免在代码块和链接中添加不必要的空格
        protected_blocks = []
        def protect_block(match):
            protected_blocks.append(match.group(0))
            return f"__PROTECTED_BLOCK_{len(protected_blocks)-1}__"
        
        # 只保护核心内容，避免过度保护
        # 1. 保护行内代码
        protected_line = re.sub(r'`[^`]*?`', protect_block, line)
        # 2. 保护 Markdown 链接和图片
        protected_line = re.sub(r'\[[^\]]*?\]\([^\)]*?\)', protect_block, protected_line)
        protected_line = re.sub(r'!\[[^\]\]]*?\]\([^\)]*?\)', protect_block, protected_line)
        
        # 中文字符范围（不包括中文标点）
        chinese_chars = r'[\u4e00-\u9fff]'
        # 中文标点符号（逗号、句号等）
        chinese_punctuation = r'[\u3000-\u303f\uff00-\uffef]'
        # 英文字符范围（字母和数字）
        english_pattern = r'[a-zA-Z0-9]'
        
        # 在中文后面跟英文的地方添加空格（如果没有空格的话）
        # 只对中文字符（非标点）后面跟英文的情况添加空格
        protected_line = re.sub(f'({chinese_chars})(?!\\s)({english_pattern})', r'\1 \2', protected_line)
        
        # 在英文后面跟中文的地方添加空格（如果没有空格的话）
        # 只对英文后面跟中文字符（非标点）的情况添加空格
        protected_line = re.sub(f'({english_pattern})(?!\\s)({chinese_chars})', r'\1 \2', protected_line)
        
        # 修复中文标点和英文之间不必要添加空格的问题
        # 中文标点后面跟英文时，不应该添加空格
        protected_line = re.sub(f'({chinese_punctuation})\\s+({english_pattern})', r'\1\2', protected_line)
        protected_line = re.sub(f'({chinese_punctuation})({english_pattern})', r'\1\2', protected_line)
        
        # 恢复保护的内容（反向恢复，避免嵌套占位符问题）
        for i in range(len(protected_blocks)-1, -1, -1):
            protected_line = protected_line.replace(f"__PROTECTED_BLOCK_{i}__", protected_blocks[i])
        
        # 清理不规范的空格
        processed_line = clean_improper_spaces(protected_line)
        processed_lines.append(processed_line)
    
    # 重新组合行，保持原有的换行符
    return '\n'.join(processed_lines)

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