#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更准确的 README.md 链接检查工具
考虑文件名空格、URL编码和submodule情况
"""

import os
import re
import urllib.parse
import requests
from datetime import datetime
from pathlib import Path

def extract_markdown_links(content):
    """提取Markdown格式的链接"""
    # 匹配 [text](url) 格式的链接，支持URL中包含括号
    links = []
    pattern = r'\[([^\]]+)\]\('
    
    for match in re.finditer(pattern, content):
        start = match.end()
        text = match.group(1)
        
        # 找到匹配的右括号，考虑嵌套括号
        paren_count = 1
        i = start
        while i < len(content) and paren_count > 0:
            if content[i] == '(':
                paren_count += 1
            elif content[i] == ')':
                paren_count -= 1
            i += 1
        
        if paren_count == 0:
            url = content[start:i-1]
            links.append((text.strip(), url.strip()))
    
    return links

def is_local_file(url):
    """判断是否为本地文件链接"""
    # 排除以协议开头的URL
    if url.startswith(('http://', 'https://', 'ftp://', 'mailto:', 'tel:')):
        return False
    # 排除锚点链接
    if url.startswith('#'):
        return False
    return True

def check_local_file_exists(file_path, base_dir):
    """检查本地文件是否存在，处理URL编码"""
    # URL解码
    decoded_path = urllib.parse.unquote(file_path)
    
    # 构建完整路径
    full_path = os.path.join(base_dir, decoded_path)
    
    # 检查文件是否存在
    exists = os.path.exists(full_path)
    
    # 如果不存在，检查是否是submodule目录
    if not exists:
        # 检查是否是submodule路径
        path_parts = decoded_path.split('/')
        if len(path_parts) > 0:
            first_dir = path_parts[0]
            if first_dir in ['AISystem', 'hands-on-ML']:
                return False, 'submodule_not_initialized'
    
    return exists, 'exists' if exists else 'not_found'

def check_external_url(url, timeout=10):
    """检查外部URL是否可访问"""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code < 400, response.status_code
    except requests.exceptions.RequestException as e:
        return False, str(e)

def main():
    # 设置基础目录
    base_dir = '/Users/wangtianqing/Project/AI-fundermentals'
    readme_path = os.path.join(base_dir, 'README.md')
    
    # 读取README.md文件
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误：找不到文件 {readme_path}")
        return
    
    # 提取所有链接
    links = extract_markdown_links(content)
    
    # 分类链接
    local_links = []
    external_links = []
    
    for text, url in links:
        if is_local_file(url):
            local_links.append((text, url))
        else:
            external_links.append((text, url))
    
    # 检查本地文件链接
    valid_local = []
    invalid_local = []
    submodule_links = []
    
    print(f"检查 {len(local_links)} 个本地文件链接...")
    for text, url in local_links:
        exists, status = check_local_file_exists(url, base_dir)
        if status == 'submodule_not_initialized':
            submodule_links.append((text, url))
        elif exists:
            valid_local.append((text, url))
        else:
            invalid_local.append((text, url))
    
    # 检查外部URL链接
    valid_external = []
    invalid_external = []
    
    print(f"检查 {len(external_links)} 个外部URL链接...")
    for text, url in external_links:
        is_valid, status = check_external_url(url)
        if is_valid:
            valid_external.append((text, url, status))
        else:
            invalid_external.append((text, url, status))
    
    # 生成报告
    report = []
    report.append("README.md 链接检查报告（修正版）")
    report.append("=" * 50)
    report.append(f"\n检查时间: {datetime.now()}")
    
    report.append(f"\n本地文件链接统计:")
    report.append(f"  有效: {len(valid_local)}")
    report.append(f"  无效: {len(invalid_local)}")
    report.append(f"  Submodule未初始化: {len(submodule_links)}")
    
    report.append(f"\n外部URL链接统计:")
    report.append(f"  有效: {len(valid_external)}")
    report.append(f"  无效: {len(invalid_external)}")
    
    # 详细的无效链接列表
    if invalid_local:
        report.append(f"\n无效的本地文件链接 ({len(invalid_local)}个):")
        for text, url in invalid_local:
            report.append(f"  - [{text}]({url})")
    
    if submodule_links:
        report.append(f"\nSubmodule未初始化的链接 ({len(submodule_links)}个):")
        for text, url in submodule_links:
            report.append(f"  - [{text}]({url})")
    
    if invalid_external:
        report.append(f"\n无效的外部URL链接 ({len(invalid_external)}个):")
        for text, url, status in invalid_external:
            report.append(f"  - [{text}]({url}) - 状态: {status}")
    
    # 路径问题分析
    report.append(f"\n路径问题分析:")
    
    # 检查常见的路径错误
    path_issues = []
    
    # 检查 coding vs ai_coding
    coding_links = [url for text, url in invalid_local if url.startswith('coding/')]
    if coding_links:
        path_issues.append(f"发现 {len(coding_links)} 个 'coding/' 路径，应该是 'ai_coding/'")
    
    # 检查 context vs agent/context
    context_links = [url for text, url in invalid_local if url.startswith('context/')]
    if context_links:
        path_issues.append(f"发现 {len(context_links)} 个 'context/' 路径，应该是 'agent/context/'")
    
    # 检查 memory vs agent/memory
    memory_links = [url for text, url in invalid_local if url.startswith('memory/')]
    if memory_links:
        path_issues.append(f"发现 {len(memory_links)} 个 'memory/' 路径，应该是 'agent/memory/'")
    
    if path_issues:
        for issue in path_issues:
            report.append(f"  - {issue}")
    else:
        report.append("  - 未发现明显的路径问题")
    
    # 保存报告
    report_content = '\n'.join(report)
    report_path = os.path.join(base_dir, 'link_check_report_v2.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n检查完成！报告已保存到: {report_path}")
    print(f"\n总结:")
    print(f"- 本地文件链接: {len(valid_local)} 有效, {len(invalid_local)} 无效, {len(submodule_links)} submodule未初始化")
    print(f"- 外部URL链接: {len(valid_external)} 有效, {len(invalid_external)} 无效")
    
    if path_issues:
        print(f"\n发现的路径问题:")
        for issue in path_issues:
            print(f"  - {issue}")

if __name__ == '__main__':
    main()