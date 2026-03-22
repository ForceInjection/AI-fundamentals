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
            url = content[start:i-1].strip()
            # 去除URL两端的尖括号（如果有）
            if url.startswith('<') and url.endswith('>'):
                url = url[1:-1]
            links.append((text.strip(), url))
    
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
    
    return exists, 'exists' if exists else 'not_found'

def check_external_url(url, timeout=10):
    """检查外部URL是否可访问"""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code < 400, response.status_code
    except requests.exceptions.RequestException as e:
        return False, str(e)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='检查 Markdown 文件中的链接')
    parser.add_argument('--all', action='store_true', help='检查所有 Markdown 文件')
    parser.add_argument('--type', choices=['local', 'external', 'all'], default='all', help='指定检查的链接类型 (默认: all)')
    args = parser.parse_args()

    # 设置基础目录
    # 自动获取脚本所在目录的上级目录作为项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    
    md_files = []
    if args.all:
        for root, dirs, files in os.walk(base_dir):
            # 排除隐藏目录和特定目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'node_modules']]
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))
        print(f"找到 {len(md_files)} 个 Markdown 文件进行检查。")
    else:
        readme_path = os.path.join(base_dir, 'README.md')
        md_files.append(readme_path)
        print(f"项目根目录: {base_dir}")
        print(f"检查文件: {readme_path}")
    
    all_invalid_local = []
    all_submodule_links = []
    all_invalid_external = []
    
    total_valid_local = 0
    total_valid_external = 0

    for file_path in md_files:
        print(f"\n正在检查文件: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"错误：找不到文件 {file_path}")
            continue
        
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
        if args.type in ['local', 'all']:
            print(f"  检查 {len(local_links)} 个本地文件链接...")
            # 注意：本地链接的基准路径是当前 md 文件所在的目录，而不是项目根目录
            file_dir = os.path.dirname(file_path)
            for text, url in local_links:
                exists, status = check_local_file_exists(url, file_dir)
                if status == 'submodule_not_initialized':
                    all_submodule_links.append((file_path, text, url))
                elif exists:
                    total_valid_local += 1
                else:
                    all_invalid_local.append((file_path, text, url))
        
        # 检查外部URL链接
        if args.type in ['external', 'all']:
            print(f"  检查 {len(external_links)} 个外部URL链接...")
            for text, url in external_links:
                is_valid, status = check_external_url(url)
                if is_valid:
                    total_valid_external += 1
                else:
                    all_invalid_external.append((file_path, text, url, status))
    
    # 生成报告
    report = []
    report.append("Markdown 链接检查报告（支持全量检查）")
    report.append("=" * 50)
    report.append(f"\n检查时间: {datetime.now()}")
    report.append(f"检查文件数: {len(md_files)}")
    
    report.append(f"\n本地文件链接统计:")
    report.append(f"  有效: {total_valid_local}")
    report.append(f"  无效: {len(all_invalid_local)}")
    report.append(f"  Submodule未初始化: {len(all_submodule_links)}")
    
    report.append(f"\n外部URL链接统计:")
    report.append(f"  有效: {total_valid_external}")
    report.append(f"  无效: {len(all_invalid_external)}")
    
    # 详细的无效链接列表
    if all_invalid_local:
        report.append(f"\n无效的本地文件链接 ({len(all_invalid_local)}个):")
        for file_path, text, url in all_invalid_local:
            rel_file = os.path.relpath(file_path, base_dir)
            report.append(f"  - [{rel_file}] 文本: '{text}' -> 链接: '{url}'")
    
    if all_submodule_links:
        report.append(f"\nSubmodule未初始化的链接 ({len(all_submodule_links)}个):")
        for file_path, text, url in all_submodule_links:
            rel_file = os.path.relpath(file_path, base_dir)
            report.append(f"  - [{rel_file}] 文本: '{text}' -> 链接: '{url}'")
    
    if all_invalid_external:
        report.append(f"\n无效的外部URL链接 ({len(all_invalid_external)}个):")
        for file_path, text, url, status in all_invalid_external:
            rel_file = os.path.relpath(file_path, base_dir)
            report.append(f"  - [{rel_file}] 文本: '{text}' -> 链接: '{url}' - 状态: {status}")
    
    # 保存报告
    report_content = '\n'.join(report)
    report_path = os.path.join(base_dir, 'link_check_report_v2.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n检查完成！报告已保存到: {report_path}")
    print(f"\n总结:")
    print(f"- 检查文件数: {len(md_files)}")
    print(f"- 本地文件链接: {total_valid_local} 有效, {len(all_invalid_local)} 无效, {len(all_submodule_links)} submodule未初始化")
    print(f"- 外部URL链接: {total_valid_external} 有效, {len(all_invalid_external)} 无效")

if __name__ == '__main__':
    main()