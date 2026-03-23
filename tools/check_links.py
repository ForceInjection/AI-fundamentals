#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown 链接检查工具
用于检查项目中 Markdown 文件内的本地文件链接和外部 URL 链接是否有效。
支持处理包含空格、URL 编码的文件名，以及 HTML 格式的图片标签。
"""

import os
import re
import urllib.parse
import argparse
import requests
from datetime import datetime

def extract_markdown_links(content):
    """
    提取 Markdown 格式的链接（包括图片链接）和 HTML 图片链接。
    
    Args:
        content (str): Markdown 文件的文本内容
        
    Returns:
        list of tuple: 返回链接元组列表，格式为 [(text, url), ...]
    """
    links = []
    
    # 匹配 Markdown 格式的链接和图片：[text](url) 或 ![alt](url)
    pattern_md = r'(?:!?)\[([^\]]*)\]\('
    
    for match in re.finditer(pattern_md, content):
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
            url_part = content[start:i-1].strip()
            # 处理带有 title 属性的链接，例如: url "title"
            if ' ' in url_part:
                url = url_part.split(' ', 1)[0].strip()
            else:
                url = url_part
            # 去除 URL 两端的尖括号（如果有）
            if url.startswith('<') and url.endswith('>'):
                url = url[1:-1]
            links.append((text.strip(), url))
            
    # 匹配 HTML 格式的图片链接：<img src="url" ... />
    pattern_html_img = r'<img[^>]+src=["\']([^"\']+)["\']'
    for match in re.finditer(pattern_html_img, content, re.IGNORECASE):
        url = match.group(1).strip()
        links.append(('<img src>', url))
    
    return links

def is_local_file(url):
    """
    判断 URL 是否为本地文件链接。
    
    Args:
        url (str): 待判断的 URL
        
    Returns:
        bool: 如果是本地链接则返回 True，否则返回 False
    """
    # 排除以协议开头的外部 URL
    if url.startswith(('http://', 'https://', 'ftp://', 'mailto:', 'tel:')):
        return False
    # 排除当前页面的锚点链接
    if url.startswith('#'):
        return False
    return True

def check_local_file_exists(file_path, base_dir):
    """
    检查本地文件是否存在，处理 URL 编码和锚点。
    
    Args:
        file_path (str): 本地文件相对路径
        base_dir (str): 当前 Markdown 文件所在的目录作为基准路径
        
    Returns:
        tuple: (是否存在(bool), 状态描述(str))
    """
    # 移除文件路径中的锚点
    if '#' in file_path:
        file_path = file_path.split('#')[0]
        
    # 对 URL 编码的路径进行解码（例如将 %20 转回空格）
    decoded_path = urllib.parse.unquote(file_path)
    
    # 构建绝对路径
    full_path = os.path.join(base_dir, decoded_path)
    
    # 检查文件是否存在
    exists = os.path.exists(full_path)
    
    return exists, 'exists' if exists else 'not_found'

def check_external_url(url, timeout=10):
    """
    检查外部 URL 是否可访问。
    
    Args:
        url (str): 外部 URL
        timeout (int): 请求超时时间，默认 10 秒
        
    Returns:
        tuple: (是否有效(bool), HTTP 状态码或错误信息)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        # 首先尝试使用 HEAD 请求以节省带宽
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        if response.status_code >= 400 and response.status_code not in [403, 404, 412, 418]:
            return False, response.status_code
        
        # 针对常见的反爬虫状态码，尝试使用 GET 请求
        if response.status_code in [403, 404, 412, 418]:
            res_get = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
            if res_get.status_code < 400:
                return True, res_get.status_code
            else:
                return False, res_get.status_code
        return response.status_code < 400, response.status_code
    except requests.exceptions.RequestException as e:
        return False, str(e)

def find_markdown_files_in_dir(directory):
    """
    递归查找指定目录下的所有 Markdown 文件。
    
    Args:
        directory (str): 要遍历的根目录
        
    Returns:
        list of str: Markdown 文件的绝对路径列表
    """
    md_files = []
    for root, dirs, files in os.walk(directory):
        # 排除隐藏目录（如 .git）和特定忽略目录（如虚拟环境、node_modules）
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'node_modules']]
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    return md_files

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='检查 Markdown 文件中的链接')
    parser.add_argument('-f', '--file', help='指定要检查的单个 Markdown 文件路径')
    parser.add_argument('-d', '--dir', help='指定要检查的目录，将递归检查该目录及其子目录下的所有 Markdown 文件')
    parser.add_argument('-a', '--all', action='store_true', help='检查整个项目的所有 Markdown 文件（等同于对项目根目录使用 --dir）')
    parser.add_argument('-t', '--type', choices=['local', 'external', 'all'], default='local', help='指定检查的链接类型 (默认: local)')
    return parser.parse_args()

def get_files_to_check(args, base_dir):
    """
    根据用户输入的参数，确定需要检查的 Markdown 文件列表。
    
    Args:
        args (Namespace): 命令行参数对象
        base_dir (str): 项目根目录
        
    Returns:
        list of str: 待检查的文件路径列表
    """
    md_files = []
    
    if args.all:
        print(f"模式：全量检查项目根目录 {base_dir}")
        md_files = find_markdown_files_in_dir(base_dir)
        print(f"找到 {len(md_files)} 个 Markdown 文件。")
        
    elif args.dir:
        dir_path = os.path.abspath(args.dir)
        if not os.path.isdir(dir_path):
            print(f"错误：指定的目录 {dir_path} 不存在。")
            return []
        print(f"模式：检查指定目录及其子目录 {dir_path}")
        md_files = find_markdown_files_in_dir(dir_path)
        print(f"找到 {len(md_files)} 个 Markdown 文件。")
        
    elif args.file:
        file_path = os.path.abspath(args.file)
        if not os.path.isfile(file_path):
            print(f"错误：指定的文件 {file_path} 不存在。")
            return []
        print(f"模式：检查指定文件 {file_path}")
        md_files.append(file_path)
        
    else:
        # 默认回退行为：检查项目根目录的 README.md
        readme_path = os.path.join(base_dir, 'README.md')
        print(f"未指定目标参数，默认检查项目 README.md: {readme_path}")
        if os.path.isfile(readme_path):
            md_files.append(readme_path)
        else:
            print(f"错误：默认文件 {readme_path} 不存在。")
            
    return md_files

def generate_report(md_files_count, local_stats, external_stats, base_dir):
    """
    生成并保存链接检查报告。
    
    Args:
        md_files_count (int): 检查的文件总数
        local_stats (dict): 本地链接统计字典
        external_stats (dict): 外部链接统计字典
        base_dir (str): 项目根目录，用于计算相对路径
    """
    report = []
    report.append("Markdown 链接检查报告")
    report.append("=" * 50)
    report.append(f"\n检查时间: {datetime.now()}")
    report.append(f"检查文件数: {md_files_count}")
    
    # 本地链接统计信息
    report.append(f"\n本地文件链接统计:")
    report.append(f"  有效: {local_stats['valid']}")
    report.append(f"  无效: {len(local_stats['invalid'])}")
    report.append(f"  Submodule未初始化: {len(local_stats['submodule'])}")
    
    # 外部链接统计信息
    report.append(f"\n外部URL链接统计:")
    report.append(f"  有效: {external_stats['valid']}")
    report.append(f"  无效: {len(external_stats['invalid'])}")
    
    # 详细列出无效链接
    if local_stats['invalid']:
        report.append(f"\n无效的本地文件链接 ({len(local_stats['invalid'])}个):")
        for file_path, text, url in local_stats['invalid']:
            rel_file = os.path.relpath(file_path, base_dir)
            report.append(f"  - [{rel_file}] 文本: '{text}' -> 链接: '{url}'")
            
    if local_stats['submodule']:
        report.append(f"\nSubmodule未初始化的链接 ({len(local_stats['submodule'])}个):")
        for file_path, text, url in local_stats['submodule']:
            rel_file = os.path.relpath(file_path, base_dir)
            report.append(f"  - [{rel_file}] 文本: '{text}' -> 链接: '{url}'")
            
    if external_stats['invalid']:
        report.append(f"\n无效的外部URL链接 ({len(external_stats['invalid'])}个):")
        for file_path, text, url, status in external_stats['invalid']:
            rel_file = os.path.relpath(file_path, base_dir)
            report.append(f"  - [{rel_file}] 文本: '{text}' -> 链接: '{url}' - 状态: {status}")
            
    # 将报告写入文件
    report_content = '\n'.join(report)
    report_path = os.path.join(base_dir, 'link_check_report_v2.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"\n检查完成！详细报告已保存到: {report_path}")
    print(f"\n检查总结:")
    print(f"- 检查文件数: {md_files_count}")
    print(f"- 本地文件链接: {local_stats['valid']} 有效, {len(local_stats['invalid'])} 无效, {len(local_stats['submodule'])} submodule未初始化")
    print(f"- 外部URL链接: {external_stats['valid']} 有效, {len(external_stats['invalid'])} 无效")

def main():
    args = parse_arguments()
    
    # 自动获取脚本所在目录的上级目录作为项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    
    # 获取待检查的文件列表
    md_files = get_files_to_check(args, base_dir)
    if not md_files:
        return
        
    # 初始化统计容器
    local_stats = {'valid': 0, 'invalid': [], 'submodule': []}
    external_stats = {'valid': 0, 'invalid': []}
    
    # 遍历检查每个 Markdown 文件
    for file_path in md_files:
        print(f"\n正在检查文件: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"错误：读取文件失败 {file_path}")
            continue
            
        # 提取文件内所有的链接
        links = extract_markdown_links(content)
        
        # 将链接按本地和外部分类
        local_links = []
        external_links = []
        for text, url in links:
            if is_local_file(url):
                local_links.append((text, url))
            else:
                external_links.append((text, url))
                
        # 检查本地链接
        if args.type in ['local', 'all']:
            print(f"  检查 {len(local_links)} 个本地文件链接...")
            file_dir = os.path.dirname(file_path) # 本地链接相对当前文件所在目录解析
            for text, url in local_links:
                exists, status = check_local_file_exists(url, file_dir)
                if status == 'submodule_not_initialized':
                    local_stats['submodule'].append((file_path, text, url))
                elif exists:
                    local_stats['valid'] += 1
                else:
                    local_stats['invalid'].append((file_path, text, url))
                    
        # 检查外部链接
        if args.type in ['external', 'all']:
            print(f"  检查 {len(external_links)} 个外部URL链接...")
            for text, url in external_links:
                is_valid, status = check_external_url(url)
                if is_valid:
                    external_stats['valid'] += 1
                else:
                    external_stats['invalid'].append((file_path, text, url, status))
                    
    # 汇总并生成最终报告
    generate_report(len(md_files), local_stats, external_stats, base_dir)

if __name__ == '__main__':
    main()