#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arXiv 论文元数据检查工具
用于通过 arXiv API 批量获取论文的标题、作者和发布年份等信息。
"""

import urllib.request
import xml.etree.ElementTree as ET
import argparse
import sys

def fetch_arxiv_metadata(ids, output_format="text"):
    """
    请求 arXiv API 并解析返回的 XML 元数据
    """
    # 将列表拼接成以逗号分隔的字符串
    id_str = ','.join(ids)
    url = f'http://export.arxiv.org/api/query?id_list={id_str}'
    
    try:
        with urllib.request.urlopen(url) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entries = root.findall('atom:entry', ns)
            if not entries:
                print("未找到对应的论文信息，请检查 arXiv ID 是否正确。")
                return

            results = []
            for entry in entries:
                # 提取并清理字段
                arxiv_id_full = entry.find('atom:id', ns).text
                arxiv_id = arxiv_id_full.split('/abs/')[-1][:10]
                title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
                while '  ' in title: 
                    title = title.replace('  ', ' ')
                
                published_elem = entry.find('atom:published', ns)
                year = published_elem.text[:4] if published_elem is not None else 'Unknown'
                
                authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
                first_author = authors[0] if authors else 'Unknown'
                
                results.append({
                    'id': arxiv_id,
                    'url': arxiv_id_full,
                    'first_author': first_author,
                    'authors': authors,
                    'year': year,
                    'title': title
                })
                
            if output_format == "json":
                import json
                print(json.dumps(results, indent=2, ensure_ascii=False))
            elif output_format == "markdown":
                for r in results:
                    print(f"- [{r['title']}]({r['url']}) ({r['year']}) by {r['first_author']} et al.")
            elif output_format == "ieee":
                for r in results:
                    author_str = f"{r['first_author']} et al." if len(r['authors']) > 1 else r['first_author']
                    print(f"{author_str}, \"{r['title']},\" arXiv preprint arXiv:{r['id']}, {r['year']}.")
            else:
                for r in results:
                    print(f"ID: {r['id']} | Author: {r['first_author']} | Year: {r['year']} | Title: {r['title']}")
    
    except urllib.error.URLError as e:
        print(f"网络请求失败: {e.reason}", file=sys.stderr)
    except ET.ParseError as e:
        print(f"XML 解析失败: {e}", file=sys.stderr)
    except Exception as e:
        print(f"发生未知错误: {e}", file=sys.stderr)

def main():
    # 使用 argparse 添加参数支持和帮助信息
    parser = argparse.ArgumentParser(
        description="arXiv 论文元数据批量获取工具",
        epilog="示例: python3 tools/arxiv_metadata_fetcher.py -i 2504.15965 2404.13501 -f markdown"
    )
    
    parser.add_argument(
        "-i", "--ids",
        nargs="+",
        required=True,
        help="一个或多个 arXiv 论文 ID，用空格分隔 (例如: 2504.15965)"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["text", "json", "markdown", "ieee"],
        default="text",
        help="输出格式：text (默认), json, markdown, ieee"
    )
    
    args = parser.parse_args()
    
    # 过滤掉可能的空字符串
    valid_ids = [aid.strip() for aid in args.ids if aid.strip()]
    if not valid_ids:
        print("错误: 请提供有效的 arXiv ID。", file=sys.stderr)
        sys.exit(1)
        
    fetch_arxiv_metadata(valid_ids, output_format=args.format)

if __name__ == "__main__":
    main()
