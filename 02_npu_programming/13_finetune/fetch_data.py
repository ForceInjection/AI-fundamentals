"""
获取 LoRA 微调训练数据

从多个数据源下载中文技术文本：
1. ModelScope smoltalk-chinese（通用中文对话，约 70 万条）
2. 昇腾社区技术文章（域内技术文档）
3. 本地 Ascend 学习文档

用法:
  python3 fetch_data.py --output ./train-data/ --max-samples 5000
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def fetch_wikipedia(output_dir: str, max_samples: int = 5000):
    """从 HuggingFace 下载中文维基百科"""
    print("下载中文维基百科 (20231101.zh)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.zh",
                          split="train", streaming=True)

        output_path = os.path.join(output_dir, "wiki-zh.txt")
        count = 0
        total_chars = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in ds:
                if count >= max_samples:
                    break
                text = sample.get("text", "").strip()
                if len(text) > 200:
                    f.write(text + "\n\n")
                    count += 1
                    total_chars += len(text)
                    if count % 1000 == 0:
                        print(f"  已处理: {count}/{max_samples} ({total_chars/1024/1024:.1f} MB)")

        size = os.path.getsize(output_path) / 1024 / 1024
        print(f"  保存 {count} 条, {size:.1f} MB")
        return count
    except ImportError:
        print("  datasets 未安装，跳过")
        return 0
    except Exception as e:
        print(f"  下载失败: {e}")
        return 0


def fetch_ascend_tech_articles(output_dir: str):
    """从昇腾社区获取技术文章（通过已知文章 URL 列表）"""
    print("下载昇腾社区技术文章...")
    import urllib.request

    # 已知的昇腾技术文章 URL（从搜索和社区首页收集）
    article_urls = [
        "https://www.hiascend.com/developer/techArticles/20240914-1",  # CANN 架构
        "https://www.hiascend.com/developer/techArticles/20231106-1",  # TensorFlow 训练
        "https://www.hiascend.com/developer/techArticles/20230817-1",  # AscendCL 学习资源
        "https://www.hiascend.com/developer/techArticles/20250303-1",  # AOL 算子加速库
        "https://www.hiascend.com/developer/techArticles/20241201-1",  # MindIE 推理
        "https://www.hiascend.com/developer/techArticles/20241015-1",  # 大模型部署
        "https://www.hiascend.com/developer/techArticles/20240820-1",  # 算子开发
        "https://www.hiascend.com/developer/techArticles/20240610-1",  # 性能调优
        "https://www.hiascend.com/developer/techArticles/20240405-1",  # 模型迁移
        "https://www.hiascend.com/developer/techArticles/20240120-1",  # 混合精度训练
        "https://www.hiascend.com/developer/techArticles/20231201-1",  # FlashAttention
        "https://www.hiascend.com/developer/techArticles/20230901-1",  # HCCL 通信
    ]

    output_path = os.path.join(output_dir, "ascend-articles.txt")
    count = 0
    total_chars = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for url in article_urls:
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; AscendDataBot/1.0)"
                })
                with urllib.request.urlopen(req, timeout=15) as resp:
                    html = resp.read().decode("utf-8", errors="replace")

                # 简单提取纯文本：去除 script/style 标签和 HTML 标记
                html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL)
                html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL)
                text = re.sub(r"<[^>]+>", " ", html)
                text = re.sub(r"\s+", " ", text).strip()

                # 找到正文区域（title 标记后的内容通常更纯净）
                if "技术干货" in text:
                    text = text[text.find("技术干货"):]

                # 去除 CSS/JS 碎片
                text = re.sub(r"\{[^}]*\}", "", text)
                text = re.sub(r"@media[^{]*\{[^}]*\}", "", text)
                text = re.sub(r"\.\w+-\w+\{[^}]*\}", "", text)
                text = re.sub(r"\s+", " ", text).strip()

                if len(text) > 200:  # 有效文章
                    f.write(text + "\n\n")
                    count += 1
                    total_chars += len(text)
                    print(f"  [{count}/{len(article_urls)}] {url.split('/')[-1]}: {len(text)} 字符")
                else:
                    print(f"  [跳过] {url.split('/')[-1]}: 内容太短 ({len(text)} 字符)")
            except Exception as e:
                print(f"  [失败] {url}: {e}")

    print(f"  下载 {count}/{len(article_urls)} 篇, 总 {total_chars:,} 字符")
    return count, total_chars


def include_local_docs(output_dir: str, docs_dir: str):
    """包含本地 Ascend 学习文档"""
    print("包含本地文档...")
    output_path = os.path.join(output_dir, "local-docs.txt")
    count = 0
    total_chars = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for md_file in sorted(Path(docs_dir).glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
                if len(content) > 100:
                    # 去除 YAML frontmatter
                    content = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)
                    f.write(f"# {md_file.stem}\n\n{content}\n\n")
                    count += 1
                    total_chars += len(content)
            except Exception as e:
                print(f"  读取 {md_file} 失败: {e}")

    print(f"  包含 {count} 篇, 总 {total_chars:,} 字符")
    return count, total_chars


def main():
    parser = argparse.ArgumentParser(description="获取 LoRA 微调训练数据")
    parser.add_argument("--output", default="./train-data/", help="输出目录")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="smoltalk 最大样本数")
    parser.add_argument("--local-docs", default="./train-docs/",
                        help="本地文档目录")
    parser.add_argument("--skip-wiki", action="store_true",
                        help="跳过维基百科下载")
    parser.add_argument("--skip-articles", action="store_true",
                        help="跳过昇腾文章下载")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  LoRA 训练数据获取")
    print("=" * 60)
    print()

    total_chars = 0

    # 1. 昇腾社区技术文章（域内数据）
    if not args.skip_articles:
        count, chars = fetch_ascend_tech_articles(args.output)
        total_chars += chars
        print()

    # 2. 本地文档
    _, local_chars = include_local_docs(args.output, args.local_docs)
    total_chars += local_chars
    print()

    # 3. 中文维基百科（保持常识能力）
    if not args.skip_wiki:
        wiki_count = fetch_wikipedia(args.output, args.max_samples)
        wiki_path = os.path.join(args.output, "wiki-zh.txt")
        if os.path.exists(wiki_path):
            chars = os.path.getsize(wiki_path)
            total_chars += chars
            print(f"  维基百科: ~{chars/1024/1024:.1f} MB")
        print()

    print("=" * 60)
    print(f"  总计: ~{total_chars/1024/1024:.1f} MB 文本")
    print(f"  文件保存在: {os.path.abspath(args.output)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
