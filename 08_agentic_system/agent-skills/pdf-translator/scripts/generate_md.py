import sys
import os
import datetime

def generate_markdown(content, output_path, source_file):
    """
    Saves content to a Markdown file with a header.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"""---
title: Translated Document
source: {source_file}
date: {timestamp}
generated_by: Claude Agent Skill (pdf-translator)
---

"""
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(content)
        print(f"Successfully generated Markdown file at: {output_path}")
    except Exception as e:
        print(f"Error writing file: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_md.py <output_path> <source_filename> [input_text_file]", file=sys.stderr)
        print("If input_text_file is not provided, reads from stdin.", file=sys.stderr)
        sys.exit(1)
    
    output_path = sys.argv[1]
    source_filename = sys.argv[2]
    
    if len(sys.argv) > 3:
        input_file = sys.argv[3]
        if os.path.exists(input_file):
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            print(f"Error: Input file not found: {input_file}", file=sys.stderr)
            sys.exit(1)
    else:
        # Read from stdin
        content = sys.stdin.read()
    
    generate_markdown(content, output_path, source_filename)
