import argparse
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
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(content)
        print(f"Successfully generated Markdown file at: {output_path}")
    except Exception as e:
        print(f"Error writing file: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Markdown file with metadata.")
    parser.add_argument("output_path", help="Path to save the generated Markdown file.")
    parser.add_argument("source_filename", help="Name of the original source file (for metadata).")
    parser.add_argument("input_text_file", nargs="?", help="Path to input text file. If omitted, reads from stdin.")
    
    args = parser.parse_args()
    
    content = ""
    if args.input_text_file:
        if os.path.exists(args.input_text_file):
            try:
                with open(args.input_text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading input file: {str(e)}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Error: Input file not found: {args.input_text_file}", file=sys.stderr)
            sys.exit(1)
    else:
        # Check if stdin has data
        if not sys.stdin.isatty():
            content = sys.stdin.read()
        else:
            print("Error: No input provided. Pipe text to stdin or provide an input file.", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
            
    generate_markdown(content, args.output_path, args.source_filename)
