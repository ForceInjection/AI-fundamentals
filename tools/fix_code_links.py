import os
import re

def fix_code_links(base_dir):
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', 'node_modules']]
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Replace vLLM links
                # e.g. ../vllm/v1/... -> https://github.com/vllm-project/vllm/blob/main/vllm/v1/...
                # Also ../../vllm/v1/... -> https://github.com/vllm-project/vllm/blob/main/vllm/v1/...
                content = re.sub(r'\]\((?:\.\./)+vllm/([^)]+\.py(?:#[^)]*)?)\)', 
                               r'](https://github.com/vllm-project/vllm/blob/main/vllm/\1)', content)
                
                # Replace LMCache links
                # e.g. ../lmcache/v1/... -> https://github.com/LMCache/LMCache/blob/main/lmcache/v1/...
                content = re.sub(r'\]\((?:\.\./)+lmcache/([^)]+\.py(?:#[^)]*)?)\)', 
                               r'](https://github.com/LMCache/LMCache/blob/main/lmcache/\1)', content)

                # Replace TGI links
                content = re.sub(r'\]\((?:\.\./)+text-generation-inference/([^)]+)\)',
                               r'](https://github.com/huggingface/text-generation-inference/blob/main/\1)', content)

                # Replace vllm-project/vllm cpp/h/cu links
                content = re.sub(r'\]\((?:\.\./)+csrc/([^)]+)\)',
                               r'](https://github.com/vllm-project/vllm/blob/main/csrc/\1)', content)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    fix_code_links(base_dir)
