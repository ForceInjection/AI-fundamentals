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
                
                # Replace pseudo code links
                # e.g. [performance_test.c](code/testing/performance/performance_test.c:34-216)
                # -> `performance_test.c`
                new_content = re.sub(r'\[([^\]]+)\]\((?:code|lib|../lib|../../vllm|../vllm|../../lmcache|../lmcache|\./kv_cache_manager|/gpu_manager|lmcache)/[^)]+\)', r'`\1`', content)
                # handle `edge_id` -> `state`
                new_content = new_content.replace('[edge_id](state)', '`edge_id`(state)')
                # handle '\1' -> '\2'
                new_content = new_content.replace(r"['\1']('\2')", r"`\1`('\2')")
                new_content = new_content.replace(r'[\1](\2)', r'`\1`(\2)')
                # handle `["'` -> `[^"']+'`
                new_content = new_content.replace(r'["\']([^"\']+)', r'`["\']([^"\']+)`')
                # also fix message_type
                new_content = new_content.replace('[message_type](sender_id, message)', '`message_type` (sender_id, message)')
                new_content = new_content.replace('[action](message.payload.parameters)', '`action` (message.payload.parameters)')
                
                # fix expert_id -> x[mask]
                new_content = new_content.replace('[expert_id](x[mask])', '`expert_id(x[mask])`')

                if new_content != content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    fix_code_links(base_dir)
